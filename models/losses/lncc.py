from __future__ import annotations

import math

import torch
import torch.nn as nn
from monai.networks.layers import gaussian_1d, separable_filtering
from monai.utils import LossReduction
from monai.utils.module import look_up_option
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from ..builder import CFG, LOSSES


def make_rectangular_kernel(kernel_size: int) -> torch.Tensor:
    return torch.ones(kernel_size)


def make_triangular_kernel(kernel_size: int) -> torch.Tensor:
    fsize = (kernel_size + 1) // 2
    if fsize % 2 == 0:
        fsize -= 1
    f = torch.ones((1, 1, fsize), dtype=torch.float).div(fsize)
    padding = (kernel_size - fsize) // 2 + fsize // 2
    return F.conv1d(f, f, padding=padding).reshape(-1)


def make_gaussian_kernel(kernel_size: int) -> torch.Tensor:
    sigma = torch.tensor(kernel_size / 3.0)
    kernel = gaussian_1d(sigma=sigma,
                         truncated=kernel_size // 2,
                         approx='sampled',
                         normalize=False) * (2.5066282 * sigma)
    return kernel[:kernel_size]


kernel_dict = {
    'rectangular': make_rectangular_kernel,
    'triangular': make_triangular_kernel,
    'gaussian': make_gaussian_kernel,
}


@LOSSES.register_module('lncc')
class LocalNormalizedCrossCorrelationLoss(_Loss):
    """Local squared zero-normalized cross-correlation. The loss is based on a moving
    kernel/window over the y_true/y_pred, within the window the square of zncc is
    calculated. The kernel can be a rectangular / triangular / gaussian window. The
    final loss is the averaged loss over all windows.

    Adapted from:DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    def __init__(
            self,
            spatial_dims: int = 3,
            kernel_size: int = 3,
            kernel_type: str = 'rectangular',
            reduction: LossReduction | str = LossReduction.MEAN,
            smooth_nr: float = 0.0,
            smooth_dr: float = 1e-5,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, {``1``, ``2``, ``3``}. Defaults to 3.
            kernel_size: kernel spatial size, must be odd.
            kernel_type: {``"rectangular"``, ``"triangular"``, ``"gaussian"``}. Defaults to ``"rectangular"``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            smooth_nr: a small constant added to the numerator to avoid nan.
            smooth_dr: a small constant added to the denominator to avoid nan.

        """
        super().__init__(reduction=LossReduction(reduction).value)

        self.ndim = spatial_dims
        if self.ndim not in {1, 2, 3}:
            raise ValueError(
                f'Unsupported ndim: {self.ndim}-d, only 1-d, 2-d, and 3-d inputs are supported'
            )

        self.kernel_size = kernel_size
        if self.kernel_size % 2 == 0:
            raise ValueError(
                f'kernel_size must be odd, got {self.kernel_size}')

        _kernel = look_up_option(kernel_type, kernel_dict)
        self.kernel = _kernel(self.kernel_size)
        self.kernel.requires_grad_(False)
        self.kernel_vol = self.get_kernel_vol()

        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def get_kernel_vol(self):
        vol = self.kernel
        for _ in range(self.ndim - 1):
            vol = torch.matmul(vol.unsqueeze(-1), self.kernel.unsqueeze(0))
        return torch.sum(vol)

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if pred.ndim - 2 != self.ndim:
            raise ValueError(
                f'expecting pred with {self.ndim} spatial dimensions, got pred of shape {pred.shape}'
            )
        if target.shape != pred.shape:
            raise ValueError(
                f'ground truth has differing shape ({target.shape}) from pred ({pred.shape})'
            )

        t2, p2, tp = target * target, pred * pred, target * pred
        kernel, kernel_vol = self.kernel.to(pred), self.kernel_vol.to(pred)
        kernels = [kernel] * self.ndim
        # sum over kernel
        t_sum = separable_filtering(target, kernels=kernels)
        p_sum = separable_filtering(pred, kernels=kernels)
        t2_sum = separable_filtering(t2, kernels=kernels)
        p2_sum = separable_filtering(p2, kernels=kernels)
        tp_sum = separable_filtering(tp, kernels=kernels)

        # average over kernel
        t_avg = t_sum / kernel_vol
        p_avg = p_sum / kernel_vol

        # normalized cross correlation between t and p
        # sum[(t - mean[t]) * (p - mean[p])] / std[t] / std[p]
        # denoted by num / denom
        # assume we sum over N values
        # num = sum[t * p - mean[t] * p - t * mean[p] + mean[t] * mean[p]]
        #     = sum[t*p] - sum[t] * sum[p] / N * 2 + sum[t] * sum[p] / N
        #     = sum[t*p] - sum[t] * sum[p] / N
        #     = sum[t*p] - sum[t] * mean[p] = cross
        # # the following is actually squared ncc
        # cross = tp_sum - p_avg * t_sum
        #
        # # different from monai implementation
        # t_var = t2_sum - t_avg * t_sum
        # p_var = p2_sum - p_avg * p_sum

        cross = tp_sum - p_avg * t_sum - t_avg * p_sum + t_avg * p_avg * kernel_vol
        t_var = t2_sum - 2 * t_avg * t_sum + t_avg * t_avg * kernel_vol
        p_var = p2_sum - 2 * p_avg * p_sum + p_avg * p_avg * kernel_vol

        ncc: torch.Tensor = (cross * cross + self.smooth_nr) / (t_var * p_var +
                                                                self.smooth_dr)

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(
                ncc).neg()  # sum over the batch, channel and spatial ndims
        if self.reduction == LossReduction.NONE.value:
            return ncc.neg()
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(
                ncc).neg()  # average over the batch, channel and spatial ndims
        raise ValueError(
            f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
        )


@LOSSES.register_module('ncc')
class NCC(nn.Module):
    """Local (over window) normalized cross correlation loss."""

    def __init__(self,
                 spatial_dims: int = 3,
                 kernel_size: int = 9,
                 smooth_nr: float = 0.0,
                 smooth_dr: float = 1e-5) -> None:
        super(NCC, self).__init__()
        self.ndim = spatial_dims
        if self.ndim not in {1, 2, 3}:
            raise ValueError(
                f'Unsupported ndim: {self.ndim}-d, only 1-d, 2-d, and 3-d inputs are supported'
            )

        self.kernel_size = kernel_size
        if self.kernel_size % 2 == 0:
            raise ValueError(
                f'kernel_size must be odd, got {self.kernel_size}')

        self.kernel = torch.ones([1, 1, *([self.kernel_size] * self.ndim)])
        self.stride = [1] * self.ndim
        self.padding = [math.floor(self.kernel_size / 2)] * self.ndim
        self.kernel.requires_grad_(False)
        self.kernel_vol = self.kernel.sum().squeeze()

        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

        self.conv = getattr(F, 'conv%dd' % self.ndim)

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:

        if pred.ndim - 2 != self.ndim:
            raise ValueError(
                f'expecting pred with {self.ndim} spatial dimensions, got pred of shape {pred.shape}'
            )
        if target.shape != pred.shape:
            raise ValueError(
                f'ground truth has differing shape ({target.shape}) from pred ({pred.shape})'
            )

        t2, p2, tp = target * target, pred * pred, target * pred
        kernel, kernel_vol = self.kernel.to(pred), self.kernel_vol.to(pred)

        # sum over kernel
        t_sum = self.conv(target,
                          kernel,
                          stride=self.stride,
                          padding=self.padding)
        p_sum = self.conv(pred,
                          kernel,
                          stride=self.stride,
                          padding=self.padding)
        t2_sum = self.conv(t2,
                           kernel,
                           stride=self.stride,
                           padding=self.padding)
        p2_sum = self.conv(p2,
                           kernel,
                           stride=self.stride,
                           padding=self.padding)
        tp_sum = self.conv(tp,
                           kernel,
                           stride=self.stride,
                           padding=self.padding)

        # average over kernel
        t_avg = t_sum / kernel_vol
        p_avg = p_sum / kernel_vol

        # the following is actually squared ncc
        cross = tp_sum - p_avg * t_sum

        # different from monai implementation
        t_var = t2_sum - t_avg * t_sum
        p_var = p2_sum - p_avg * p_sum

        # cross = tp_sum - p_avg * t_sum - t_avg * p_sum + t_avg * p_avg * kernel_vol
        # t_var = t2_sum - 2 * t_avg * t_sum + t_avg * t_avg * kernel_vol
        # p_var = p2_sum - 2 * p_avg * p_sum + p_avg * p_avg * kernel_vol

        ncc: torch.Tensor = (cross * cross + self.smooth_nr) / (t_var * p_var +
                                                                self.smooth_dr)

        return torch.mean(ncc).neg()


@LOSSES.register_module('enn_ncc')
class ENN_NCC(nn.Module):
    """Local (over window) normalized cross correlation loss."""

    def __init__(self,
                 spatial_dims: int = 3,
                 kernel_size: int = 9,
                 smooth_nr: float = 0.0,
                 smooth_dr: float = 1e-5) -> None:
        super(ENN_NCC, self).__init__()
        self.ndim = spatial_dims
        if self.ndim not in {1, 2, 3}:
            raise ValueError(
                f'Unsupported ndim: {self.ndim}-d, only 1-d, 2-d, and 3-d inputs are supported'
            )

        self.kernel_size = kernel_size
        if self.kernel_size % 2 == 0:
            raise ValueError(
                f'kernel_size must be odd, got {self.kernel_size}')

        self.kernel = torch.ones([1, 1, *([self.kernel_size] * self.ndim)])
        self.stride = [1] * self.ndim
        self.padding = [math.floor(self.kernel_size / 2)] * self.ndim
        self.kernel.requires_grad_(False)
        self.kernel_vol = self.kernel.sum().squeeze()

        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

        self.conv = getattr(F, 'conv%dd' % self.ndim)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if pred.ndim - 2 != self.ndim:
            raise ValueError(
                f'expecting pred with {self.ndim} spatial dimensions, got pred of shape {pred.shape}'
            )
        if target.shape != pred.shape:
            raise ValueError(
                f'ground truth has differing shape ({target.shape}) from pred ({pred.shape})'
            )

        t2, p2, tp = target * target, pred * pred, target * pred
        kernel, kernel_vol = self.kernel.to(pred), self.kernel_vol.to(pred)

        # sum over kernel
        t_sum = self.conv(target,
                          kernel,
                          stride=self.stride,
                          padding=self.padding)
        p_sum = self.conv(pred,
                          kernel,
                          stride=self.stride,
                          padding=self.padding)
        t2_sum = self.conv(t2,
                           kernel,
                           stride=self.stride,
                           padding=self.padding)
        p2_sum = self.conv(p2,
                           kernel,
                           stride=self.stride,
                           padding=self.padding)
        tp_sum = self.conv(tp,
                           kernel,
                           stride=self.stride,
                           padding=self.padding)

        # average over kernel
        t_avg = t_sum / kernel_vol
        p_avg = p_sum / kernel_vol

        # the following is actually squared ncc
        cross = tp_sum - p_avg * t_sum

        # different from monai implementation
        t_var = t2_sum - t_avg * t_sum
        p_var = p2_sum - p_avg * p_sum

        # cross = tp_sum - p_avg * t_sum - t_avg * p_sum + t_avg * p_avg * kernel_vol
        # t_var = t2_sum - 2 * t_avg * t_sum + t_avg * t_avg * kernel_vol
        # p_var = p2_sum - 2 * p_avg * p_sum + p_avg * p_avg * kernel_vol

        ncc: torch.Tensor = (cross * cross + self.smooth_nr) / (t_var * p_var +
                                                                self.smooth_dr)

        return 1 - ncc


@LOSSES.register_module('enn_cum_ncc')
class ENN_CUM_NCC(nn.Module):
    """LNCC implemented with cumulative sum."""
    def __init__(self,
                 spatial_dims: int = 3,
                 kernel_size: int = 9,
                 smooth_nr: float = 0.0,
                 smooth_dr: float = 1e-5) -> None:
        super(ENN_CUM_NCC, self).__init__()
        assert spatial_dims == 3, 'Only 3D inputs are supported'
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

        self.win_size = kernel_size

    @staticmethod
    def compute_window_sum(volume: torch.Tensor, window_size: int):
        radius = window_size // 2
        padding = [radius + 1, radius] * 3

        padded_volume = F.pad(volume, pad=padding, mode='constant', value=0)

        # Compute cumulative sums along each dimension
        cum_sum_x = torch.cumsum(padded_volume, dim=2)
        cum_sum_xy = torch.cumsum(cum_sum_x, dim=3)
        cum_sum_xyz = torch.cumsum(cum_sum_xy, dim=4)

        depth, height, width = volume.shape[2:]

        # Calculate window sum using inclusion-exclusion principle
        window_sum = (
                cum_sum_xyz[:, :, window_size:, window_size:, window_size:]
                - cum_sum_xyz[:, :, window_size:, window_size:, :width]
                - cum_sum_xyz[:, :, window_size:, :height, window_size:]
                - cum_sum_xyz[:, :, :depth, window_size:, window_size:]
                + cum_sum_xyz[:, :, window_size:, :height, :width]
                + cum_sum_xyz[:, :, :depth, window_size:, :width]
                + cum_sum_xyz[:, :, :depth, :height, window_size:]
                - cum_sum_xyz[:, :, :depth, :height, :width]
        )

        return window_sum

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        I = pred.double()
        J = target.double()

        I2 = I * I
        J2 = J * J
        IJ = I * J

        # compute local sums via cumsum trick
        I_sum_cs = self.compute_window_sum(I, self.win_size)
        J_sum_cs = self.compute_window_sum(J, self.win_size)
        I2_sum_cs = self.compute_window_sum(I2, self.win_size)
        J2_sum_cs = self.compute_window_sum(J2, self.win_size)
        IJ_sum_cs = self.compute_window_sum(IJ, self.win_size)

        win_vol = (self.win_size * 1.) ** 3

        mean_I = I_sum_cs / win_vol
        mean_J = J_sum_cs / win_vol

        cross = IJ_sum_cs - mean_J * I_sum_cs - mean_I * J_sum_cs + mean_I * mean_J * win_vol
        var_I = I2_sum_cs - 2 * mean_I * I_sum_cs + mean_I * mean_I * win_vol
        var_J = J2_sum_cs - 2 * mean_J * J_sum_cs + mean_J * mean_J * win_vol

        ncc_squared = (cross * cross + self.smooth_nr) / (var_I * var_J + self.smooth_dr)
        return (1.0 - ncc_squared.float())


@LOSSES.register_module('ncc_vxm')
class NCC_VXM(torch.nn.Module):
    """Local (over window) normalized cross correlation loss."""

    def __init__(self, win=None):
        super(NCC_vxm, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):
        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [
            1, 2, 3
        ], 'volumes should be 1 to 3 dimensions. found: %d' % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to('cuda')

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)
