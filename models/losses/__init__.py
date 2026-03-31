from monai.losses import DiceLoss

from ..builder import LOSSES
from .diffusion_regularizer import GradientDiffusionLoss
from .flow_loss import FlowLoss
from .icon import GradICONLoss, ICONLoss
from .inverse_consistency import InverseConsistentLoss
from .kernels import (average_kernel_1d, average_kernel_2d, average_kernel_3d,
                     gauss_kernel_1d, gauss_kernel_2d, gauss_kernel_3d,
                     gradient_kernel_1d, gradient_kernel_2d, gradient_kernel_3d,
                     spatial_filter_nd)
from .lncc import LocalNormalizedCrossCorrelationLoss
from .long_constraint import LongitudinalConsistentLoss
from .long_icon import CompositionConsistencyLoss, GradCompositionConsistencyLoss
from .mse import MeanSquaredErrorLoss
from .np_jacdet import NonPositiveJacDetLoss

# Register MONAI losses
LOSSES.register_module('dice_loss', module=DiceLoss)

__all__ = [
    'GradientDiffusionLoss',
    'FlowLoss',
    'InverseConsistentLoss',
    'ICONLoss',
    'GradICONLoss',
    'LongitudinalConsistentLoss',
    'CompositionConsistencyLoss',
    'GradCompositionConsistencyLoss',
    'LocalNormalizedCrossCorrelationLoss',
    'MeanSquaredErrorLoss',
    'NonPositiveJacDetLoss',
    'spatial_filter_nd',
    'gauss_kernel_1d',
    'gauss_kernel_2d',
    'gauss_kernel_3d',
    'average_kernel_1d',
    'average_kernel_2d',
    'average_kernel_3d',
    'gradient_kernel_1d',
    'gradient_kernel_2d',
    'gradient_kernel_3d',
]
