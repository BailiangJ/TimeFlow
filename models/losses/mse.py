from __future__ import annotations

import torch
import torch.nn as nn
from monai.utils import LossReduction
from torch.nn.modules.loss import _Loss

from ..builder import LOSSES


@LOSSES.register_module('mse')
class MeanSquaredErrorLoss(_Loss):
    """Mean Squared Error (MSE) Loss."""

    def __init__(
        self,
        reduction: LossReduction | str = LossReduction.MEAN,
    ) -> None:
        """
        Args:
            reduction: {``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
        """
        super().__init__(reduction=LossReduction(reduction).value)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if target.shape != pred.shape:
            raise ValueError(
                f'ground truth has differing shape ({target.shape}) from pred ({pred.shape})'
            )

        mse = (pred - target) ** 2

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(mse)  # sum over the batch, channel and spatial ndims
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(mse)  # average over the batch, channel and spatial ndims
        raise ValueError(
            f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum"].'
        )