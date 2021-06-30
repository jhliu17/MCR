"""The rank cross entropy loss."""
import torch
from torch import nn
import torch.nn.functional as F


class PointCrossEntropyLoss(nn.Module):
    """Creates a criterion that measures point cross entropy loss."""

    __constants__ = ['num_neg']

    def __init__(self, threshold: int):
        """
        :class:`PointCrossEntropyLoss` constructor.
        """
        super().__init__()
        self._threshold = threshold

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Calculate point cross entropy loss.

        :param y_pred: Predicted result.
        :param y_true: Label.
        :return: Point cross loss.
        """
        y_true = (y_true > self._threshold).float()
        return F.binary_cross_entropy_with_logits(
            y_pred,
            y_true
        )
