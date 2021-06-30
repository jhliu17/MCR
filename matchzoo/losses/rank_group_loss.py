"""The rank hinge loss."""
import torch
from torch import nn
import torch.nn.functional as F


class GroupwiseLoss(nn.Module):
    """
    Creates a criterion that measures rank hinge loss.

    Given inputs :math:`x1`, :math:`x2`, two 1D mini-batch `Tensors`,
    and a label 1D mini-batch tensor :math:`y` (containing 1 or -1).

    If :math:`y = 1` then it assumed the first input should be ranked
    higher (have a larger value) than the second input, and vice-versa
    for :math:`y = -1`.

    The loss function for each sample in the mini-batch is:

    .. math::
        loss_{x, y} = max(0, -y * (x1 - x2) + margin)
    """

    __constants__ = ['num_neg', 'margin', 'reduction']

    def __init__(self, group_size: int, reduction: str = 'mean'):
        """
        :class:`GroupwiseLoss` constructor.

        :param group_size: Number of instances in groupwise loss.
        :param reduction: String. Specifies the reduction to apply to
            the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the
                number of elements in the output,
            ``'sum'``: the output will be summed.
        """
        super().__init__()
        self.group_size = group_size
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Calculate rank hinge loss.

        :param y_pred: Predicted result.
        :param y_true: Label.
        :return: Groupwise KL loss.
        """
        group_num = int(y_pred.size(0) / self.group_size)
        y_pred = y_pred.view(group_num, self.group_size, -1)
        y_true = y_true.view(group_num, self.group_size, -1)
        normalized_y_pred = F.softmax(y_pred, dim=1)
        normalized_y_true = F.softmax(y_true, dim=1)

        return F.kl_div(
            normalized_y_pred, normalized_y_true,
            reduction=self.reduction
        )

    @property
    def group_size(self):
        """`group_size` getter."""
        return self._group_size

    @group_size.setter
    def group_size(self, value):
        """`group_size` setter."""
        self._group_size = value
