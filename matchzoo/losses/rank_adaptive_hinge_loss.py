"""The rank hinge loss."""
import torch
from torch import nn


class AdaptiveRankHingeLoss(nn.Module):
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

    def __init__(self, num_neg: int = 1, reduction: str = 'mean'):
        """
        :class:`RankHingeLoss` constructor.

        :param num_neg: Number of negative instances in hinge loss.
        :param margin: Margin between positive and negative scores.
            Float. Has a default value of :math:`0`.
        :param reduction: String. Specifies the reduction to apply to
            the output: ``'none'`` | ``'mean'`` | ``'sum'``.
            ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the
                number of elements in the output,
            ``'sum'``: the output will be summed.
        """
        super().__init__()
        self.num_neg = num_neg
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Calculate rank hinge loss.

        :param y_pred: Predicted result.
        :param y_true: Label.
        :return: Hinge loss computed by user-defined margin.
        """
        y_pred_pos, y_pred_neg = self.get_part_tensor(y_pred)
        y_true_pos, y_true_neg = self.get_part_tensor(y_true)

        y_pred_diff = y_pred_pos - y_pred_neg  # [B, num_neg]
        y_true_diff = y_true_pos - y_true_neg
        loss = y_true_diff - y_pred_diff
        mask = loss < 0
        loss = loss.masked_fill(mask, 0)
        loss = loss.sum(dim=-1).mean()
        return loss

    def get_part_tensor(self, y):
        y_pos = y[::(self.num_neg + 1), :]  # [B, 1]
        y_neg = []
        for neg_idx in range(self.num_neg):
            neg = y[(neg_idx + 1)::(self.num_neg + 1), :]
            y_neg.append(neg)
        y_neg = torch.cat(y_neg, dim=-1)  # [B, num_neg]
        return y_pos, y_neg

    @property
    def num_neg(self):
        """`num_neg` getter."""
        return self._num_neg

    @num_neg.setter
    def num_neg(self, value):
        """`num_neg` setter."""
        self._num_neg = value
