import numpy as np

from sklearn.metrics import roc_auc_score
from matchzoo.engine.base_metric import ClassificationMetric, RankingMetric


class AUCScore(ClassificationMetric, RankingMetric):
    """AUC metric."""

    ALIAS = ['auc']

    def __init__(self, threshold):
        """:class:`auc` constructor."""
        self._threshold = int(threshold)

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def __call__(self, y_true: np.array, y_pred: np.array) -> float:
        """
        Calculate auc score.

        Example:
            >>> import numpy as np
            >>> y_true = np.array([1])
            >>> y_pred = np.array([[0, 1]])
            >>> AUCScore()(y_true, y_pred)
            1.0

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: AUCScore.
        """
        y_pred = 1 / (1 + np.exp(-y_pred))  # sigmoid activation
        y_true = np.where(y_true > self._threshold, 1.0, 0.0)
        return roc_auc_score(y_true, y_pred)
