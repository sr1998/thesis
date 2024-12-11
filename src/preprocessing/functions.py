import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NumpyReplace(TransformerMixin, BaseEstimator):
    def __init__(self, replace: int = None):
        self.replace = replace

    def fit(self, X, y=None):
        if self.replace is None:
            self.replace = np.min(X[X.nonzero()])

    def transform(self, X):
        return np.place(X, X == 0, self.replace)


def total_sum_scaling(data: np.ndarray | pd.DataFrame) -> np.ndarray:
    """
    Normalize the data by dividing each feature count of a sample by the total count of that sample.
    Args:
        data: The data to be normalized. Each row is a sample and each column is a feature.
    """
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    
    row_sums = data.sum(axis=1, keepdims=True)
    normalized_data = np.divide(data, row_sums + np.isclose(row_sums, 0))
    return normalized_data
