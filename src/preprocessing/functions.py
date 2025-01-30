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


def total_sum_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Total sum scaling of a dataframe.

    Normalizes the dataframe by the sum of each row.

    Args:
        df: The dataframe to normalize.
    """
    row_sums = df.sum(axis=1)
    df = df.div(row_sums + (row_sums == 0), axis=0)
    return df


def centered_arcsine_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Centered arcsine transform of a dataframe.

    Assumes that the dataframe contains proportions of some kind, i.e. values in [0, 1].
    """

    def arcsine(x):
        return np.arcsin(np.sqrt(x))

    r, c = df.shape
    C = np.eye(c) - 1 / c * np.ones((c, c))
    df_vals = df.to_numpy()
    df_vals = arcsine(df_vals)
    df_vals = df_vals @ C
    df = pd.DataFrame(df_vals, index=df.index, columns=df.columns)
    return df


def pandas_label_encoder(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the labels of a dataframe. All columns of type "object" are encoded.

    Args:
        df: The dataframe to encode.

    Returns:
        The encoded dataframe.

    """
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col])

    return df
