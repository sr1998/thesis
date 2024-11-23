import pandas as pd


def total_sum_scaling(data: pd.DataFrame) -> pd.DataFrame:
    """
        Normalize the data by dividing each feature count of a sample by the total count of that sample.

        Args:
            data (pd.DataFrame): The data to be normalized. Each row is a sample and each column is a feature.
    """
    row_sums = data.sum(axis=1)
    row_sums = row_sums.replace(0, pd.NA)
    normalized_data = data.div(row_sums, axis=0)
    normalized_data = normalized_data.fillna(0)
    return normalized_data