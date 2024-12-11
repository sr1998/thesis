#!/usr/bin/env python3

# source: https://github.com/maxibor/microbiomeNormalization/blob/master/micronorm

import multiprocessing
from functools import partial

import numpy as np
import pandas as pd


# TODO we can optimize this function
# ij is same as ji when getting rij, so we can optimize this. See paper
def gmpr_size_factor(col, ar):
    pr = np.apply_along_axis(lambda x: np.divide(ar[:, col], x), 0, ar)
    pr[np.isinf(pr)] = np.nan
    pr[pr == 0] = np.nan
    pr_median = np.nanmedian(pr, axis=0)
    return np.exp(np.mean(np.log(pr_median)))


def GMPR_normalize(df: pd.DataFrame | np.ndarray, process: int = None):
    """
    Global Mean of Pairwise Ratios
    Chen, L., Reeve, J., Zhang, L., Huang, S., Wang, X., & Chen, J. (2018).
    GMPR: A robust normalization method for zero-inflated count data
    with application to microbiome sequencing data.
    PeerJ, 6, e4600.

    Use without imputation
    """
    if isinstance(df, pd.DataFrame):
        ar = np.asarray(df)
    else:
        ar = df

    ar = ar.transpose()  # samples in columns needed

    # added this
    if process is None:
        process = min(multiprocessing.cpu_count(), df.shape[1])

    gmpr_sf_partial = partial(gmpr_size_factor, ar=ar)
    with multiprocessing.Pool(process) as p:
        sf = p.map(gmpr_sf_partial, list(range(np.shape(ar)[1])))

    return pd.DataFrame(
        np.divide(ar, sf).transpose(), index=df.index, columns=df.columns
    )  # transpose back
