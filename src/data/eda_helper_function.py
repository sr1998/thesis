import os
from pathlib import Path
from shutil import rmtree

from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from fire import Fire
from matplotlib import pyplot as plt
from pandas.io.parsers import read_csv as _read_csv
from pandas.io.parsers import read_table as _read_txt
from seaborn import heatmap as sns_heatmap

from joblib import Memory
from src.global_vars import BASE_DIR
from src.sparcc.SparCC import apply_sparcc


def sparcc_correlation(
    data: pd.DataFrame, cache_dir: str, verbose: bool = False, log: bool = False
):
    sparcc_corr_dir = BASE_DIR / "data" / "sparcc" / "correlation"
    sparcc_cov_dir = BASE_DIR / "data" / "sparcc" / "covariance"
    sparcc_corr_dir.mkdir(parents=True, exist_ok=True)
    sparcc_cov_dir.mkdir(parents=True, exist_ok=True)

    mem = Memory(cache_dir)
    sparcc_w_mem = mem.cache(apply_sparcc, ignore=["verbose", "log"])
    cor, cov = sparcc_w_mem(
        frame=data,
        method="sparcc",
        norm="dirichlet",
        n_iter=20,
        verbose=verbose,
        log=log,
        th=0.1,
        x_iter=10,
        path_subdir_cor=str(sparcc_corr_dir),
        path_subdir_cov=str(sparcc_cov_dir),
    )

    if sparcc_corr_dir.exists():
        rmtree(sparcc_corr_dir)
    if sparcc_cov_dir.exists():
        rmtree(sparcc_cov_dir)

    return cor, cov


# TODO p-values for correlation can also be found if we want


def plot_heatmap(data: np.ndarray, columns: list):
    # Generate the heatmap
    fig, ax = plt.subplots(figsize=(12, 12))  # Adjust the figure size as needed

    # Plot the heatmap with matplotlib
    im = ax.imshow(data, cmap="RdBu", vmin=-1, vmax=1)

    # Add color bar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("SparCC Correlation", rotation=270, labelpad=15)

    # Set axis labels and ticks
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(columns, fontsize=8)

    # Set title
    ax.set_title("SparCC Correlation", fontsize=16)

    # Adjust layout to fit axis labels
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_jaccard_similarities(
    dir_path: str | Path,
    title: str,
    column_name: str,
    biome_dict: dict | None = None,
    file_ending: str = ".tsv",
    axis: Axes | None = None,
):
    """..."""
    # get dataframe for each file in dir_path ending with file_ending and only column column_name
    data = {}
    for file in os.listdir(dir_path):
        if file.endswith(file_ending):
            data[file] = pd.read_table(
                os.path.join(dir_path, file), usecols=[column_name]
            )

    # triangular jaccard similarity of descriptions
    jaccard_similarities = np.zeros((len(data), len(data)))
    for i, (file1, df1) in enumerate(data.items()):
        for j, (file2, df2) in enumerate(data.items()):
            if i > j:
                continue
            descriptions1 = set(df1[column_name])
            descriptions2 = set(df2[column_name])
            jaccard_similarities[i, j] = len(
                descriptions1.intersection(descriptions2)
            ) / len(descriptions1.union(descriptions2))

    # plot heatmap
    xticks = [file.split("_")[0] for file in data.keys()]
    if biome_dict:
        try:
            biome_dict = {k: v.split(":")[-1] for k, v in biome_dict.items()}
            xticks = [x + " (" + biome_dict[x] + ")" for x in xticks]
        except KeyError:
            print("Biome_dict not compatible with xticks")

    sns_heatmap(
        jaccard_similarities,
        xticklabels=xticks,
        yticklabels=xticks,
        cmap="YlGnBu",
        annot=True,
        ax=axis,
    )
    plt.title(title)
