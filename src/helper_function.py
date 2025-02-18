import hashlib
import os
from random import randint
from typing import Iterable

import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from numpy.random import RandomState
from requests import Session as requests_session
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline
from torch import Tensor

import wandb
from joblib import Memory
from src.global_vars import (
    BASE_RUN_DIR,
    HTTP_ADAPTER_FOR_REQUESTS,
)


# TODO make it general
def tsv_to_csv():
    # Initialize a flag to handle headers
    write_header = False
    i = 0
    # Read TSV in chunks and write to CSV in append mode
    with pd.read_table(
        "data/american_gut_project_ERP012803_taxonomy_abundances_SSU_v5.0.tsv",
        chunksize=500,
    ) as reader:
        for chunk in reader:
            if i < 5:
                i += 1
                continue
            chunk.to_csv(
                "american_gut_project_ERP012803_taxonomy_abundances_SSU_v5.csv",
                mode="a",
                index=False,
                header=write_header,
            )
            # After the first chunk, do not write the header
            write_header = False


def hasher(iterator: Iterable) -> str:
    """Hashes the elements of an iterator in a deterministic way."""
    iterator_list = list(iterator)
    iterator_list.sort()
    iterator_str = str(iterator_list)
    return hashlib.md5(iterator_str.encode()).hexdigest()


def config_session(session: requests_session):
    session.mount("http://", HTTP_ADAPTER_FOR_REQUESTS)
    session.mount("https://", HTTP_ADAPTER_FOR_REQUESTS)
    session.headers.update(
        {
            "Connection": "keep-alive",
            "User-Agent": "Mozilla/5.0 (X11; CrOS x86_64 12871.102.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.141 Safari/537.36",
        }
    )


def df_str_for_loguru(df: pd.DataFrame) -> str:
    """Returns a string representation of a DataFrame that is suitable for loguru logging, with #rows and #columns limited to 6.

    Args:
        df: DataFrame to be converted to string.

    Returns:
        String representation of the DataFrame.
    """
    df_str = df.to_string(max_cols=6, max_rows=6, max_colwidth=15)
    return "\n\t" + df_str.replace("\n", "\n\t")


def create_pipeline(steps: list[object], config: dict[str, object]) -> Pipeline:
    """Create a pipeline from the steps and config."""
    # Create a cache object if caching is enabled with default cache location
    if config.get("cache_pipeline_steps", True):
        cacher = Memory(
            location=get_run_dir_for_experiment(config) / "pipeline_cache", verbose=1
        )
    else:
        cacher = None

    return Pipeline(steps, memory=cacher, verbose=config.get("verbose_pipeline", True))


def get_run_dir_for_experiment(config: dict[str, object]):
    run_dir = BASE_RUN_DIR / config["wandb_params"]["name"]
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
    return run_dir


# def get_data_dir_for_experiment(config: dict[str, object]):
#     data_dir = BASE_DATA_DIR / config["wandb_params"]["name"]
#     data_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
#     return data_dir


def get_scores(
    model: Pipeline | BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    scoring: dict,
    train_or_test: str,
) -> dict:
    """Calculate the scores for the predictions."""
    scores = {}
    y_n_unique = len(np.unique(y))

    y_pred = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    y_dec = model.decision_function(X) if hasattr(model, "decision_function") else None
    y_pred = model.predict(X)

    for score_name, scorer in scoring.items():
        try:
            scoring_function = get_scorer(scorer)._score_func
            if "roc_auc" in score_name or "average_precision" in score_name:
                kwargs = scorer._kwargs if hasattr(scorer, "_kwargs") else {}
                # Get kwargs available, e.g. {average="micro"} if scorer is not a string but a make_scorer object
                if y_pred is not None:
                    y_model = y_pred
                    scoring_function._response_method = "predict_proba"
                elif y_dec is not None:
                    y_model = y_dec
                    scoring_function._response_method = "decision_function"
                else:
                    y_model = y_pred
                    scoring_function._response_method = "predict"

                if (
                    y_n_unique == 2 and y_pred is not None
                ):  # [:, 1] needed somewhere, but this seems to work and [:, 0] does not
                    scores[train_or_test + "/" + score_name] = scoring_function(
                        y, y_model, **kwargs
                    )
                else:
                    scores[train_or_test + "/" + score_name] = scoring_function(
                        y, y_model, **kwargs
                    )
            else:
                y_pred = y_pred
                scores[train_or_test + "/" + score_name] = scoring_function(y, y_pred)
        except Exception as e:
            logger.error(f"Error calculating {score_name} for {train_or_test}: {e}")
            scores[train_or_test + "/" + score_name] = None

    return scores


def plotly_bar_plot_with_error_bars(df: pd.DataFrame, title: str) -> go.Figure:
    """Create a bar plot with error bars using Plotly.

    Args:
        df: DataFrame containing the data to be plotted. it should have columns "Metric", "Mean", and "Std".
        title: Title of the plot.
    """
    fig = go.Figure(
        data=[
            go.Bar(
                x=df["Metric"],
                y=df["Mean"],
                error_y=dict(type="data", array=df["Std"], visible=True),
                marker=dict(color="blue"),
                name=title,
                orientation="h",
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Metric",
        yaxis_title="Mean Value",
        barmode="group",
        template="plotly_white",
    )

    return fig


def is_cluster_environment():
    """Detect whether the script is running in a SLURM cluster environment."""
    return "SLURM_JOB_ID" in os.environ


def circular_slice(arr: Iterable, start: int, end: int) -> Iterable:
    """Slice a list with wrap-around.

    Args:
        arr: The list to slice.
        start: The start index of the slice.
        end: The end index of the slice.

    Returns:
        The sliced list.
    """
    n = len(arr)
    if n == 0:
        return np.array([])  # Handle empty array case

    indices = np.arange(start, end) % n  # Compute wrapped indices
    return arr[indices]  # Direct NumPy indexing


def metalearning_binary_target_changer(labels: Tensor) -> Tensor:
    """Change the binary labels randomly.

    Args:
        labels: The binary labels to change.

    Returns:
        The changed binary labels.
    """
    to_change = randint(0, 1)
    labels = (labels + to_change) % 2
    return labels

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr