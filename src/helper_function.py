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
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from torch import Tensor

import wandb
from joblib import Memory, Parallel, delayed
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
    score_name_prefix: str,
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
                    scores[score_name_prefix + "/" + score_name] = scoring_function(
                        y, y_model, **kwargs
                    )
                else:
                    scores[score_name_prefix + "/" + score_name] = scoring_function(
                        y, y_model, **kwargs
                    )
            else:
                y_pred = y_pred
                scores[score_name_prefix + "/" + score_name] = scoring_function(
                    y, y_pred
                )
        except Exception as e:
            logger.error(f"Error calculating {score_name} for {score_name_prefix}: {e}")
            scores[score_name_prefix + "/" + score_name] = None

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


def get_pipeline(what, standard_pipeline, search_space_sampler, optuna_trial):
    """Get the pipeline with the hyperparameters sampled from the search space."""
    trial_config = search_space_sampler(optuna_trial)

    if what == "mgnify":
        n_neighbors = trial_config["preprocessor__feature_space_change__n_neighbors"]
        preprocessor__feature_space_change = SelectPercentile(
            lambda X, y: mutual_info_classif(
                X, y, n_neighbors=n_neighbors, discrete_features=False
            ),
            percentile=trial_config["preprocessor__feature_space_change__percentile"],
        )

        standard_pipeline = standard_pipeline.set_params(
            preprocessor__feature_space_change=preprocessor__feature_space_change
        )

    if not trial_config.get("model__bootstrap", False):
        trial_config["model__oob_score"] = False

    if trial_config.get("model__oob_score", False):
        trial_config["model__oob_score"] = get_scorer(
            trial_config["model__oob_score"]
        )._score_func

    standard_pipeline = standard_pipeline.set_params(
        **{k: v for k, v in trial_config.items() if "model" in k}
    )

    return standard_pipeline


def hyp_param_eval_with_cv(
    what,
    data,
    labels,
    cv,
    standard_pipeline,
    scoring,
    best_fit_scorer,
    outer_cv_step,
    search_space_sampler,
    trial_config,
):
    """Evaluate the hyperparameters with cross-validation for a given dataset and pipeline with the given search space sampler."""
    pipeline = get_pipeline(what, standard_pipeline, search_space_sampler, trial_config)

    logger.info("pipeline:")
    print(pipeline)

    cross_val_results = cross_validate(
        pipeline,
        data,
        labels,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=None,
    )

    cross_val_res = {
        **{
            k.replace("train_", "mean_inner_cv_train/"): np.mean(v)
            for k, v in cross_val_results.items()
            if "train" in k
        },
        **{
            k.replace("test_", "mean_inner_cv_val/"): np.mean(v)
            for k, v in cross_val_results.items()
            if "test" in k
        },
    }

    wandb.log(cross_val_res, step=outer_cv_step)

    return cross_val_results["test_" + best_fit_scorer].mean()


def inner_cv_eval_for_baseline_metalearning(
    train_data,
    train_labels,
    val_data,
    val_labels,
    support_set_indices,
    pipeline,
    scoring,
) -> dict:
    train_data, train_labels, val_data, val_labels = (
        extend_train_with_support_set_from_eval(
            train_data, train_labels, val_data, val_labels, support_set_indices
        )
    )

    pipeline.fit(train_data, train_labels)
    scores = get_scores(
        pipeline,
        train_data,
        train_labels,
        scoring,
        score_name_prefix="mean_inner_cv_train",
    )
    scores.update(
        get_scores(
            pipeline,
            val_data,
            val_labels,
            scoring,
            score_name_prefix="mean_inner_cv_val",
        )
    )

    return scores


def hyp_param_eval_for_baseline_metalearning(
    what: str,
    train_data: pd.DataFrame,
    train_labels: pd.Series,
    val_data: pd.DataFrame,
    val_labels: pd.Series,
    n_inner_cv_splits: int,
    eval_k_shot: int,
    standard_pipeline,
    scoring,
    best_fit_scorer,
    outer_cv_step,
    search_space_sampler,
    trial_config,
):
    """Evaluate the hyperparameters with cross-validation for a given dataset and pipeline with the given search space sampler.

    This function is used for the baseline meta-learning approach where the evaluation data is used as the support set for the validation data.
    """
    pipeline = get_pipeline(what, standard_pipeline, search_space_sampler, trial_config)

    logger.info("pipeline:")
    print(pipeline)

    # Select val indices to be used for training as "support set" for the val data
    inner_cv_val_k_shot_indices = []
    rng = np.random.default_rng(outer_cv_step)
    for _ in range(n_inner_cv_splits):
        inner_cv_val_k_shot_indices.append(
            rng.choice(val_data.index, eval_k_shot, replace=False).tolist()
        )

    # Evaluate the pipeline with the inner cross-validation
    results = []
    for support_set_indices in inner_cv_val_k_shot_indices:
        results.append(
            inner_cv_eval_for_baseline_metalearning(
                train_data,
                train_labels,
                val_data,
                val_labels,
                support_set_indices,
                clone(pipeline),
                scoring,
            )
        )

    # Concatenate results per metric
    cross_val_res = {}
    for res in results:
        for metric, val in res.items():
            if metric not in cross_val_res:
                cross_val_res[metric] = []
            cross_val_res[metric].append(val)
    # Get mean value for all metrics and in wandb-useful format (train and val separated)
    cross_val_res = {k: np.mean(v) for k, v in cross_val_res.items()}

    wandb.log(cross_val_res, step=outer_cv_step)

    return cross_val_res["mean_inner_cv_val/" + best_fit_scorer].mean() # mean_inner_cv_val hard-coded!!!


def column_rename_for_sun_et_al_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata[["Group", "Project_1"]]
    metadata = metadata.rename(columns={"Group": "label", "Project_1": "project"})
    return metadata


def pca_reduction(
    train_data,
    test_data,
    val_data,
    n_components_reduction_factor: int,
    use_cache: bool = False,
):
    if not use_cache:
        pca = PCA(
            n_components=int(train_data.shape[1] // n_components_reduction_factor)
        )
        print("fitting and transforming")
        train_data = pd.DataFrame(pca.fit_transform(train_data), index=train_data.index)
        print("transforming")
        test_data = pd.DataFrame(pca.transform(test_data), index=test_data.index)
        print("transforming")
        val_data = pd.DataFrame(pca.transform(val_data), index=val_data.index)
        train_data.to_csv("train_data_PCA.csv")
        test_data.to_csv("test_data_PCA.csv")
        val_data.to_csv("val_data_PCA.csv")
    else:
        train_data = pd.read_csv("train_data_PCA.csv", index_col=0)
        test_data = pd.read_csv("test_data_PCA.csv", index_col=0)
        val_data = pd.read_csv("val_data_PCA.csv", index_col=0)

    return train_data, test_data, val_data


def encode_labels(
    encoder, labels: pd.Series, positive_class_label: str = None
) -> pd.Series:
    """Encode the labels using the given encoder making sure the positive class is labeled as 1."""
    encoder.fit(labels)
    classes = list(encoder.classes_)
    if positive_class_label in classes:
        positive_class_index = classes.index(positive_class_label)
        if positive_class_index != 1:
            # Swap labels to ensure the desired class is labeled as 1
            classes[1], classes[positive_class_index] = (
                classes[positive_class_index],
                classes[1],
            )
            encoder.classes_ = np.array(classes)
    return pd.Series(encoder.transform(labels), index=labels.index)


def extend_train_with_support_set_from_eval(
    train_data: pd.DataFrame,
    train_labels: pd.Series,
    eval_data: pd.DataFrame,
    eval_labels: pd.Series,
    eval_support_indices: list[object],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Extend the training data with the support data from the evaluation data."""
    train_data = pd.concat([train_data, eval_data.loc[eval_support_indices]])
    train_labels = pd.concat([train_labels, eval_labels.loc[eval_support_indices]])

    eval_data = eval_data.drop(eval_support_indices)
    eval_labels = eval_labels.drop(eval_support_indices)

    return train_data, train_labels, eval_data, eval_labels
