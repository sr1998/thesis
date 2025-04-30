import hashlib
import os
from pathlib import Path
import random
import time
from typing import Iterable

import torch
import filelock
import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
import yaml
from imblearn.pipeline import Pipeline as ImbPipeline
from loguru import logger
from requests import Session as requests_session
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_validate

import wandb
from joblib import Memory
from src.global_vars import (
    BASE_RESUME_DIR,
    BASE_RUN_DIR,
    HTTP_ADAPTER_FOR_REQUESTS,
    RANDOM_SEED,
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
    """Hashes the elements of an iterator in a deterministic way, regardless of the order of the elements."""
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


def create_pipeline(steps: list[object], config: dict[str, object]) -> ImbPipeline:
    """Create a pipeline from the steps and config."""
    # Create a cache object if caching is enabled with default cache location
    if config.get("cache_pipeline_steps", True):
        cacher = Memory(
            location=get_run_dir_for_experiment(config) / "pipeline_cache", verbose=1
        )
    else:
        cacher = None

    return ImbPipeline(
        steps, memory=cacher, verbose=config.get("verbose_pipeline", True)
    )


def get_run_dir_for_experiment(job_name: str, algorithm: str, study: str, wandb_name):
    run_dir = BASE_RUN_DIR / job_name / algorithm / str(study) / wandb_name
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
    return run_dir


def get_resume_dir_for_experiment(job_name, algorithm, study, wandb_name_wo_jobid):
    resum_dir = (
        BASE_RESUME_DIR / job_name / algorithm / str(study) / wandb_name_wo_jobid
    )
    resum_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
    return resum_dir


# def get_data_dir_for_experiment(config: dict[str, object]):
#     data_dir = BASE_DATA_DIR / config["wandb_params"]["name"]
#     data_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
#     return data_dir


def get_scores(
    model: ImbPipeline | BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    scoring: dict,
    score_name_prefix: str = "",
) -> dict:
    """Calculate the scores for the predictions."""
    scores = {}
    y_n_unique = len(np.unique(y))

    y_pred_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    y_dec = model.decision_function(X) if hasattr(model, "decision_function") else None
    y_pred = model.predict(X)

    for score_name, scorer in scoring.items():
        try:
            scoring_function = get_scorer(scorer)._score_func
            if "roc_auc" in score_name or "average_precision" in score_name:
                kwargs = scorer._kwargs if hasattr(scorer, "_kwargs") else {}
                # Get kwargs available, e.g. {average="micro"} if scorer is not a string but a make_scorer object
                if y_pred_proba is not None:
                    y_model = y_pred_proba
                    scoring_function._response_method = "predict_proba"
                    if y_n_unique == 2:
                        y_model = y_pred_proba[
                            :, 1
                        ]  # Explicitly use the positive class
                elif y_dec is not None:
                    y_model = y_dec
                    scoring_function._response_method = "decision_function"
                else:
                    y_model = y_pred
                    scoring_function._response_method = "predict"

                scores[score_name_prefix + score_name] = scoring_function(
                    y, y_model, **kwargs
                )
            else:
                scores[score_name_prefix + score_name] = scoring_function(y, y_pred)
        except Exception as e:
            logger.error(f"Error calculating {score_name} for {score_name_prefix}: {e}")
            scores[score_name_prefix + score_name] = None

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


def get_pipeline(what, standard_pipeline, search_space_sampler, optuna_trial):
    """Get the pipeline with the hyperparameters sampled from the search space."""
    if search_space_sampler is not None:
        trial_config = search_space_sampler(optuna_trial)
    else:
        trial_config = {}

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

    if not trial_config.get("do_normalization_before_scaling"):
        standard_pipeline.named_steps["normalizer"] = "passthrough"

    if "RandomForest" in standard_pipeline.named_steps["model"].__class__.__name__:
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
    trial: optuna.Trial,
):
    """Evaluate the hyperparameters with cross-validation for a given dataset and pipeline with the given search space sampler."""
    pipeline = get_pipeline(what, standard_pipeline, search_space_sampler, trial)

    cross_val_results = cross_validate(
        pipeline,
        data,
        labels,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=None,
    )

    # Get mean and std for all metrics
    wandb_data = {}

    # Add mean train metrics
    mean_train_data = {
        k.replace("train_", f"mean_inner_trains_{outer_cv_step}/"): np.mean(v)
        for k, v in cross_val_results.items()
        if "train" in k
    }
    wandb_data.update(mean_train_data)

    # Add mean test metrics
    mean_test_data = {
        k.replace("test_", f"mean_inner_vals_{outer_cv_step}/"): np.mean(v)
        for k, v in cross_val_results.items()
        if "test" in k
    }
    wandb_data.update(mean_test_data)

    # Add std train metrics
    std_train_data = {
        k.replace("train_", f"std_inner_trains_{outer_cv_step}/"): np.std(v)
        for k, v in cross_val_results.items()
        if "train" in k
    }
    wandb_data.update(std_train_data)

    # Add std test metrics
    std_test_data = {
        k.replace("test_", f"std_inner_vals_{outer_cv_step}/"): np.std(v)
        for k, v in cross_val_results.items()
        if "test" in k
    }
    wandb_data.update(std_test_data)
    wandb_data["trial"] = trial.number

    wandb.log(wandb_data)

    return cross_val_results["test_" + best_fit_scorer].mean()


def cv_eval_for_baseline_metalearning(
    train_data,
    train_labels,
    eval_data,
    eval_labels,
    pipeline,
    scoring,
    score_name_prefix_train="",
    score_name_prefix_eval="",
) -> dict:
    pipeline.fit(train_data, train_labels)
    train_scores = get_scores(
        pipeline,
        train_data,
        train_labels,
        scoring,
        score_name_prefix=score_name_prefix_train,
    )
    val_scores = get_scores(
        pipeline,
        eval_data,
        eval_labels,
        scoring,
        score_name_prefix=score_name_prefix_eval,
    )

    return train_scores, val_scores


def hyp_param_eval_for_baseline_metalearning(
    datasource: str,
    inner_loop_splits: dict[int, list[str | list[str]]],
    orig_train_data: pd.DataFrame,
    orig_train_metadata: pd.DataFrame,
    balanced_or_unbalanced,
    standard_pipeline,
    scoring,
    best_fit_scorer,
    search_space_sampler,
    trial,
):
    """Evaluate the hyperparameters with cross-validation for a given dataset and pipeline with the given search space sampler.

    This function is used for the baseline meta-learning approach where the evaluation data is used as the support set for the validation data.
    """
    pipeline = get_pipeline(datasource, standard_pipeline, search_space_sampler, trial)

    # Evaluate the pipeline with the inner cross-validation
    train_scores = []
    val_scores = []
    for _, (val_study_name, val_support_sets) in inner_loop_splits.items():
        val_metadata = orig_train_metadata[
            orig_train_metadata["Project_1"] == val_study_name
        ]
        val_data = orig_train_data.loc[val_metadata.index]
        train_data = orig_train_data.drop(val_metadata.index)
        train_metadata = orig_train_metadata.drop(val_metadata.index)
        # Make sure the metadata is in the same order as the data
        train_metadata = train_metadata.loc[train_data.index]

        (
            train_data,
            train_metadata,
            val_data,
            val_metadata,
        ) = extend_train_with_support_set_from_eval(
            train_data,
            train_metadata,
            val_data,
            val_metadata,
            val_support_sets,
        )

        if balanced_or_unbalanced == "balanced":
            # We give metadata but get labels back
            train_data, train_labels = make_data_balanced_per_study(
                train_data, train_metadata
            )
        else:
            train_labels = train_metadata["Group"]

        val_labels = val_metadata["Group"]

        train_res, val_res = cv_eval_for_baseline_metalearning(
            train_data, train_labels, val_data, val_labels, clone(pipeline), scoring
        )

        train_scores.append(train_res)
        val_scores.append(val_res)

    # log mean and std of the results
    train_scores = pd.DataFrame(train_scores)
    val_scores = pd.DataFrame(val_scores)

    train_mean = train_scores.mean()
    val_mean = val_scores.mean()

    train_std = train_scores.std()
    val_std = val_scores.std()

    # Create a dictionary for wandb logging
    wandb_data = {}

    # Add mean train metrics
    for metric, value in train_mean.items():
        wandb_data[f"mean_hyp_param_opt_trains/{metric}"] = value

    # Add mean validation metrics
    for metric, value in val_mean.items():
        wandb_data[f"mean_hyp_param_opt_vals/{metric}"] = value

    # Add std train metrics
    for metric, value in train_std.items():
        wandb_data[f"std_hyp_param_opt_trains/{metric}"] = value

    # Add std validation metrics
    for metric, value in val_std.items():
        wandb_data[f"std_hyp_param_opt_vals/{metric}"] = value

    wandb_data["trial"] = trial.number

    wandb.log(wandb_data)

    return wandb_data[f"mean_hyp_param_opt_vals/{best_fit_scorer}"]


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
    train_metadata: pd.Series,
    eval_data: pd.DataFrame,
    eval_metadata: pd.Series,
    eval_support_indices: list[object],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Extend the training data with the support data from the evaluation data."""
    train_data = pd.concat([train_data, eval_data.loc[eval_support_indices]])
    train_metadata = pd.concat(
        [train_metadata, eval_metadata.loc[eval_support_indices]]
    )
    train_metadata = train_metadata.loc[train_data.index]

    eval_data = eval_data.drop(eval_support_indices)
    eval_metadata = eval_metadata.drop(eval_support_indices)
    eval_metadata = eval_metadata.loc[eval_data.index]

    return train_data, train_metadata, eval_data, eval_metadata


def make_data_balanced_per_study(train_data, train_metadata, method="SMOTE"):
    """Make the data balanced per study by applying SMOTE or any other method.

    Args:
        train_data: The training data.
        train_metadata: The metadata for the training data.
        method: The method to use for balancing the data. Default is "SMOTE".

    Returns:
        A tuple containing the balanced training data and the corresponding labels.
        Although metadata is passed, labels are returned.
    """
    grouped_per_study = train_metadata.groupby("Project_1")
    train_labels = train_metadata["Group"]
    balanced_data = []
    balanced_labels = []

    for study_name, idx in grouped_per_study.groups.items():
        study_data = train_data.loc[idx]
        study_labels = train_labels.loc[idx]

        # Apply SMOTE or any other method to balance the data
        if method == "SMOTE":
            from imblearn.over_sampling import SMOTE

            smote = SMOTE()
            study_data, study_labels = smote.fit_resample(study_data, study_labels)

        balanced_data.append(study_data)
        balanced_labels.append(study_labels)

    # Concatenate the balanced data and labels and shuffle
    balanced_data = pd.concat(balanced_data)
    balanced_labels = pd.concat(balanced_labels)
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
    balanced_labels = balanced_labels.sample(frac=1, random_state=42).reset_index(
        drop=True
    )

    return balanced_data, balanced_labels


def optuna_wandb_callback(study, trial, outer_cv_step: int | None = None):
    # Log parameters
    prefix = f"trial_for_outer_{outer_cv_step}" if outer_cv_step else "trial"
    trial_params = {f"{prefix}/{k}": v for k, v in trial.params.items()}

    # Log metrics
    metrics = {
        f"{prefix}/number": trial.number,
        f"{prefix}/value": trial.value,
        f"{prefix}/best_value": study.best_value,
        f"{prefix}/duration_seconds": trial.duration.total_seconds()
        if hasattr(trial, "duration")
        else None,
    }

    # Log everything using W&B's global step counter
    wandb.log({**trial_params, **metrics})


def check_run_finished(project_name, run_name, entity_name="shayan000"):
    api = wandb.Api()
    runs = api.runs(f"{entity_name}/{project_name}")

    for run in runs:
        if run_name[:-10] in run.name and run.state == "finished":
            print(f"Run '{run_name}' exists and is finished. Stopping script.")
            exit(0)

    print(f"No finished run named '{run_name}' found. Continuing...")


def load_checkpoint(checkpoint_path, resume=True):
    """Load checkpoint data or create empty checkpoint if none exists."""
    if os.path.exists(checkpoint_path) and resume:
        with open(checkpoint_path, "r") as f:
            return yaml.safe_load(f)
    return {
            "completed_folds": [],
            "trials_done_per_job": {},
            "wandb_run_id": None,
            "fold_metrics": {},
            "warmup_completed": False,  # Flag for initial warmup phase
            "optimization_done": False,  # Flag for optimization completion
            "best_trial_params": None,
            "primary_job_id": None,
        }


def save_checkpoint(checkpoint_path, checkpoint_data):
    """Save checkpoint data to disk."""
    with open(checkpoint_path, "w") as f:
        yaml.safe_dump(checkpoint_data, f, default_flow_style=False)

def checkpoint_updater_callback_optuna(
    study: optuna.Study,
    trial: optuna.Trial,
    checkpoint_path: str,
    job_id: str,
):
    """Callback function to update the checkpoint with the latest trial data."""
    try:
        with filelock.FileLock(checkpoint_path + ".lock", timeout=30):
            # Load existing checkpoint data
            checkpoint_data = load_checkpoint(checkpoint_path)

            # Update checkpoint trials done for job
            checkpoint_data["trials_done_per_job"][job_id] = checkpoint_data.get("trials_done_per_job", {}).get(job_id, 0) + 1

            # Save updated checkpoint data
            save_checkpoint(checkpoint_path, checkpoint_data)
    except filelock.Timeout:
        logger.error(f"Failed to acquire lock for checkpoint file: {checkpoint_path}")
    except Exception as e:
        logger.error(f"Error updating checkpoint: {e}")

def set_seed(seed=RANDOM_SEED):
    # Python's built-in random module
    random.seed(seed)
    
    # Numpy's random module
    np.random.seed(seed)
    
    # PyTorch seed for CPU
    torch.manual_seed(seed)
    
    # PyTorch seed for all GPU devices (if using CUDA)
    torch.cuda.manual_seed_all(seed)
    
    # Make sure to disable CuDNN's non-deterministic optimizations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False