import os
import sys
from importlib import import_module
from pathlib import Path

from sklearn.inspection import permutation_importance

from src.data.dataloader import get_mgnify_data, get_sun_et_al_study_data

sys.path.append(".")
import fire
import numpy as np
import optuna
import pandas as pd
from loguru import logger
from numpy.random import RandomState

import wandb
from src.helper_function import (
    encode_labels,
    get_pipeline,
    get_run_dir_for_experiment,
    get_scores,
    hyp_param_eval_with_cv,
)


def main(
    what: str,
    config_script: str,
    *,
    study: str | list[str],
    abundance_file: str | Path,  # for sun et al. data for now
    metadata_file: str | Path,  # for sun et al. data for now
    summary_type: str | None = None,
    pipeline_version: str | None = None,
    label_col: str | None = None,
    positive_class_label: str | None = None,
    metadata_cols_to_use_as_features: list[str] = [],
    load_from_cache_if_available: bool = True,
):
    """Run the pipeline for the given study accessions and config script.

    Args:
        what: This decided what dataset is ran.
        config_script: The path to the config script
        study:
            For mgnify: Leave out if all studies desired; if provided, only the summaries of
                those studies are used if studies are given, they have to contain the desired
                summary given by study download_label_start. For now only one study is supported.
            For others: Give the study name that will give the right data. E.g. for sun et al.
                give the Project_1 term desired so the right samples are selected.
        summary_type: Required with mgnify. Indicates what summary file to use for the studies.
            Possible values:
                - GO_abundances
                - GO-slim_abundances
                - phylum_taxonomy_abundances_SSU
                - taxonomy_abundances_SSU
                - IPR_abundances
                - ... (see MGnify API for more)
        pipeline_version: Required with mgnify. Indicates what pipeline version to use.
            Possible values:
                - v3.0
                - v4.0
                - v4.1
                - v5.0
        label_col: Required with mgnify. Label column for classification.
        positive_class_label: Required with mgnify. Positive class label to be used in label encoding.
        metadata_cols_to_use_as_features: Metadata columns to use; leave empty for none, or give value "all" for all columns.
        load_from_cache_if_available: This will reuse cross-validation splits if this same experiment has been done before.
            Note: This is not implemented yet.
        wandb_run_name: The name of the wandb run. Defaults to None.

    Returns:
        None

    """
    config_module = import_module(config_script)
    setup = config_module.get_setup()
    (
        misc_config,
        outer_cv_config,
        inner_cv_config,
        standard_pipeline,
        label_preprocessor,
        scoring,
        best_fit_scorer,
        tuning_mode,
        search_space_sampler,
        tuning_num_samples,
    ) = setup.values()

    setup["what"] = what
    setup["study"] = study
    setup["abundance_file"] = abundance_file
    setup["metadata_file"] = metadata_file
    tax_level = abundance_file.split("_")[1]
    setup["tax_level"] = tax_level
    setup["model"] = standard_pipeline.named_steps["model"].__class__.__name__
    if what == "mgnify":
        setup["summary_type"] = summary_type
        setup["pipeline_version"] = pipeline_version
        setup["label_col"] = label_col
    setup["positive_class_label"] = positive_class_label
    setup["metdata_cols_to_use_as_features"] = metadata_cols_to_use_as_features

    job_id = os.getenv("SLURM_JOB_ID")
    wandb_name = f"w_{what}__d_{study}__j_{job_id}__t_{tax_level}"
    wandb_name += f"s_{summary_type.split("_")[0]}" if summary_type else ""

    # get misc config parameters
    use_wandb = misc_config["wandb"]
    misc_config["wandb_params"]["name"] = (
        wandb_name  # Not nice to change the config like this, better to use name directly
    )
    wandb_params = misc_config["wandb_params"]
    verbose_pipeline = misc_config.get("verbose_pipeline", True)

    # Set up file logging
    logger_path = get_run_dir_for_experiment(misc_config) / "log.log"
    logger.add(logger_path, colorize=True, level="DEBUG")
    logger.info("Setting up everything")

    wandb_base_tags = [
        "d_" + str(study),
        "m_" + standard_pipeline.named_steps["model"].__class__.__name__,
        "j_" + job_id if job_id else "j_local",
        "t_" + tax_level,
    ]

    if what == "mgnify":
        assert (
            summary_type is not None
            and pipeline_version is not None
            and label_col is not None
        ), "For mgnify, summary_type, pipeline_version and label_col must be provided as arguments."

        wandb_base_tags.append("w_mgnify")
        wandb_base_tags.append("s_" + summary_type)
        wandb_base_tags.append("p_" + pipeline_version)
        wandb_base_tags.append("l_" + label_col)
    elif what == "sun et al":
        wandb_base_tags.append("w_sun_et_al")
    else:
        raise ValueError("Invalid value for 'what'")

    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            notes=str(setup),
            config=setup,
            **wandb_params,
            tags=wandb_base_tags,
        )
    else:
        wandb.init(
            mode="disabled",
            notes=str(setup),
            config=setup,
            **wandb_params,
            tags=wandb_base_tags,
        )

    logger.success("wandb init done")

    # Load data
    if what == "mgnify":
        data, labels = get_mgnify_data(
            study,
            summary_type,
            pipeline_version,
            label_col,
            metadata_cols_to_use_as_features,
        )
    elif what == "sun et al":
        data, labels = get_sun_et_al_study_data(study, abundance_file, metadata_file)
    else:
        raise ValueError("Invalid value for 'what'")

    logger.success("Data obtained")

    # log data statistics to wandb
    wandb.log(
        {"Data description": wandb.Table(dataframe=data.describe().T.reset_index())},
        step=0,
    )
    wandb.log(
        {
            "labels": wandb.Table(
                dataframe=labels.value_counts(dropna=False).reset_index()
            )
        },
        step=0,
    )

    encoded_labels = encode_labels(label_preprocessor, labels, positive_class_label=positive_class_label).values

    train_scores = []
    test_scores = []
    split_config = []

    outer_cv = outer_cv_config["type"](**outer_cv_config["params"])

    logger.info("Starting with outer cv")
    # split_permutation_importance = pd.DataFrame()
    split_rf_importance_df = pd.DataFrame()

    for i, (train_index, test_index) in enumerate(outer_cv.split(data, encoded_labels)):
        # outer cv data split
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = encoded_labels[train_index], encoded_labels[test_index]

        # With random state defined like this, each experiment is reproducible but the inner cv splits are different per outer cv split
        random_state = RandomState(i)

        inner_cv = inner_cv_config["type"](
            **inner_cv_config["params"], random_state=random_state
        )

        optuna_study = optuna.create_study(
            direction=tuning_mode, study_name=f"outer_cv_{i}_for_{wandb.run.name}"
        )
        optuna_study.optimize(
            lambda trial: hyp_param_eval_with_cv(
                what,
                X_train,
                y_train,
                inner_cv,
                standard_pipeline,
                scoring,
                best_fit_scorer,
                i,
                search_space_sampler,
                trial,
            ),
            n_trials=tuning_num_samples,
        )

        best_trial = optuna_study.best_trial
        # save best trial parameters + split for this loop
        best_trial_params = best_trial.params
        best_trial_params = {k: str(v) for k, v in best_trial_params.items()}
        # Convert to a dictionary format for easier table storage
        split_entry = {
            "outer_cv_split": i,
            "train_size": len(train_index),
            "test_size": len(test_index),
            **best_trial_params,  # Add all hyperparameters
            "train_indices": ";".join(
                map(str, train_index)
            ),  # Store indices as a semicolon-separated string
            "test_indices": ";".join(map(str, test_index)),
        }

        split_config.append(split_entry)

        best_model = get_pipeline(
            what, standard_pipeline, search_space_sampler, best_trial
        )
        best_model.fit(X_train, y_train)

        train_outer_cv_score = get_scores(
            best_model, X_train, y_train, scoring, score_name_prefix="train"
        )
        test_outer_cv_score = get_scores(
            best_model, X_test, y_test, scoring, score_name_prefix="test"
        )

        # Permutation importance (all zero. I guess due to correlation of features)
        # perm_importance = permutation_importance(
        #     best_model,
        #     X_test,
        #     y_test,
        #     scoring=best_fit_scorer,
        #     n_repeats=5,
        #     random_state=i,
        # )
        # perm_importance_df = pd.DataFrame(
        #     {
        #         "Feature": X_train.columns,
        #         "Importance": perm_importance.importances_mean,  # Mean importance over repeats
        #         "Std": perm_importance.importances_std,  # Standard deviation
        #         "Outer CV Split": i,
        #     }
        # )
        # split_permutation_importance = pd.concat(
        #     [split_permutation_importance, perm_importance_df], axis=0
        # )

        # Random Forest feature importance
        if hasattr(best_model.named_steps["model"], "feature_importances_"):
            rf_importance = best_model.named_steps["model"].feature_importances_

            rf_importance_df = pd.DataFrame(
                {
                    "Feature": X_train.columns,
                    "RF Importance": rf_importance,
                    "Outer CV Split": i,
                }
            )

            split_rf_importance_df = pd.concat(
                [split_rf_importance_df, rf_importance_df], axis=0
            )

        wandb.log(
            {"Outer fold": dict(train_outer_cv_score, **test_outer_cv_score)},
            step=i,
        )

        # tuner_results.append(tuner_cv_result)
        train_scores.append(train_outer_cv_score)
        test_scores.append(test_outer_cv_score)

    # log mean and std of the results
    train_scores = pd.DataFrame(train_scores)
    test_scores = pd.DataFrame(test_scores)

    train_mean = train_scores.mean()
    test_mean = test_scores.mean()

    train_std = train_scores.std()
    test_std = test_scores.std()

    # Log bar plots for train and test metrics
    train_summary_df = pd.DataFrame(
        {"Metric": train_mean.index, "Mean": train_mean.values, "Std": train_std.values}
    )

    test_summary_df = pd.DataFrame(
        {"Metric": test_mean.index, "Mean": test_mean.values, "Std": test_std.values}
    )

    wandb.log(
        {"Train Metrics Summary table": wandb.Table(dataframe=train_summary_df)},
        step=i + 1,
    )
    wandb.log(
        {"Test Metrics Summary table": wandb.Table(dataframe=test_summary_df)},
        step=i + 1,
    )

    # Save all outer CV splits and best trial parameters
    results_df = pd.DataFrame(split_config)
    results_path = get_run_dir_for_experiment(misc_config) / "outer_cv_results.csv"
    results_df.to_csv(results_path, index=False)
    wandb.log({"Outer CV Results": wandb.Table(dataframe=results_df)})
    logger.success(
        f"Saved all outer CV splits and best trial parameters to {results_path} and wandb."
    )

    # Save permutation importance
    # perm_importance_path = (
    #     get_run_dir_for_experiment(misc_config) / "permutation_importance.csv"
    # )
    # split_permutation_importance.to_csv(perm_importance_path, index=False)
    # wandb.log(
    #     {"Permutation Feature Imp": wandb.Table(dataframe=split_permutation_importance)}
    # )

    # Save RF feature importance
    feature_importance_path = (
        get_run_dir_for_experiment(misc_config) / "feature_importance.csv"
    )
    split_rf_importance_df.to_csv(feature_importance_path, index=False)
    wandb.log({"RF Feature Imp": wandb.Table(dataframe=split_rf_importance_df)})

    # mean and std of importance of outer runs
    rf_importance_mean = split_rf_importance_df.groupby("Feature").mean()
    rf_importance_std = split_rf_importance_df.groupby("Feature").std()

    rf_importance_summary_df = pd.DataFrame(
        {
            "Feature": rf_importance_mean.index,
            "Mean Importance": rf_importance_mean["RF Importance"],
            "Std Importance": rf_importance_std["RF Importance"],
        }
    )

    wandb.log(
        {
            "RF Feature Importance Summary": wandb.Table(
                dataframe=rf_importance_summary_df
            )
        }
    )
    importance_summary_path = (
        get_run_dir_for_experiment(misc_config) / "feature_importance_summary.csv"
    )
    rf_importance_summary_df.to_csv(importance_summary_path, index=False)

    logger.success("Done!")
    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)

    # main(
    #     "mgnify",
    #     "run_configs.simple_rf_baseline_for_optuna",
    #     tax_level="species",
    #     study=["MGYS00003677"],
    #     summary_type="GO_abundances",
    #     pipeline_version="v4.1",
    #     label_col="disease status__biosamples",
    #     positive_class_label="Sick",
    # )
