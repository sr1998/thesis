import os
import sys
from importlib import import_module
from pathlib import Path

from sklearn.inspection import permutation_importance

from src.data.dataloader import (
    split_sun_et_al_data,
)
from src.global_vars import BASE_DATA_DIR

sys.path.append(".")
import fire
import numpy as np
import optuna
import pandas as pd
from loguru import logger

import wandb
from src.helper_function import (
    df_str_for_loguru,
    encode_labels,
    extend_train_with_support_set_from_eval,
    get_pipeline,
    get_run_dir_for_experiment,
    get_scores,
    hyp_param_eval_for_baseline_metalearning,
)


def main(
    what: str,
    config_script: str,
    *,
    test_study: list,
    val_study: list,
    abundance_file: str | Path,  # for sun et al. data for now
    metadata_file: str | Path,  # for sun et al. data for now
    eval_k_shot: int,
    positive_class_label: str | None = None,
    metadata_cols_to_use_as_features: list[str] = [],
    load_from_cache_if_available: bool = True,
):
    """Run the baseline pipeline for the baseline meta-learning inspired approach."""
    config_module = import_module(config_script)
    setup = config_module.get_setup()
    (
        misc_config,
        n_outer_cv_splits,
        n_inner_cv_splits,
        standard_pipeline,
        label_preprocessor,
        scoring,
        best_fit_scorer,
        tuning_mode,
        search_space_sampler,
        tuning_num_samples,
    ) = setup.values()

    setup["what"] = what
    setup["test_study"] = test_study
    setup["val_study"] = val_study
    setup["abundance_file"] = abundance_file
    setup["metadata_file"] = metadata_file
    tax_level = abundance_file.split("_")[1]
    setup["tax_level"] = tax_level
    setup["model"] = standard_pipeline.named_steps["model"].__class__.__name__
    setup["positive_class_label"] = positive_class_label
    setup["metdata_cols_to_use_as_features"] = metadata_cols_to_use_as_features

    job_id = os.getenv("SLURM_JOB_ID")
    wandb_name = (
        f"w_{what}__TS{test_study}_VS{val_study}_J{job_id}_T{tax_level}_EK{eval_k_shot}"
    )

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
        "t_s" + str(test_study),
        "v_s" + str(val_study),
        "m_" + standard_pipeline.named_steps["model"].__class__.__name__,
        "j_" + job_id if job_id else "j_local",
        "tax_" + tax_level,
        "e_k" + str(eval_k_shot),
    ]

    if what == "sun et al":
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
    if what == "sun et al":
        data_root_dir = BASE_DATA_DIR / "sun_et_al_data"
        data = pd.read_csv(
            f"{data_root_dir}/{abundance_file}",
            index_col=0,
            header=0,
        )

        metadata = pd.read_csv(
            f"{data_root_dir}/{metadata_file}",
            index_col=0,
            header=0,
        )

        # Sort samples
        data = data.sort_index()
        metadata = metadata.sort_index()

        train_data, test_data, val_data, train_metadata, test_metadata, val_metadata = (
            split_sun_et_al_data(data, metadata, test_study, val_study)
        )

        train_labels = train_metadata["Group"]
        test_labels = test_metadata["Group"]
        val_labels = val_metadata["Group"]
        assert isinstance(train_labels, pd.Series)
    else:
        raise ValueError("Invalid value for 'what'")

    logger.success("Data obtained")

    # log data statistics to wandb
    wandb.log(
        {
            "Train data description": wandb.Table(
                dataframe=train_data.describe().T.reset_index()
            )
        },
        step=0,
    )
    wandb.log(
        {
            "Val data description": wandb.Table(
                dataframe=val_data.describe().T.reset_index()
            )
        },
        step=0,
    )
    wandb.log(
        {
            "Test data description": wandb.Table(
                dataframe=test_data.describe().T.reset_index()
            )
        },
        step=0,
    )

    wandb.log(
        {
            "Train data labels": wandb.Table(
                dataframe=train_labels.value_counts(dropna=False).reset_index()
            )
        },
        step=0,
    )
    wandb.log(
        {
            "Val data labels": wandb.Table(
                dataframe=val_labels.value_counts(dropna=False).reset_index()
            )
        },
        step=0,
    )
    wandb.log(
        {
            "Test data labels": wandb.Table(
                dataframe=test_labels.value_counts(dropna=False).reset_index()
            )
        },
        step=0,
    )

    train_labels = encode_labels(label_preprocessor, train_labels, positive_class_label)
    val_labels = encode_labels(label_preprocessor, val_labels, positive_class_label)
    test_labels = encode_labels(label_preprocessor, test_labels, positive_class_label)

    train_scores = []
    test_scores = []

    logger.info("Starting with outer cv")
    split_results = []
    # split_permutation_importance = pd.DataFrame()
    split_rf_importance_df = pd.DataFrame()

    # Select test indices to be used for training as "support set" for the test data
    outer_cv_test_k_shot_indices = []
    rng = np.random.default_rng(42)
    for _ in range(n_outer_cv_splits):
        k_shot_indices = rng.choice(test_data.index, eval_k_shot, replace=False)
        outer_cv_test_k_shot_indices.append(k_shot_indices.tolist())

    for i, support_set_indices in enumerate(outer_cv_test_k_shot_indices):
        # With random state defined like this, each experiment is reproducible but the inner cv splits are different per outer cv split
        optuna_study = optuna.create_study(
            direction=tuning_mode, study_name=f"outer_cv_{i}_for_{wandb.run.name}"
        )
        optuna_study.optimize(
            lambda trial: hyp_param_eval_for_baseline_metalearning(
                what,
                train_data,
                train_labels,
                val_data,
                val_labels,
                n_inner_cv_splits,
                eval_k_shot,
                standard_pipeline,
                scoring,
                best_fit_scorer,
                i,
                search_space_sampler,
                trial,
            ),
            n_trials=tuning_num_samples,
        )

        # outer cv data split (done here, as we need to extend the train data with the support set)
        (
            train_data_extended,
            train_labels_extended,
            test_query_data,
            test_query_labels,
        ) = extend_train_with_support_set_from_eval(
            train_data,
            train_labels,
            test_data,
            test_labels,
            support_set_indices,
        )

        best_trial = optuna_study.best_trial
        # save best trial parameters + split for this loop
        best_trial_params = best_trial.params
        best_trial_params = {k: str(v) for k, v in best_trial_params.items()}
        # Convert to a dictionary format for easier table storage
        split_entry = {
            "outer_cv_split": i,
            "train_size": len(train_data_extended),
            "test_size": len(test_query_data),
            **best_trial_params,  # Add all hyperparameters
            "test_indices": ";".join(map(str, support_set_indices)),
        }

        split_results.append(split_entry)

        best_model = get_pipeline(
            what, standard_pipeline, search_space_sampler, best_trial
        )
        best_model.fit(train_data_extended, train_labels_extended)

        train_outer_cv_score = get_scores(
            best_model,
            train_data_extended,
            train_labels_extended,
            scoring,
            score_name_prefix="train",
        )
        test_outer_cv_score = get_scores(
            best_model, test_query_data, test_query_labels, scoring, score_name_prefix="test"
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
                    "Feature": train_data_extended.columns,
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
    results_df = pd.DataFrame(split_results)
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
    # fire.Fire(main)

    main(
        "sun et al",
        "run_configs.rf_metalearning_baseline_for_sun_et_al",
        test_study="HanL_2021",
        val_study="JieZ_2017",
        abundance_file="mpa4_species_profile_preprocessed.csv",
        metadata_file="sample_group_species_preprocessed.csv",
        eval_k_shot=10,
        positive_class_label="Sick",
    )
