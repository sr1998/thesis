import os
import sys
from importlib import import_module
from pathlib import Path

from joblib import dump as joblib_dump
from sklearn.inspection import permutation_importance

from src.data.dataloader import (
    get_cross_validation_sun_et_al_data_splits,
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
    datasource: str,
    config_script: str,
    *,
    test_study: str | list[str],
    abundance_file: str | Path,  # for sun et al. data for now
    metadata_file: str | Path,  # for sun et al. data for now
    train_k_shot: int,
    balanced_or_unbalanced: str = "balanced",
    positive_class_label: str | None = None,
    metadata_cols_to_use_as_features: list[str] = [],
    load_from_cache_if_available: bool = True,
    features_to_use: list[str] = None,
):
    """Run the baseline pipeline for the baseline meta-learning inspired approach."""
    config_module = import_module(config_script)
    setup = config_module.get_setup()
    (
        misc_config,
        n_outer_splits,
        n_inner_splits,
        standard_pipeline,
        label_preprocessor,
        scoring,
        best_fit_scorer,
        tuning_mode,
        search_space_sampler,
        tuning_num_samples,
    ) = setup.values()

    setup["datasource"] = datasource
    setup["test_study"] = test_study
    # setup["val_study"] = val_study
    setup["abundance_file"] = abundance_file
    setup["metadata_file"] = metadata_file
    tax_level = abundance_file.split("_")[1]
    setup["tax_level"] = tax_level
    setup["model"] = standard_pipeline.named_steps["model"].__class__.__name__
    setup["positive_class_label"] = positive_class_label
    setup["metdata_cols_to_use_as_features"] = metadata_cols_to_use_as_features
    setup["balanced_or_unbalanced"] = balanced_or_unbalanced
    setup["train_k_shot"] = train_k_shot

    job_id = os.getenv("SLURM_JOB_ID")
    wandb_name = f"w_{datasource}__TS{test_study}_J{job_id}_T{tax_level}_EK{train_k_shot}"  # _VS{val_study}

    # get misc config parameters
    use_wandb = misc_config["wandb"]
    misc_config["wandb_params"]["name"] = (
        wandb_name  # Not nice to change the config like this, better to use name directly
    )
    wandb_params = misc_config["wandb_params"]
    verbose_pipeline = misc_config.get("verbose_pipeline", True)

    run_dir = get_run_dir_for_experiment(misc_config)

    # Set up file logging
    logger_path = run_dir / "log.log"
    logger.add(logger_path, colorize=True, level="DEBUG")
    logger.info("Setting up everything")

    wandb_base_tags = [
        "t_s" + str(test_study),
        "w_" + datasource,
        # "v_s" + str(val_study),
        "m_" + standard_pipeline.named_steps["model"].__class__.__name__,
        "tax_" + tax_level,
        "t_k" + str(train_k_shot),
        balanced_or_unbalanced,
    ]

    if datasource == "sun et al":
        wandb_base_tags.append("w_sun_et_al")
    else:
        raise ValueError("Invalid value for 'what'")

    logger.success("wandb init done")

    # Load data
    if datasource == "sun et al":
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

        metadata = metadata.loc[data.index]

        if features_to_use:
            data = data.loc[:, features_to_use]

        # Get the data splits: outer and inner cross val splits
        (
            test_loop_data_selection,
            val_loop_data_selection,
            train_data,
            train_metadata,
            test_data,
            test_metadata,
        ) = get_cross_validation_sun_et_al_data_splits(
            data,
            metadata,
            test_study=test_study,
            k_shot=train_k_shot,
            balanced_or_unbalanced=balanced_or_unbalanced,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
        )

        train_metadata["Group"] = encode_labels(
            label_preprocessor, train_metadata["Group"], positive_class_label
        )
        test_metadata["Group"] = encode_labels(
            label_preprocessor, test_metadata["Group"], positive_class_label
        )
    else:
        raise ValueError("Invalid value for 'datasource'")

    logger.success("Data obtained")

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

    train_scores = []
    test_scores = []
    # split_config = []

    # split_permutation_importance = pd.DataFrame()
    split_rf_importance_df = pd.DataFrame()

    optuna_study = optuna.create_study(
            direction=tuning_mode,
            study_name=f"hyper-param_optimization_for_{wandb.run.name}",
        )
    optuna_study.optimize(
        lambda trial: hyp_param_eval_for_baseline_metalearning(
            datasource,
            val_loop_data_selection,
            train_data,
            train_metadata,
            train_k_shot,
            train_k_shot,
            standard_pipeline,
            scoring,
            best_fit_scorer,
            search_space_sampler,
            trial,
            label_preprocessor,
            positive_class_label,
        ),
        n_trials=tuning_num_samples,
    )

    for i, test_support_set in enumerate(test_loop_data_selection):
        # outer cv data split (done here, as we need to extend the train data with the support set)
        (
            train_data_extended,
            train_labels_extended,
            test_query_data,
            test_query_labels,
        ) = extend_train_with_support_set_from_eval(
            train_data,
            train_metadata["Group"],
            test_data,
            test_metadata["Group"],
            test_support_set,
        )

        best_trial = optuna_study.best_trial
        # save best trial parameters + split for this loop
        best_trial_params = best_trial.params
        best_trial_params = {k: str(v) for k, v in best_trial_params.items()}
        # Convert to a dictionary format for easier table storage
        # split_entry = {
        #     "outer_cv_split": i,
        #     **best_trial_params,  # Add all hyperparameters
        # }

        # split_config.append(split_entry)

        best_model = get_pipeline(
            datasource, standard_pipeline, search_space_sampler, best_trial
        )
        best_model.fit(train_data_extended, train_labels_extended)
        # save the model
        model_path = run_dir / f"pipeline_outer_cv_{i}.joblib"
        joblib_dump(best_model, model_path)

        train_outer_cv_score = get_scores(
            best_model,
            train_data_extended,
            train_labels_extended,
            scoring,
            score_name_prefix="train/",
        )
        test_outer_cv_score = get_scores(
            best_model,
            test_query_data,
            test_query_labels,
            scoring,
            score_name_prefix="test/",
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

        wandb.log({"Outer fold": dict(train_outer_cv_score, **test_outer_cv_score)})

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

    wandb.log({"Train Metrics Summary table": wandb.Table(dataframe=train_summary_df)})
    wandb.log({"Test Metrics Summary table": wandb.Table(dataframe=test_summary_df)})

    # Save all outer CV splits and best trial parameters
    # results_df = pd.DataFrame(split_config)
    # results_path = run_dir / "outer_cv_results.csv"
    # results_df.to_csv(results_path, index=False)
    # wandb.log({"Outer CV Results": wandb.Table(dataframe=results_df)})
    # logger.success(
    #     f"Saved all outer CV splits and best trial parameters to {results_path} and wandb."
    # )

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
        run_dir / "feature_importance.csv"
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
        run_dir / "feature_importance_summary.csv"
    )
    rf_importance_summary_df.to_csv(importance_summary_path, index=False)

    logger.success("Done!")
    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)

    # main(
    #     datasource="sun et al",
    #     config_script="run_configs.rf_metalearning_baseline_for_sun_et_al",
    #     test_study="LiJ_2017",
    #     abundance_file="mpa4_species_profile_preprocessed.csv",
    #     metadata_file="sample_group_species_preprocessed.csv",
    #     train_k_shot=10,
    #     balanced_or_unbalanced="balanced",
    #     positive_class_label="Disease",
    # )
