import sys
from importlib import import_module

import fire
import numpy as np
import pandas as pd
import ray
from loguru import logger
from numpy.random import RandomState
from ray import tune
from ray.train import RunConfig
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from ray.tune.stopper import ExperimentPlateauStopper
from ray.util.joblib import register_ray
from sklearn.model_selection import cross_validate

import joblib
import wandb
from src.data.dataloader import load_data
from src.global_vars import BASE_DATA_DIR, BASE_DIR
from src.helper_function import (
    get_run_dir_for_experiment,
    get_scores,
    is_cluster_environment,
)


def get_data(
    base_data_dir: str,
    study_accessions: str | list[str] | None,
    summary_type: str,
    pipeline_version: str,
    label_col: str,
    metdata_cols_to_use_as_features: list[str] = [],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(study_accessions, list) and study_accessions is not None:
        study_accessions = [study_accessions]

    data, labels = load_data(
        base_data_dir,
        study_accessions,
        summary_type,
        pipeline_version,
        metdata_cols_to_use_as_features,
        label_col,
    )

    return data, labels


def hyp_param_train_with_cv(
    config, data, labels, cv, standard_pipeline, scoring, best_fit_scorer, outer_cv_step
):
    standard_pipeline.set_params(**config)

    logger.info("config")
    print(config)
    logger.info("standard_pipeline")
    print(standard_pipeline)

    register_ray()

    with joblib.parallel_backend("ray"):
        cross_val_results = cross_validate(
            standard_pipeline,
            data,
            labels,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=cv.n_splits if "n_splits" in cv.__dict__ else 1,
        )

        cross_val_res_dict = {
            **{
                k.replace("train_", "train/"): v
                for k, v in cross_val_results.items()
                if "train" in k
            },
            **{
                k.replace("test_", "val/"): v
                for k, v in cross_val_results.items()
                if "test" in k
            },
        }

    wandb.log(cross_val_res_dict, step=outer_cv_step)

    tune.report(
        {
            **cross_val_res_dict,
            "best_fit_metric": cross_val_results["test_" + best_fit_scorer].mean(),
            "fit_times": cross_val_results["fit_time"],
            "score_times": cross_val_results["score_time"],
            "done": True,
        }
    )


def main(
    config_script: str,
    *,
    study_accessions: str | list[str],
    summary_type: str,
    pipeline_version: str,
    label_col: str,
    positive_class_label: str,
    metdata_cols_to_use_as_features: list[str] = [],
    load_from_cache_if_available: bool = True,
):
    """Run the pipeline for the given study accessions and config script.

    Args:
        config_script: The path to the config script
        study_accessions: Leave out if all studies desired; if provided, only the summaries of
            those studies are used if studies are given, they have to contain the desired
            summary given by study download_label_start. For now only one study is supported.
        summary_type: Indicates what summary file to use for the studies.
            Possible values:
                - GO_abundances
                - GO-slim_abundances
                - phylum_taxonomy_abundances_SSU
                - taxonomy_abundances_SSU
                - IPR_abundances
                - ... (see MGnify API for more)
        pipeline_version: Indicates what pipeline version to use.
            Possible values:
                - v3.0
                - v4.0
                - v4.1
                - v5.0
        label_col: Label column for classification.
        positive_class_label: Positive class label to be used in label encoding.
        metdata_cols_to_use_as_features: Metadata columns to use; leave empty for none, or give value "all" for all columns.
        load_from_cache_if_available: This will reuse cross-validation splits if this same experiment has been done before.
            Note: This is not implemented yet.
        wandb_run_name: The name of the wandb run. Defaults to None.

    Returns:
        None

    """
    if is_cluster_environment():
        logger.info("Running in SLURM cluster environment. Initializing Ray...")
        ray.init(address="auto")  # Connect to Ray cluster
    else:
        logger.info("Running locally. Initializing Ray with default settings...")
        ray.init()  # Local initialization

    register_ray()  # Register Ray backend for joblib
    assert ray.is_initialized(), "Ray is not initialized"

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
        tuning_grid,
        tuning_num_samples,
    ) = setup.values()

    # get misc config parameters
    use_wandb = misc_config["wandb"]
    wandb_params = misc_config["wandb_params"]
    verbose_pipeline = misc_config.get("verbose_pipeline", True)

    # Set up file logging
    logger_path = get_run_dir_for_experiment(misc_config) / "log.log"
    logger.add(logger_path, colorize=True, level="DEBUG")
    logger.info("Starting with it all")

    wandb_base_tags = [
        "s_" + summary_type,
        "p_" + pipeline_version,
        "m_" + standard_pipeline.named_steps["model"].__class__.__name__,
        "label_" + label_col,
    ]

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
    data, labels = get_data(
        BASE_DATA_DIR,
        study_accessions,
        summary_type,
        pipeline_version,
        label_col,
        metdata_cols_to_use_as_features,
    )

    logger.success("Data obtained")

    # Encode labels. Make sure the positive class is labeled as 1

    label_preprocessor.fit(labels)
    classes = list(label_preprocessor.classes_)
    if positive_class_label in classes:
        positive_class_index = classes.index(positive_class_label)
        if positive_class_index != 1:
            # Swap labels to ensure the desired class is labeled as 1
            classes[1], classes[positive_class_index] = (
                classes[positive_class_index],
                classes[1],
            )
            label_preprocessor.classes_ = np.array(classes)
    encoded_labels = label_preprocessor.transform(labels)

    train_scores = []
    test_scores = []

    outer_cv = outer_cv_config["type"](**outer_cv_config["params"])

    logger.info("Starting with outer cv")

    for i, (train_index, test_index) in enumerate(outer_cv.split(data, encoded_labels)):
        # outer cv data split
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = encoded_labels[train_index], encoded_labels[test_index]

        # With random state defined like this, each experiment is reproducible but the inner cv splits are different per outer cv split
        random_state = RandomState(i)

        inner_cv = inner_cv_config["type"](
            **inner_cv_config["params"], random_state=random_state
        )

        partial_hyp_param_train_with_cv = tune.with_parameters(
            hyp_param_train_with_cv,
            data=X_train,
            labels=y_train,
            cv=inner_cv,
            standard_pipeline=standard_pipeline,
            scoring=scoring,
            best_fit_scorer=best_fit_scorer,
            outer_cv_step=i,
        )

        # Use tune.with_resources only on cluster; skip on local machines
        if is_cluster_environment():
            trainable = tune.with_resources(
                partial_hyp_param_train_with_cv,
                resources=PlacementGroupFactory(
                    [{"CPU": 1, "GPU": 0}] * inner_cv_config["params"]["n_splits"]
                ),
            )
        else:
            trainable = partial_hyp_param_train_with_cv  # Skip resource specification

        tuner = tune.Tuner(
            trainable,
            param_space=tuning_grid,
            run_config=RunConfig(
                name=wandb_params["name"],
                storage_path=str(BASE_DIR / "ray_results"),
                stop=ExperimentPlateauStopper(
                    metric="best_fit_metric", mode=tuning_mode, patience=3
                ),
            ),
            tune_config=tune.TuneConfig(
                search_alg=ConcurrencyLimiter(OptunaSearch(), max_concurrent=3),
                metric="best_fit_metric",
                mode=tuning_mode,
                num_samples=tuning_num_samples,
                trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}",
            ),
        )

        # tuner = GridSearchCV(
        #     standard_pipeline,
        #     tuning_grid,
        #     cv=inner_cv,
        #     scoring=scoring,
        #     refit=best_fit_scorer,
        #     # verbose=3,  # we are logging, so not necessary
        #     n_jobs=-1,
        #     return_train_score=True,
        # )

        tuning_results = tuner.fit()

        # table = wandb.Table(dataframe=pd.DataFrame(tuner.cv_results_).astype(str))
        # wandb.log({f"Hyperparam tuning scores @ outer cv split {i}": table}, step=i)

        best_config = tuning_results.get_best_model().config
        best_model = standard_pipeline.set_params(**best_config)
        best_model.fit(X_train, y_train)

        train_outer_cv_score = get_scores(
            best_model, X_train, y_train, scoring, train_or_test="train"
        )
        test_outer_cv_score = get_scores(
            best_model, X_test, y_test, scoring, train_or_test="test"
        )

        wandb.log(
            {"Outer fold": dict(train_outer_cv_score, **test_outer_cv_score)},
            step=i,
        )

        # tuner_results.append(tuner_cv_result)
        train_scores.append(train_outer_cv_score)
        test_scores.append(test_outer_cv_score)

        # feature importance plot wandb
        if hasattr(best_model.named_steps["model"], "feature_importances_"):
            feature_importances = best_model.named_steps["model"].feature_importances_
            feature_importances_df = pd.DataFrame(
                {"Feature": X_train.columns, "Importance": feature_importances}
            )

            # Log all features
            # wandb.log(
            #     {
            #         f"Feature Importance (All) @ outer cv split {i}": wandb.plot.bar(
            #             wandb.Table(dataframe=feature_importances_df),
            #             "Feature",
            #             "Importance",
            #             title=f"Feature Importance (All) @ outer cv split {i}",
            #         )
            #     },
            #     step=i,
            # )

            # Log top 5 and bottom 5 features
            sorted_feature_importances_df = feature_importances_df.sort_values(
                by="Importance", ascending=False
            )
            best_and_worst_features = pd.concat(
                [
                    sorted_feature_importances_df.head(5),
                    sorted_feature_importances_df.tail(5),
                ]
            )
            wandb.log(
                {
                    f"Feature Importance (Top 5 and Bottom 5) @ outer cv split {i}": wandb.plot.bar(
                        wandb.Table(dataframe=best_and_worst_features),
                        "Feature",
                        "Importance",
                        title=f"Feature Importance (Top 5 and Bottom 5) @ outer cv split {i}",
                    )
                },
                step=i,
            )

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

    # NOT WORKING
    # # Use WandB's plotting capabilities to create bar plots
    # wandb.log(
    #     {
    #         "Train Metrics Summary": wandb.plot.bar(
    #             wandb.Table(dataframe=train_summary_df),
    #             "Metric",
    #             "Mean",
    #             title="Train Metrics Summary",
    #             error_y="Std",
    #         ),
    #         "Test Metrics Summary": wandb.plot.bar(
    #             wandb.Table(dataframe=test_summary_df),
    #             "Metric",
    #             "Mean",
    #             title="Test Metrics Summary",
    #             error_y="Std",
    #         ),
    #     }
    # )

    ray.shutdown()
    logger.success("Done!")
    wandb.finish()


if __name__ == "__main__":
    # fire.Fire(main)

    main("run_configs.simple_rf_baseline",
         study_accessions=['MGYS00003677'],
         summary_type="GO_abundances",
         pipeline_version="v4.1",
         label_col="disease status__biosamples",
         positive_class_label="Sick")