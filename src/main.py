from importlib import import_module

import fire
import numpy as np
import pandas as pd
from loguru import logger
from numpy.random import RandomState
from sklearn.model_selection import GridSearchCV

import wandb
from src.data.dataloader import load_data
from src.global_vars import BASE_DATA_DIR
from src.helper_function import (
    get_run_dir_for_experiment,
    get_scores,
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


# def build_preprocessor_pipeline(
#     cacher: Memory, *, verbose: bool = False, **kwargs
# ) -> Pipeline:
#     batch_correction_methods = kwargs.get("batch_correction_methods", [])
#     imputations = kwargs.get("imputations", [])
#     normalizations_and_transformations = kwargs.get(
#         "normalizations_and_transformations", []
#     )
#     feature_space_changes = kwargs.get("feature_space_changes", [])

#     # Create a pipeline for batch correction methods
#     p = []
#     for batch_corrector in batch_correction_methods:
#         pass
#     batch_correction_methods = Pipeline(p)

#     # Create a pipeline for imputation methods
#     p = []
#     for imputation in imputations:
#         pass
#     imputations = Pipeline(p)

#     # Create a pipeline for normalization and transformation methods
#     p = []
#     for n_or_t in normalizations_and_transformations:
#         if n_or_t["type"] == "total_sum_scaling":
#             p.append((n_or_t, Normalizer(**n_or_t["params"])))
#         elif n_or_t["type"] == "...":
#             pass
#     normalizations_and_transformations = Pipeline(p)

#     # Create a pipeline for feature space changes
#     p = []
#     for feature_space_change in feature_space_changes:
#         pass
#     feature_space_changes = Pipeline(p)

#     return Pipeline(
#         [
#             ("batch_correction_methods", batch_correction_methods),
#             ("imputations", imputations),
#             ("normalizations_and_transformations", normalizations_and_transformations),
#             ("feature_space_changes", feature_space_changes),
#         ]
#     )


# def build_model_pipeline(
#     cacher: Memory, *, verbose: bool = False, **kwargs
# ) -> Pipeline:


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
    wandb_run_name: str | None = None,
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
    # can we change SLURM stdout and stderr file?
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
        tuning_grid,
    ) = setup.values()

    # get misc config parameters
    use_wandb = misc_config["wandb"]
    wandb_params = misc_config["wandb_params"]
    wandb_params["name"] = wandb_run_name or study_accessions
    verbose_pipeline = misc_config.get("verbose_pipeline", True)

    # Set up file logging
    logger_path = get_run_dir_for_experiment(misc_config) / "log.log"
    logger.add(logger_path, colorize=True, level="DEBUG")

    wandb_base_tags = [
        "s_" + summary_type,
        "p_" + pipeline_version,
        "m_" + tuning_grid[0]["model"][0].__class__.__name__,
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

    # Load data
    data, labels = get_data(
        BASE_DATA_DIR,
        study_accessions,
        summary_type,
        pipeline_version,
        label_col,
        metdata_cols_to_use_as_features,
    )

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

    # nested_score = cross_validate(
    #     tuner,
    #     data,
    #     encoded_labels,
    #     cv=outer_cv,
    #     scoring=score_functions,
    #     return_train_score=True,
    #     return_indices=True,
    # )

    train_scores = []
    test_scores = []

    outer_cv = outer_cv_config["type"](**outer_cv_config["params"])

    for i, (train_index, test_index) in enumerate(outer_cv.split(data, encoded_labels)):
        # outer cv data split
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = encoded_labels[train_index], encoded_labels[test_index]

        # With random state defined like this, each experiment is reproducible but the inner cv splits are different per outer cv split
        random_state = RandomState(i)

        inner_cv = inner_cv_config["type"](
            **inner_cv_config["params"], random_state=random_state
        )
        tuner = GridSearchCV(
            estimator=standard_pipeline,
            param_grid=tuning_grid,
            cv=inner_cv,
            scoring=scoring,
            refit=best_fit_scorer,
            # verbose=3,  # we are logging, so not necessary
            n_jobs=-1,
            return_train_score=True,
        )

        tuner = tuner.fit(X_train, y_train)

        table = wandb.Table(dataframe=pd.DataFrame(tuner.cv_results_).astype(str))
        wandb.log({f"Hyperparam tuning scores @ outer cv split {i}": table}, step=i)

        best_model = tuner.best_estimator_
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

    logger.success("Done!")
    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)
