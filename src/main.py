import os
from importlib import import_module

import fire
import numpy as np
import optuna
import pandas as pd
from loguru import logger
from numpy.random import RandomState
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_validate

import wandb
from src.data.dataloader import load_data
from src.global_vars import BASE_DATA_DIR
from src.helper_function import (
    get_run_dir_for_experiment,
    get_scores,
)


def get_pipeline(standard_pipeline, search_space_sampler, optuna_trial):
    trial_config = search_space_sampler(optuna_trial)

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

    if trial_config.get("model__oob_score", False):
        trial_config["model__oob_score"] = get_scorer(
            trial_config["model__oob_score"]
        )._score_func

    standard_pipeline = standard_pipeline.set_params(
        **{k: v for k, v in trial_config.items() if "model" in k}
    )

    return standard_pipeline


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


def hyp_param_eval_with_cv(
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
    pipeline = get_pipeline(standard_pipeline, search_space_sampler, trial_config)

    logger.info("pipeline:")
    print(pipeline)

    cross_val_results = cross_validate(
        pipeline,
        data,
        labels,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=1,
    )

    cross_val_res_dict = {
        **{
            k.replace("train_", "train/"): np.mean(v)
            for k, v in cross_val_results.items()
            if "train" in k
        },
        **{
            k.replace("test_", "val/"): np.mean(v)
            for k, v in cross_val_results.items()
            if "test" in k
        },
    }

    wandb.log(cross_val_res_dict, step=outer_cv_step)

    return cross_val_results["test_" + best_fit_scorer].mean()

# TODO class imbalance fix
# TODO feature importance analysis
# TODO save most stuff locally instead of wandb
#FIXME: Improve what is logged to wandb: hyper-param tuning is only showing last one. ...
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

    # get misc config parameters
    use_wandb = misc_config["wandb"]
    wandb_params = misc_config["wandb_params"]
    verbose_pipeline = misc_config.get("verbose_pipeline", True)

    # Set up file logging
    logger_path = get_run_dir_for_experiment(misc_config) / "log.log"
    logger.add(logger_path, colorize=True, level="DEBUG")
    logger.info("Starting with it all")

    job_id = os.getenv("SLURM_JOB_ID")
    wandb_base_tags = [
        "d_" + str(study_accessions),
        "s_" + summary_type,
        "p_" + pipeline_version,
        "m_" + standard_pipeline.named_steps["model"].__class__.__name__,
        "l_" + label_col,
        "j_" + job_id if job_id else "j_local",
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

        optuna_study = optuna.create_study(
            direction=tuning_mode, study_name=f"outer_cv_{i}_for_{wandb.run.name}"
        )
        optuna_study.optimize(
            lambda trial: hyp_param_eval_with_cv(
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
        best_model = get_pipeline(standard_pipeline, search_space_sampler, best_trial)
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

        # # feature importance plot wandb
        # if hasattr(best_model.named_steps["model"], "feature_importances_"):
        #     feature_importances = best_model.named_steps["model"].feature_importances_
        #     print(feature_importances)
        #     feature_importances_df = pd.DataFrame(feature_importances, index=X_train.columns, columns=["Importance"])

        #     # Log all features
        #     # wandb.log(
        #     #     {
        #     #         f"Feature Importance (All) @ outer cv split {i}": wandb.plot.bar(
        #     #             wandb.Table(dataframe=feature_importances_df),
        #     #             "Feature",
        #     #             "Importance",
        #     #             title=f"Feature Importance (All) @ outer cv split {i}",
        #     #         )
        #     #     },
        #     #     step=i,
        #     # )

        #     # Log top 5 and bottom 5 features
        #     sorted_feature_importances_df = feature_importances_df.sort_values(
        #         by="Importance", ascending=False
        #     )
        #     best_and_worst_features = pd.concat(
        #         [
        #             sorted_feature_importances_df.head(5),
        #             sorted_feature_importances_df.tail(5),
        #         ]
        #     )
        #     wandb.log(
        #         {
        #             f"Feature Importance (Top 5 and Bottom 5) @ outer cv split {i}": wandb.plot.bar(
        #                 wandb.Table(dataframe=best_and_worst_features),
        #                 "Feature",
        #                 "Importance",
        #                 title=f"Feature Importance (Top 5 and Bottom 5) @ outer cv split {i}",
        #             )
        #         },
        #         step=i,
        #     )

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

    # main(
    #     "run_configs.simple_rf_baseline_for_optuna",
    #     study_accessions=["MGYS00003677"],
    #     summary_type="GO_abundances",
    #     pipeline_version="v4.1",
    #     label_col="disease status__biosamples",
    #     positive_class_label="Sick",
    # )
