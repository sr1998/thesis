from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, make_scorer, roc_auc_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import Normalizer

from src.helper_function import create_pipeline


def get_setup():
    # TODO Define the misc config. Here is a dummy example
    misc_config = {
        "wandb": True,  # whether to use wandb or not
        "wandb_params": {
            "project": "thesis_baselines",
            "name": "...",  # dataset name can be useful here
            "group": "...",  # model name can be useful here
        },
        "verbose_pipeline": True,  # whether to print verbose output from the pipeline
        "cache_pipeline_steps": True,
    }

    # TODO Define the data load config. Here is a dummy example
    data_load_config = {
        # This will reuse cross-validation splits if this same experiment has been done before
        "load_from_cache_if_available": True,  # TODO: implement this
        # Leave out if all studies desired; if provided, only the summaries of those studies are used
        # if studies are given, they have to contain the desired summary given by study_download_label_start
        "study_accessions": "...",  # for now only one study is supported
        # indicates what summary file to use for the studies
        # possible values:
        # - GO_abundances
        # - GO-slim_abundances
        # - phylum_taxonomy_abundances_SSU
        # - taxonomy_abundances_SSU
        # - IPR_abundances
        # ... (see MGnify API for more)
        "summary_type": "...",
        # indicates what pipeline version to use
        # possible values:
        # - v3.0
        # - v4.0
        # - v4.1
        # - v5.0
        "pipeline_version": "...",
        # metadata columns to use; leave empty for none, or give value "all" for all columns
        "metdata_cols_to_use_as_features": [],
        # label column for classification
        "label_col": "...",
        # positive class label
        "positive_class_label": "...",
    }

    # TODO Define the cross-validation configurations
    outer_cv_config = {
        "type": ShuffleSplit,
        "params": {"n_splits": 2, "test_size": 0.2, "random_state": 42},
    }

    # TODO Define the inner cross-validation configurations
    inner_cv_config = {
        "type": ShuffleSplit,
        "params": {
            "n_splits": 2,
            "test_size": 0.2,
        },  # don't provide random_state, as we want to change it per outer fold
    }

    preprocessor_pipeline = create_pipeline(
        [
            ("batch_effect_corrections", "passthrough"),
            ("imputations", "passthrough"),
            ("normalizations_and_transformations", "passthrough"),
            ("feature_space_change", "passthrough"),
        ],
        misc_config,
    )

    # TODO Define the label preprocessor. Here is a useful example
    label_preprocessor = LabelEncoder()

    standard_pipeline = create_pipeline(
        [
            ("preprocessor", preprocessor_pipeline),
            ("model", "passthrough"),
        ],
        misc_config,
    )

    # TODO Define the tuning grid. Here is a dummy example
    tuning_grid = [
        {
            "preprocessor__batch_effect_corrections": ["passthrough"],
            "preprocessor__imputations": ["passthrough"],
            "preprocessor__normalizations_and_transformations": [
                create_pipeline([("l1_normalizer", Normalizer())], misc_config)
            ],
            "preprocessor__normalizations_and_transformations__l1_normalizer__norm": [
                "l1"
            ],
            "preprocessor__feature_space_change": ["passthrough"],
            "model": [RandomForestClassifier()],
            "model__n_estimators": [100],
            "model__max_depth": [100],
        }
    ]

    # TODO Define the scoring functions. Here is a useful example
    score_functions = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1(_binary)": "f1",
        "f1_macro": "f1_macro",
        "f1_micro": "f1_micro",
        "f1_weighted": "f1_weighted",
        "roc_auc(_macro)": "roc_auc",
        "roc_auc_micro": make_scorer(roc_auc_score, average="micro"),
        "roc_auc_weighted": make_scorer(roc_auc_score, average="weighted"),
        "average_precision(_macro)": make_scorer(average_precision_score),
        "average_precision_micro": make_scorer(
            average_precision_score, average="micro"
        ),
        "average_precision_weighted": make_scorer(
            average_precision_score, average="weighted"
        ),
        "precision(_binary)": "precision",
        "precision_macro": "precision_macro",
        "precision_micro": "precision_micro",
        "precision_weighted": "precision_weighted",
        "recall(_binary)": "recall",
        "recall_micro": "recall_micro",
        "recall_macro": "recall_macro",
        "recall_weighted": "recall_weighted",
    }

    # TODO Define the best fit scorer. Here is a useful example
    best_fit_scorer = "f1_weighted"

    # tuner = GridSearchCV(
    #     estimator=standard_pipeline,
    #     param_grid=tuning_grid,
    #     cv=inner_cv_config,
    #     n_jobs=-1,
    #     verbose=2 if misc_config.get("verbose_pipeline", True) else 0,
    #     scoring=score_functions,
    #     refit=best_fit_scorer,  # This should be a string indicating the metric to use for refitting. We want refitting in any case.
    #     return_train_score=True,
    # )

    return {
        "misc_config": misc_config,
        "data_load_config": data_load_config,
        "outer_cv_config": outer_cv_config,
        "inner_cv_config": inner_cv_config,
        "ml_pipeline": standard_pipeline,
        "label_preprocessor": label_preprocessor,
        "tuning_grid": tuning_grid,
        "scoring": score_functions,
        "best_fit_scorer": best_fit_scorer,
    }
