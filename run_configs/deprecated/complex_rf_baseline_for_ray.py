from arfs.feature_selection.lasso import LassoFeatureSelection
from ray import tune
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    get_scorer,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import Normalizer

from run_configs.deprecated.predefined_hyp_param_search_spaceearch_space import (
    get_rf_search_space,
    select_percentile_search_space,
    umap_search_space,
)
from src.preprocessing.functions import NumpyReplace, total_sum_scaling
from src.preprocessing.micronorm import GMPR_normalize


def get_setup():
    # TODO Define the misc config. Here is a dummy example
    misc_config = {
        "wandb": True,  # whether to use wandb or not
        "wandb_params": {
            "project": "thesis_baselines",
            "group": "RF",  # model name can be useful here
        },
        "verbose_pipeline": True,  # whether to print verbose output from the pipeline
        "cache_pipeline_steps": True,
    }

    # outer_cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    outer_cv_config = {
        "type": ShuffleSplit,
        "params": {"n_splits": 5, "test_size": 0.2, "random_state": 42},
    }

    inner_cv_config = {
        "type": ShuffleSplit,
        "params": {
            "n_splits": 5,
            "test_size": 0.2,
        },  # don't provide random_state, as we want to change it per outer fold
    }

    # preprocessor_pipeline = create_pipeline(
    #     [
    #         (
    #             "batch_effect_corrections_and_imputations",
    #             "passthrough",
    #         ),  # Together as imputation may be needed after/before batch effect correction depending on whether can handle sparse data or not
    #         ("normalizations_and_transformations", "passthrough"),
    #         ("feature_space_change", "passthrough"),
    #     ],
    #     misc_config,
    # )

    label_preprocessor = LabelEncoder()

    # standard_pipeline = create_pipeline(
    #     [
    #         ("preprocessor", preprocessor_pipeline),
    #         ("model", "passthrough"),
    #     ],
    #     misc_config,
    # )

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
    best_fit_scorer = "f1_weighted"

    # TODO Define the tuning grid. Here is a dummy example
    tuning_grid = [
        {
            "batch_effect_corrections_and_imputations": tune.choice(
                [
                    "passthrough",
                    NumpyReplace(),
                ]
            ),
            "preprocessor__normalizations_and_transformations": tune.sample_from(
                lambda spec: tune.choice(
                    [
                        Normalizer(norm="l1"),
                        Normalizer(norm="l2"),
                        FunctionTransformer(
                            total_sum_scaling,
                            validate=True,
                            feature_names_out="one-to-one",
                        ),
                        "passthrough",
                    ]
                )
                if spec.config["batch_effect_corrections_and_imputations"]
                != "passthrough"
                else tune.choice(
                    [
                        Normalizer(norm="l1"),
                        Normalizer(norm="l2"),
                        FunctionTransformer(
                            total_sum_scaling,
                            validate=True,
                            feature_names_out="one-to-one",
                        ),
                        FunctionTransformer(
                            GMPR_normalize,
                            validate=True,
                            feature_names_out="one-to-one",
                        ),
                        "passthrough",
                    ]
                )
            ),
            "preprocessor__feature_space_change": tune.sample_from(
                lambda spec: tune.choice(
                    [
                        LassoFeatureSelection(n_jobs=1),
                        umap_search_space,
                        select_percentile_search_space,
                    ]
                )
            ),
            **get_rf_search_space(best_fit_scorer),
        },
    ]

    model = RandomForestClassifier()

    tuning_method = ...

    return {
        "misc_config": misc_config,
        "outer_cv_config": outer_cv_config,
        "inner_cv_config": inner_cv_config,
        # "standard_pipeline": standard_pipeline,
        "label_preprocessor": label_preprocessor,
        "scoring": score_functions,
        "best_fit_scorer": best_fit_scorer,
        "tuning_grid": tuning_grid,
        "model": model,
        "tuner": tuning_method,
    }
