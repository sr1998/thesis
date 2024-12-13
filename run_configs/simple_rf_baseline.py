from arfs.feature_selection.lasso import LassoFeatureSelection
from ray import tune
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.metrics import (
    average_precision_score,
    get_scorer,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import Normalizer

from src.helper_function import create_pipeline
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

    preprocessor_pipeline = create_pipeline(
        [
            ("normalizations_and_transformations", "pass_through"),
            ("feature_space_change", "passthrough"),
        ],
        misc_config,
    )

    label_preprocessor = LabelEncoder()

    standard_pipeline = create_pipeline(
        [
            ("preprocessor", preprocessor_pipeline),
            ("model", RandomForestClassifier()),
        ],
        misc_config,
    )

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
    tuning_mode = "max"  # "max" or "min"

    # TODO Define the tuning grid. Here is a dummy example
    tuning_grid = [
        {
            "preprocessor__normalizations_and_transformations": Normalizer(norm="l1"),
            "preprocessor__feature_space_change": tune.choice(
                "passthrough",
                tune.sample_from(
                    lambda spec: SelectPercentile(
                        mutual_info_classif(
                            discrete_features=False, n_neighbors=tune.randint(2, 100)
                        ),
                        percentile=tune.randint(0, 100),
                    )
                ),
            ),
            "model__n_estimators": tune.randint(
                10, 500
            ),  # Replace get_rf_search_space logic
            "model__max_depth": tune.choice([None, tune.randint(10, 200)]),
            "model__criterion": tune.choice(["gini", "entropy"]),
            "model__class_weight": tune.choice(["balanced", None]),
            "model__oob_score": tune.choice(
                [False, get_scorer(best_fit_scorer)._score_func]
            ),
        }
    ]
    tuning_num_samples = 100

    return {
        "misc_config": misc_config,
        "outer_cv_config": outer_cv_config,
        "inner_cv_config": inner_cv_config,
        "standard_pipeline": standard_pipeline,
        "label_preprocessor": label_preprocessor,
        "scoring": score_functions,
        "best_fit_scorer": best_fit_scorer,
        "tuning_mode": tuning_mode,
        "tuning_grid": tuning_grid,
        "tuning_num_samples": tuning_num_samples
    }
