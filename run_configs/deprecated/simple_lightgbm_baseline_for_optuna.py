from lightgbm import LGBMClassifier
from sklearn.calibration import LabelEncoder
from sklearn.metrics import (
    average_precision_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import Normalizer

from src.helper_function import create_pipeline


def get_setup():
    misc_config = {
        "wandb": True,  # whether to use wandb or not
        "wandb_params": {
            "project": "thesis_baselines",
            "group": "RF",  # model name can be useful here
            "name": "RF_baseline",
        },
        "verbose_pipeline": True,  # whether to print verbose output from the pipeline
        "cache_pipeline_steps": False,  # True giving errors
    }

    # outer_cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    outer_cv_config = {
        "type": ShuffleSplit,
        "params": {"n_splits": 2, "test_size": 0.2, "random_state": 42},
    }

    inner_cv_config = {
        "type": ShuffleSplit,
        "params": {
            "n_splits": 2,
            "test_size": 0.2,
        },  # don't provide random_state, as we want to change it per outer fold
    }

    preprocessor_pipeline = create_pipeline(
        [
            ("normalizations_and_transformations", Normalizer(norm="l1")),
            (
                "feature_space_change",
                "passthrough",
            ),  # assumed to be SelectPercentile(MutualInfoClassif)
        ],
        misc_config,
    )

    label_preprocessor = LabelEncoder()

    standard_pipeline = create_pipeline(
        [
            ("preprocessor", preprocessor_pipeline),
            ("model", LGBMClassifier()),
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
    tuning_mode = "maximize"  # "maximize" or "minimize"

    def search_space_sampler(optuna_trial):
        preprocessor__feature_space_change__percentile = optuna_trial.suggest_int(
            "preprocessor__feature_space_change__percentile", 10, 100
        )
        preprocessor__feature_space_change__n_neighbors = optuna_trial.suggest_int(
            "preprocessor__feature_space_change__n_neighbors", 2, 100
        )
        model__boosting_type = optuna_trial.suggest_categorical("model__boosting_type", ["gbdt", "dart"])
        model__n_estimators = optuna_trial.suggest_int("model__n_estimators", 20, 100)
        model__learning_rate = optuna_trial.suggest_float("model__learning_rate", 0.001, 0.5)
        model__num_leaves = optuna_trial.suggest_int("model__num_leaves", 20, 1000)
        model__colsample_bytree = optuna_trial.suggest_float("model__colsample_bytree", 0.4, 1)
        model__subsample = optuna_trial.suggest_float("model__subsample", 0.7, 1)
        model__subsample_freq = optuna_trial.suggest_int("model__subsample_freq", 1, 100)
        model__num_iterations = optuna_trial.suggest_int("model__num_iterations", 50, 300)
        model__max_depth = optuna_trial.suggest_int("model__max_depth", -1, 100)
        model__min_child_samples = optuna_trial.suggest_int("model__min_child_samples", 10, 30)
        model__reg_alpha = optuna_trial.suggest_float("model__reg_alpha", 0.0, 1.0)
        model__reg_lambda = optuna_trial.suggest_float("model__reg_lambda", 0.0, 1.0)

        return {
            "preprocessor__feature_space_change__percentile": preprocessor__feature_space_change__percentile,
            "preprocessor__feature_space_change__n_neighbors": preprocessor__feature_space_change__n_neighbors,
            "model__boosting_type": model__boosting_type,
            "model__n_estimators": model__n_estimators,
            "model__learning_rate": model__learning_rate,
            "model__num_leaves": model__num_leaves,
            "model__colsample_bytree": model__colsample_bytree,
            "model__subsample": model__subsample,
            "model__subsample_freq": model__subsample_freq,
            "model__num_iterations": model__num_iterations,
            "model__max_depth": model__max_depth,
            "model__min_child_samples": model__min_child_samples,
            "model__reg_alpha": model__reg_alpha,
            "model__reg_lambda": model__reg_lambda,
        }

    tuning_num_samples = 2

    return {
        "misc_config": misc_config,
        "outer_cv_config": outer_cv_config,
        "inner_cv_config": inner_cv_config,
        "standard_pipeline": standard_pipeline,
        "label_preprocessor": label_preprocessor,
        "scoring": score_functions,
        "best_fit_scorer": best_fit_scorer,
        "tuning_mode": tuning_mode,
        "search_space_sampler": search_space_sampler,
        "tuning_num_samples": tuning_num_samples,
    }
