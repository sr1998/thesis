from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    make_scorer,
)
from sklearn.model_selection import ShuffleSplit

from src.helper_function import create_pipeline

# studies interested in:
# HanL_2021
# JieZ_2017
# QinJ_2012
# WangQ_2021
# ZengQ_2021


def get_setup():
    misc_config = {
        "wandb": True,  # whether to use wandb or not
        "wandb_params": {
            "project": "thesis_baselines",
            "group": "RF",  # model name can be useful here
        },
        "verbose_pipeline": True,  # whether to print verbose output from the pipeline
        "cache_pipeline_steps": False,  # True giving errors
    }

    # outer_cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    n_outer_splits = 10
    n_inner_splits = 5
    tuning_num_samples = 100

    outer_cv_config = {
        "type": ShuffleSplit,
        "params": {"n_splits": n_outer_splits, "test_size": 0.2, "random_state": 42},
    }

    inner_cv_config = {
        "type": ShuffleSplit,
        "params": {
            "n_splits": n_inner_splits,
            "test_size": 0.2,
        },  # don't provide random_state, as we want to change it per outer fold
    }

    # preprocessor_pipeline = create_pipeline(
    #     [
    #         ("normalizations_and_transformations", Normalizer(norm="l1")),
    #         (
    #             "feature_space_change",
    #             "passthrough",
    #         ),  # assumed to be SelectPercentile(MutualInfoClassif)
    #     ],
    #     misc_config,
    # )

    label_preprocessor = LabelEncoder()

    standard_pipeline = create_pipeline(
        [
            ("model", RandomForestClassifier()),
        ],
        misc_config,
    )

    score_functions = {
        "accuracy": "accuracy",
        "f1": "f1",
        # "f1_micro": "f1_micro",
        # "f1_weighted": "f1_weighted",
        "roc_auc(_macro)": "roc_auc",
        # "roc_auc_micro": make_scorer(roc_auc_score, average="micro"),
        # "roc_auc_weighted": make_scorer(roc_auc_score, average="weighted"),
        "average_precision(_macro)": make_scorer(average_precision_score),
        # "average_precision_micro": make_scorer(
        #     average_precision_score, average="micro"
        # ),
        # "average_precision_weighted": make_scorer(
        #     average_precision_score, average="weighted"
        # ),
        "precision(_binary)": "precision",
        # "precision_micro": "precision_micro",
        # "precision_weighted": "precision_weighted",
        "recall(_binary)": "recall",
        # "recall_micro": "recall_micro",
        # "recall_weighted": "recall_weighted",
    }
    best_fit_scorer = "f1"
    tuning_mode = "maximize"  # "maximize" or "minimize"

    def search_space_sampler(optuna_trial):
        model__n_estimators = optuna_trial.suggest_int("model__n_estimators", 10, 500)
        model__max_depth = optuna_trial.suggest_int("model__max_depth", 10, 200)
        model__criterion = optuna_trial.suggest_categorical(
            "model__criterion", ["gini", "entropy"]
        )
        model__class_weight = optuna_trial.suggest_categorical(
            "model__class_weight", ["balanced", None]
        )
        model__bootstrap = optuna_trial.suggest_categorical(
            "model__bootstrap", [False, True]
        )
        model__oob_score = optuna_trial.suggest_categorical(
            "model__oob_score", [False, best_fit_scorer]
        )

        return {
            # "preprocessor__feature_space_change__percentile": preprocessor__feature_space_change__percentile,
            # "preprocessor__feature_space_change__n_neighbors": preprocessor__feature_space_change__n_neighbors,
            "model__n_estimators": model__n_estimators,
            "model__max_depth": model__max_depth,
            "model__criterion": model__criterion,
            "model__class_weight": model__class_weight,
            "model__bootstrap": model__bootstrap,
            "model__oob_score": model__oob_score,
        }

    return {
        "misc_config": misc_config,
        "n_outer_splits": n_outer_splits,
        "n_inner_splits": n_inner_splits,
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
