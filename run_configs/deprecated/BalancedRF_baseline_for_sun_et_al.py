from sklearn.calibration import LabelEncoder
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import Normalizer

from run_configs.optuna_search_space_samplers import rf_search_space_sampler
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

    n_outer_splits = 10
    n_inner_splits = 5
    tuning_num_samples = 100

    # outer_cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
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
            ("model", BalancedRandomForestClassifier()),
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

    tuning_num_samples = 100

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
        "search_space_sampler": rf_search_space_sampler,
        "tuning_num_samples": tuning_num_samples,
    }
