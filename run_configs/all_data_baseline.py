from functools import partial
import os
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    make_scorer,
)
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import Normalizer
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier

import run_configs.optuna_search_space_samplers as sss
from src.helper_function import create_pipeline
from src.models.neural_net import NeuralNetWrapper
from src.preprocessing.functions import ScaleTransformer


def get_setup(model_name, with_oversampling=True):
    misc_config = {
        "wandb": True,  # whether to use wandb or not
        "wandb_params": {
            "project": "all_data_baseline",
            "group": model_name,  # model name can be useful here
        },
        "verbose_pipeline": True,  # whether to print verbose output from the pipeline
        "cache_pipeline_steps": False,  # True giving errors
    }

    # outer_cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    n_outer_splits = 10
    n_inner_splits = 3
    tuning_num_samples = 25

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

    if with_oversampling and model_name == "BalancedRandomForestClassifier":
        raise ValueError(
            "BalancedRandomForestClassifier should not be used with oversampling"
        )
    
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    logger.info(f"n_cpus found:{n_cpus}")

    model = {
        "RandomForestClassifier": RandomForestClassifier(n_jobs=n_cpus),
        "XGBoost": XGBClassifier(n_jobs=n_cpus),
        "NeuralNet": NeuralNetWrapper(),
        "BalancedRandomForestClassifier": BalancedRandomForestClassifier(n_jobs=n_cpus),
        "TabPFN": TabPFNClassifier(memory_saving_mode=False, n_jobs=n_cpus, ignore_pretraining_limits=True),
    }[model_name]

    standard_pipeline = create_pipeline(
        [
            ("normalizer", Normalizer() if model_name == "NeuralNet" else "passthrough"),
            ("scaler", ScaleTransformer() if model_name == "NeuralNet" else "passthrough"),
            (
                "sampler",
                SMOTE(random_state=42) if with_oversampling else "passthrough",
            ),
            ("model", model),
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

    search_space_sampler = {
        "NeuralNet": sss.nn_search_space_sampler,
        "RandomForestClassifier": partial(sss.rf_search_space_sampler, best_fit_scorer=best_fit_scorer),
        "XGBoost": sss.xgboost_search_space_sampler,
        "BalancedRandomForestClassifier": partial(sss.rf_search_space_sampler, best_fit_scorer=best_fit_scorer),
        "DutchDraw": None,
    }[model_name]

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
