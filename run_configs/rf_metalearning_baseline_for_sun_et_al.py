from functools import partial
import os

from imblearn.ensemble import BalancedRandomForestClassifier
from loguru import logger
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import ShuffleSplit
from xgboost import XGBClassifier

import run_configs.optuna_search_space_samplers as sss
from src.helper_function import create_pipeline
from src.models.neural_net import NeuralNetWrapper

# studies interested in:
# HanL_2021
# JieZ_2017
# QinJ_2012
# WangQ_2021
# ZengQ_2021


def get_setup(algorithm):
    misc_config = {
        "wandb": True,  # whether to use wandb or not
        "wandb_params": {
            "project": "thesis_metalearning_inspired_baselines",
            "group": algorithm,  # model name can be useful here
        },
        "verbose_pipeline": True,  # whether to print verbose output from the pipeline
        "cache_pipeline_steps": False,  # True giving errors
    }

    n_outer_splits = 10
    n_inner_splits = 3
    tuning_num_samples = 50

    label_preprocessor = LabelEncoder()

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    logger.info(f"n_cpus found:{n_cpus}")

    model = {
        "RandomForestClassifier": RandomForestClassifier(n_jobs=n_cpus),
        "XGBoost": XGBClassifier(n_jobs=n_cpus),
        "NeuralNet": NeuralNetWrapper(),
        "BalancedRandomForestClassifier": BalancedRandomForestClassifier(n_jobs=n_cpus),
    }[algorithm]

    standard_pipeline = create_pipeline(
        [
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
        "RandomForestClassifier": partial(
            sss.rf_search_space_sampler, best_fit_scorer=best_fit_scorer
        ),
        "XGBoost": sss.xgboost_search_space_sampler,
        "BalancedRandomForestClassifier": partial(
            sss.rf_search_space_sampler, best_fit_scorer=best_fit_scorer
        ),
    }[algorithm]

    return {
        "misc_config": misc_config,
        "n_outer_splits": n_outer_splits,
        "n_inner_splits": n_inner_splits,
        "standard_pipeline": standard_pipeline,
        "label_preprocessor": label_preprocessor,
        "scoring": score_functions,
        "best_fit_scorer": best_fit_scorer,
        "tuning_mode": tuning_mode,
        "search_space_sampler": search_space_sampler,
        "tuning_num_samples": tuning_num_samples,
    }
