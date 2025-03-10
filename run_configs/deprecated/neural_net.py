from sklearn.calibration import LabelEncoder
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import ShuffleSplit

from run_configs.optuna_search_space_samplers import nn_search_space_sampler
from src.models.neural_net import NeuralNetWrapper


def get_setup():
    """Configuration for neural network training similar to RF configuration."""
    misc_config = {
        "wandb": True,
        "wandb_params": {
            "project": "baseline_predetermined_data_splits",
            "group": "NeuralNet",
        },
        "verbose_pipeline": True,
        "cache_pipeline_steps": False,
    }

    n_outer_splits = 10
    n_inner_splits = 5
    tuning_num_samples = 100

    outer_cv_config = {
        "type": ShuffleSplit,
        "params": {"n_splits": n_outer_splits, "test_size": 0.2, "random_state": 42},
    }

    inner_cv_config = {
        "type": ShuffleSplit,
        "params": {"n_splits": n_inner_splits, "test_size": 0.2},
    }

    label_preprocessor = LabelEncoder()

    # Create neural network as a stand-alone pipeline component
    from src.helper_function import create_pipeline

    standard_pipeline = create_pipeline([("model", NeuralNetWrapper())], misc_config)

    # Define scoring functions - same as RF
    score_functions = {
        "accuracy": "accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": make_scorer(average_precision_score),
        "precision": "precision",
        "recall": "recall",
    }

    best_fit_scorer = "f1"
    tuning_mode = "maximize"

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
        "search_space_sampler": nn_search_space_sampler,
        "tuning_num_samples": tuning_num_samples,
    }
