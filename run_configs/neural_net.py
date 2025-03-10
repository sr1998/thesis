from sklearn.calibration import LabelEncoder
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.model_selection import ShuffleSplit

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

    # Neural network hyperparameter search space
    def nn_search_space_sampler(optuna_trial):
        model__n_epochs = optuna_trial.suggest_int("model__n_epochs", 1, 200)
        model__batch_size = optuna_trial.suggest_int("model__batch_size", 2, 32, step=2)
        model__lr = optuna_trial.suggest_float("model__lr", 1e-5, 1e-2, log=True)
        # model__scale_factor = optuna_trial.suggest_float("model__scale_factor", 1.0, 1000.0, log=True)
        model__num_layers = optuna_trial.suggest_int("model__num_layers", 1, 4)
        model__dropout_rate = optuna_trial.suggest_float(
            "model__dropout_rate", 0.1, 0.7
        )
        model__layer_norm = optuna_trial.suggest_categorical(
            "model__layer_norm", [True, False]
        )
        # model__batch_norm = optuna_trial.suggest_categorical("model__batch_norm", [True, False])
        # model__activation = optuna_trial.suggest_categorical("model__activation", ["relu", "leaky_relu", "elu", "gelu", "selu"])
        base_size = optuna_trial.suggest_int(
            "model__base_size", 16, 1024, step=16
        )  # Much smaller maximum
        reduction_factor = optuna_trial.suggest_float(
            "model__reduction_factor", 1.0, 3.0
        )

        # Dynamically generate layer sizes
        model__layer_sizes = []
        for i in range(model__num_layers):
            # Calculate size based on layer position
            if i == 0:
                # First layer size based on base_size
                max_size = base_size
            else:
                # Subsequent layers get progressively smaller
                max_size = max(8, int(model__layer_sizes[i - 1] / reduction_factor))

            min_size = max(8, max_size // 4)  # Allow much smaller minimum sizes

            # Suggest layer size
            layer_size = optuna_trial.suggest_int(
                f"model__layer_{i}_size", min_size, max_size, step=8
            )
            model__layer_sizes.append(layer_size)

        return {
            "model__n_epochs": model__n_epochs,
            "model__batch_size": model__batch_size,
            "model__lr": model__lr,
            "do_normalization_before_scaling": False,
            "model__scale_factor": 1,
            "model__num_layers": model__num_layers,
            "model__layer_sizes": model__layer_sizes,
            "model__dropout_rate": model__dropout_rate,
            "model__layer_norm": model__layer_norm,
            "model__batch_norm": False,
            "model__activation": "relu",
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
        "search_space_sampler": nn_search_space_sampler,
        "tuning_num_samples": tuning_num_samples,
    }
