def xgboost_search_space_sampler(optuna_trial):
    model__learning_rate = optuna_trial.suggest_float("model__learning_rate", 0.2, 1.0)
    model__gamma = optuna_trial.suggest_int("model__gamma", 0, 3)
    model__max_depth = optuna_trial.suggest_int("model__max_depth", 3, 8)
    model__reg_lambda = optuna_trial.suggest_float("model__reg_lambda", 0.0, 1.0)
    model__reg_alpha = optuna_trial.suggest_float("model__reg_alpha", 0.0, 1.0)

    return {
        "model__learning_rate": model__learning_rate,
        "model__gamma": model__gamma,
        "model__max_depth": model__max_depth,
        "model__reg_lambda": model__reg_lambda,
        "model__reg_alpha": model__reg_alpha,
    }


def rf_search_space_sampler(optuna_trial, best_fit_scorer):
    model__n_estimators = optuna_trial.suggest_int("model__n_estimators", 10, 500, step=10)
    model__max_depth = optuna_trial.suggest_int("model__max_depth", 10, 200, step=10)
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


def nn_search_space_sampler(optuna_trial):
    if optuna_trial is None:
        return {
            "model__n_epochs": 500,
            "model__batch_size": 16,
            "model__lr": 1e-3,
            "do_normalization_before_scaling": True,
            "scaler__scale_factor": 100,
            "model__num_layers": 2,
            "model__layer_sizes": None,
            "model__dropout_rate": 0.5,
            "model__layer_norm": False,
            "model__batch_norm": True,
            "model__activation": "relu",
            "model__log_metrics_every_n_epoch": 10,
            "model__log_gradients_every_n_epoch": 10,
        }
    scale_factor_before_training = optuna_trial.suggest_int("scale_factor_before_training", 1, 1001, step=100)

    # model__n_epochs = optuna_trial.suggest_int("model__n_epochs", 1, 200)
    model__batch_size = optuna_trial.suggest_int("model__batch_size", 2, 64, step=2)
    model__lr = optuna_trial.suggest_float("model__lr", 1e-5, 1e-2, log=True)
    model__num_layers = optuna_trial.suggest_int("model__num_layers", 1, 4)
    model__dropout_rate = optuna_trial.suggest_float("model__dropout_rate", 0.0, 0.7, step=0.1)
    # model__layer_norm = optuna_trial.suggest_categorical(
    #     "model__layer_norm", [True, False]
    # )
    # model__batch_norm = optuna_trial.suggest_categorical("model__batch_norm", [True, False])
    # model__activation = optuna_trial.suggest_categorical("model__activation", ["relu", "leaky_relu", "elu", "gelu", "selu"])
    base_size = optuna_trial.suggest_int(
        "model__base_size", 16, 1008, step=32
    )  # Much smaller maximum
    reduction_factor = optuna_trial.suggest_float("model__reduction_factor", 1.0, 3.0, step=0.5)

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
        "model__n_epochs": 100,
        "model__batch_size": model__batch_size,
        "model__lr": model__lr,
        "do_normalization_before_scaling": True,
        "scaler__scale_factor": scale_factor_before_training,
        "model__num_layers": model__num_layers,
        "model__layer_sizes": model__layer_sizes,
        "model__dropout_rate": model__dropout_rate,
        "model__layer_norm": True,
        "model__batch_norm": False,
        "model__activation": "relu",
    }


def maml_search_space_sampler(optuna_trial):
    if optuna_trial is None:
        return {
            # Learning rates
            "outer_lr_range": (1, 1),
            "inner_lr_range": (0.5, 0.5),  # Same value for now as specified
            "inner_lr_reduction_factor": 2,
            # Training configuration
            "max_epochs": 100,
            "do_normalization_before_scaling": True,
            "scale_factor_before_training": 100,
            # Model architecture
            "model__num_layers": 2,
            "model__layer_sizes": None,
            "model__dropout_rate": 0.5,
            "model__layer_norm": False,
            "model__batch_norm": True,
            "model__activation": "relu",
            "model__weight_decay": 0.0,
        } 
    # Meta-learning specific hyperparameters
    # outer_lr_min = optuna_trial.suggest_float("outer_lr_min", 1e-3, 2)
    # outer_lr_max = optuna_trial.suggest_float("outer_lr_max", outer_lr_min, 2)

    # Inner learning rate (currently the same min/max with reduction factor)
    # max and min are the same for now as we use a reduction_factor
    # inner_lr = optuna_trial.suggest_float("inner_lr", 1e-6, 1e-4)
    inner_lr_reduction_factor = optuna_trial.suggest_int(
        "inner_lr_reduction_factor", 1, 10
    )   # Division used with reduction factor

    # Training parameters
    max_epochs = optuna_trial.suggest_int("max_epochs", 10, 300)   # Fixed for now as we use early stopping
    # do_normalization_before_scaling = optuna_trial.suggest_categorical(
    #     "do_normalization_before_scaling", [True, False]
    # )
    scale_factor_before_training = optuna_trial.suggest_int("scale_factor_before_training", 1, 1000, step=100)

    # Model architecture hyperparameters
    model__num_layers = optuna_trial.suggest_int(
        "model__num_layers", 1, 5
    )  # Number of hidden layers

    # Option 2: Use a base size parameter for more control
    base_size = optuna_trial.suggest_int(
        "model__base_size", 16, 1024, step=16
    )  # Much smaller maximum
    reduction_factor = optuna_trial.suggest_float("model__reduction_factor", 1.0, 3.0)

    # Dynamic creation of layer sizes based on num_layers
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

    # Model configuration parameters
    model__dropout_rate = optuna_trial.suggest_float("model__dropout_rate", 0.0, 0.7)
    model__layer_norm = optuna_trial.suggest_categorical(
        "model__layer_norm", [True, False]
    )
    model__weight_decay = optuna_trial.suggest_float("model__weight_decay", 0.0, 1.0)
    model__batch_norm = optuna_trial.suggest_categorical("model__batch_norm", [True, False])
    # # Don't use both layer norm and batch norm together
    if model__layer_norm and model__batch_norm:
        model__batch_norm = False

    # model__activation = optuna_trial.suggest_categorical(
    #     "model__activation", ["relu", "leaky_relu", "elu", "gelu", "selu"]
    # )

    return {
        # Learning rates
        "outer_lr_range": (1, 1),
        "inner_lr_range": (0.5, 0.5),  # Same value for now as specified
        "inner_lr_reduction_factor": inner_lr_reduction_factor,
        # Training configuration
        "max_epochs": max_epochs,
        "do_normalization_before_scaling": True,
        "scale_factor_before_training": scale_factor_before_training,
        # Model architecture
        "model__num_layers": model__num_layers,
        "model__layer_sizes": model__layer_sizes,
        "model__dropout_rate": model__dropout_rate,
        "model__layer_norm": model__layer_norm,
        "model__batch_norm": model__batch_norm,
        "model__activation": "relu",
        "model__weight_decay": model__weight_decay,
    }


def reptile_search_space_sampler(optuna_trial):
    maml_configs = maml_search_space_sampler(optuna_trial)
    maml_configs["betas"] = (
        optuna_trial.suggest_float("betas_0", 0.0, 1),
        optuna_trial.suggest_float("betas_1", 0.0, 1),
    )
    return maml_configs
