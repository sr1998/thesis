def get_setup():
    def search_space_sampler(optuna_trial):
        # Meta-learning specific hyperparameters
        outer_lr_min = optuna_trial.suggest_float("outer_lr_min", 1e-2, 1, log=True)
        outer_lr_max = optuna_trial.suggest_float("outer_lr_max", outer_lr_min, 2, log=True)
        
        # Inner learning rate (currently the same min/max with reduction factor)
        inner_lr = optuna_trial.suggest_float("inner_lr", 1e-4, 1, log=True)    # max and min are the same for now as we use a reduction_factor
        inner_lr_reduction_factor = optuna_trial.suggest_float("inner_lr_reduction_factor", 0.1, 0.9)
        
        # Training parameters
        max_epochs = optuna_trial.suggest_int("max_epochs", 10, 300)
        do_normalization_before_scaling = optuna_trial.suggest_categorical(
            "do_normalization_before_scaling", [True, False]
        )
        scale_factor_before_training = optuna_trial.suggest_float("scale_factor_before_training", 0.1, 10.0, log=True)
        
        # Model architecture hyperparameters
        model__num_layers = optuna_trial.suggest_int("model__num_layers", 1, 5)  # Number of hidden layers
        
        # Dynamic creation of layer sizes based on num_layers
        model__layer_sizes = []
        for i in range(model__num_layers):
            # Layer sizes get progressively smaller
            max_size = 2048 // (2**i) if i < 4 else 128
            min_size = max_size // 8
            layer_size = optuna_trial.suggest_int(f"model__layer_{i}_size", min_size, max_size, step=16)
            model__layer_sizes.append(layer_size)
        
        # Model configuration parameters
        model__dropout_rate = optuna_trial.suggest_float("model__dropout_rate", 0.0, 0.7)
        model__layer_norm = optuna_trial.suggest_categorical("model__layer_norm", [True, False])
        model__batch_norm = optuna_trial.suggest_categorical("model__batch_norm", [True, False])
        # Don't use both layer norm and batch norm together
        if model__layer_norm and model__batch_norm:
            model__batch_norm = False
            
        model__activation = optuna_trial.suggest_categorical(
            "model__activation", ["relu", "leaky_relu", "elu", "gelu", "selu"]
        )
        
        return {
            # Learning rates
            "outer_lr_range": (outer_lr_min, outer_lr_max),
            "inner_lr_range": (inner_lr, inner_lr),  # Same value for now as specified
            "inner_lr_reduction_factor": inner_lr_reduction_factor,
            
            # Training configuration
            "max_epochs": max_epochs,
            "do_normalization_before_scaling": do_normalization_before_scaling,
            "scale_factor_before_training": scale_factor_before_training,
            
            # Model architecture
            "model__num_layers": model__num_layers,
            "model__layer_sizes": model__layer_sizes,
            "model__dropout_rate": model__dropout_rate,
            "model__layer_norm": model__layer_norm,
            "model__batch_norm": model__batch_norm,
            "model__activation": model__activation,
        }
    
    n_outer_splits = 10
    n_inner_splits = 5
    tuning_mode = "maximize"
    best_fit_scorer = "f1"
    tuning_num_samples = 50

    return {
        "n_outer_splits": n_outer_splits,
        "n_inner_splits": n_inner_splits,
        "tuning_mode": tuning_mode,
        "best_fit_scorer": best_fit_scorer,
        "tuning_num_samples": tuning_num_samples,
        "search_space_sampler": search_space_sampler,
    }