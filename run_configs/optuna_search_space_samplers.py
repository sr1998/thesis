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

def nn_search_space_sampler(optuna_trial):
        # model__n_epochs = optuna_trial.suggest_int("model__n_epochs", 1, 200)
        model__batch_size = optuna_trial.suggest_int("model__batch_size", 2, 32, step=2)
        model__lr = optuna_trial.suggest_float("model__lr", 1e-5, 1e-2, log=True)
        # model__scale_factor = optuna_trial.suggest_float("model__scale_factor", 1.0, 1000.0, log=True)
        model__num_layers = optuna_trial.suggest_int("model__num_layers", 1, 4)
        model__dropout_rate = optuna_trial.suggest_float(
            "model__dropout_rate", 0.1, 0.7
        )
        # model__layer_norm = optuna_trial.suggest_categorical(
        #     "model__layer_norm", [True, False]
        # )
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
            "model__n_epochs": 100,
            "model__batch_size": model__batch_size,
            "model__lr": model__lr,
            "do_normalization_before_scaling": False,
            "model__scale_factor": 1,
            "model__num_layers": model__num_layers,
            "model__layer_sizes": model__layer_sizes,
            "model__dropout_rate": model__dropout_rate,
            "model__layer_norm": True,
            "model__batch_norm": False,
            "model__activation": "relu",
        }
