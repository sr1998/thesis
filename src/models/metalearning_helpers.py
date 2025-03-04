from math import ceil

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.preprocessing import Normalizer
from torch.utils.data import DataLoader

import wandb
from src.data.sun_et_al import BinaryFewShotBatchSampler, MicrobiomeDataset
from src.helper_function import column_rename_for_sun_et_al_metadata, df_str_for_loguru
from src.models import maml_with_l2l
from src.models.models import HighlyFlexibleModel


def get_metalearning_model_from_trial(
    train_data: pd.DataFrame,
    eval_data: pd.DataFrame,
    train_metadata: pd.DataFrame,
    eval_metadata: pd.DataFrame,
    train_k_shot: int,
    eval_k_shot: int,
    eval_support_sets: dict[str, list[str]],
    algorithm: str,
    trial_config: dict[str, object],
    extra_configs: dict[str, object],
) -> tuple[maml_with_l2l.MAML, DataLoader, DataLoader]:
    do_normalization_before_scaling = trial_config["do_normalization_before_scaling"]
    scale_factor_before_training = trial_config["scale_factor_before_training"]

    # normalize the data
    if do_normalization_before_scaling:
        train_data = pd.DataFrame(
            Normalizer().fit_transform(train_data),
            index=train_data.index,
            columns=train_data.columns,
        )
        eval_data = pd.DataFrame(
            Normalizer().fit_transform(eval_data),
            index=eval_data.index,
            columns=eval_data.columns,
        )
        # test_data = pd.DataFrame(
        #     Normalizer().fit_transform(test_data),
        #     index=test_data.index,
        #     columns=test_data.columns,
        # )

    train_data = train_data * scale_factor_before_training
    eval_data = eval_data * scale_factor_before_training

    train_metadata = column_rename_for_sun_et_al_metadata(train_metadata)
    eval_metadata = column_rename_for_sun_et_al_metadata(eval_metadata)

    # Create Datasets for DataLoader
    train = MicrobiomeDataset(train_data, train_metadata)
    eval = MicrobiomeDataset(
        eval_data, eval_metadata, preselected_support_set=eval_support_sets
    )

    # Create DataLoaders
    sampler = BinaryFewShotBatchSampler(
        train, train_k_shot, include_query=True, shuffle=True
    )
    train_loader = DataLoader(train, batch_sampler=sampler)

    sampler = BinaryFewShotBatchSampler(
        eval,
        train_k_shot,
        include_query=True,
        shuffle=False,
        shuffle_once=False,
        training=False,
    )
    eval_loader = DataLoader(eval, batch_sampler=sampler)

    # Get model
    n_input_features = train_data.shape[1]
    assert (
        n_input_features == eval_data.shape[1]  # == test_data.shape[1]
    ), "Number of features of train, test and val must be the same."

    # Create model with the sampled hyperparameters
    model = HighlyFlexibleModel(
        n_input=n_input_features,  # Set based on your dataset
        num_layers=trial_config["model__num_layers"],
        layer_sizes=trial_config["model__layer_sizes"],
        dropout_rate=trial_config["model__dropout_rate"],
        layer_norm=trial_config["model__layer_norm"],
        batch_norm=trial_config["model__batch_norm"],
        activation=trial_config["model__activation"],
    ).to(extra_configs["device"])

    if algorithm == "MAML":
        model = maml_with_l2l.MAML(
            model=model,
            train_n_gradient_steps=extra_configs["n_gradient_steps"],
            eval_n_gradient_steps=extra_configs["n_gradient_steps"],
            device=extra_configs["device"],
            inner_lr_range=trial_config["inner_lr_range"],
            inner_lr_reduction_factor=trial_config["inner_lr_reduction_factor"],
            outer_lr_range=trial_config["outer_lr_range"],
            train_k_shot=train_k_shot,
            eval_k_shot=eval_k_shot,
            loss_fn=extra_configs["loss_fn"],
        )

    # Not converging at all with some tested hyperparams. Wrong implementation maybe. To be figured out when time allows.
    elif algorithm == "Reptile":
        # Instantiate the Reptile meta-learner.
        # reptile = reptile_with_l2l.Reptile(
        #     model=model,
        #     train_n_gradient_steps=n_gradient_steps,
        #     eval_n_gradient_steps=n_gradient_steps,
        #     device=device,
        #     inner_lr_range=inner_lr_range,
        #     inner_lr_reduction_factor=inner_lr_reduction_factor,
        #     outer_lr_range=outer_lr_range,
        #     betas=betas,
        #     k_shot=train_k_shot,
        #     loss_fn=loss_fn,
        # )

        # reptile.fit(
        #     train_dataloader=train_loader,
        #     n_epochs=n_epochs,
        #     n_parallel_tasks=n_parallel_tasks,
        #     evaluate_train=True,
        #     val_dataloader=val_loader,
        # )
        raise NotImplementedError("Reptile is not implemented yet.")
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return model, train_loader, eval_loader


def hyp_param_val_for_metalearning(
    algorithm: str,
    inner_loop_splits: dict[int, tuple[str, list[str]]],
    orig_train_data: pd.DataFrame,
    orig_train_metadata: pd.DataFrame,
    # test_support_sets: dict[str : list[str]],
    train_k_shot: int,
    val_k_shot: int | None,
    search_space_sampler: callable,
    trial: optuna.Trial,
    extra_configs: dict,
    outer_cv_step,
):
    if val_k_shot is None:
        val_k_shot = train_k_shot

    cross_val_results = {}
    for _, (val_study_name, val_support_sets) in inner_loop_splits.items():
        val_metadata = orig_train_metadata[
            orig_train_metadata["Project_1"] == val_study_name
        ]
        val_data = orig_train_data.loc[val_metadata.index]
        train_data = orig_train_data.drop(val_data.index)
        train_metadata = orig_train_metadata.drop(val_metadata.index)

        trial_config = search_space_sampler(trial)

        logger.info("Setting up model and dataloaders from trial")
        model, train_loader, val_loader = get_metalearning_model_from_trial(
            train_data,
            val_data,
            train_metadata,
            val_metadata,
            train_k_shot,
            val_k_shot,
            val_support_sets,
            algorithm,
            trial_config,
            extra_configs,
        )

        early_stop_pat = 10
        early_stop_metric = "loss"

        logger.info("Fitting model")
        _, last_val_result = model.fit(
            train_dataloader=train_loader,
            n_epochs=trial_config["max_epochs"],
            n_parallel_tasks=extra_configs["n_parallel_tasks"],
            eval_dataloader=val_loader,
            early_stopping_patience=early_stop_pat,
            early_stopping_metric=early_stop_metric,
            log_metrics=False,  # Disable wandb logging during optimization
        )

        for metric, val in last_val_result.items():
            if metric == "predictions" or metric == "targets":
                continue
            if metric not in cross_val_results:
                cross_val_results[metric] = []
            cross_val_results[metric].append(val)
        cross_val_results["actual_epochs"] = model.current_epoch + 1

    # Get mean and std for all metrics
    cross_val_results_mean = {k: np.mean(v) for k, v in cross_val_results.items()}
    cross_val_results_std = {k: np.std(v) for k, v in cross_val_results.items()}
    wandb_data = {"mean/" + k: v for k, v in cross_val_results_mean.items()}
    wandb_data.update({"std/" + k: v for k, v in cross_val_results_std.items()})

    # log to wandb
    wandb.log({str(outer_cv_step) + "trial_summary": wandb_data, "trial": trial.number})

    trial.set_user_attr(
        "actual_epochs", ceil(np.mean(cross_val_results["actual_epochs"]).item())
    )
    return np.mean(wandb_data["mean/" + extra_configs["best_fit_scorer"]])
