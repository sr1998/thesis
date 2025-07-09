from functools import partial
from math import ceil
import os

import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import FastICA, PCA
from src.data.helper_functions import select_features_by_pc_loadings
from src.preprocessing.functions import pandas_label_encoder
from torch.utils.data import DataLoader
import torch
import torch.multiprocessing as mp
# Must be set before any other multiprocessing code runs
mp.set_start_method('spawn', force=True)

import wandb
from src.data.sun_et_al import BinaryFewShotBatchSampler, KShotBatchSampler, LabelOnlyDataset, MicrobiomeDataset
from src.helper_function import column_rename_for_sun_et_al_metadata, df_str_for_loguru, encode_labels, make_data_balanced_per_study
from src.models import maml_with_l2l, protonet, reptile_with_l2l
from src.models.models import HighlyFlexibleModel


def gpu_collate(batch, device="cuda"):
    """Move batch to GPU after collation."""
    samples, labels = zip(*batch)
    samples = torch.stack(samples)
    labels = torch.stack(labels)
    return samples, labels


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
) -> tuple[maml_with_l2l.MAML, DataLoader, DataLoader, DataLoader]:
    do_normalization_before_scaling = trial_config["do_normalization_before_scaling"]
    scale_factor_before_training = trial_config["scale_factor_before_training"]
    feature_reduction_alg = extra_configs["feature_reduction_alg"]
    feature_reduction_n_components = trial_config["feature_reduction_n_components"]
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    balanced_or_unbalanced = extra_configs["balanced_or_unbalanced"]
    device = extra_configs["device"]

    if feature_reduction_n_components != 0:
            if feature_reduction_alg == "PCA":
                feature_reduction = PCA(n_components=feature_reduction_n_components)
                feature_reduction = feature_reduction.fit(train_data)
                train_data = pd.DataFrame(feature_reduction.transform(train_data), index=train_data.index)
                eval_data = pd.DataFrame(feature_reduction.transform(eval_data), index=eval_data.index)
            elif feature_reduction_alg == "PCA_feat_sel":
                selected_feats, _, _ = select_features_by_pc_loadings(train_data, n_features=feature_reduction_n_components, n_components=500)
                # selected_feats, _, _, _ = select_features_by_pc_loadings(train_data)
                train_data = train_data[selected_feats]
                eval_data = eval_data[selected_feats]
            else:
                raise ValueError(f"Unknown feature reduction algorithm: {feature_reduction_alg}")

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

    # if trial_config["early_stopping_patience"]:
    #     frac = trial_config["early_stopping_fraction"]
    #     label_wise_indices = train_metadata.groupby("label", sort=False).apply(
    #         lambda x: x.sample(frac=frac, random_state=extra_configs["random_seed"])
    #     )
        
    #     label_wise_indices = label_wise_indices.index.get_level_values(1).tolist()
    #     val_data = train_data.loc[label_wise_indices]
    #     val_metadata = train_metadata.loc[label_wise_indices]
    #     train_data = train_data.drop(index=label_wise_indices)
    #     train_metadata = train_metadata.drop(index=label_wise_indices)

    #     val_metadata = val_metadata.loc[val_data.index]

    #     val_labels = encode_labels(LabelEncoder(), val_metadata["label"], extra_configs["positive_class_label"])
    #     val_dataset = LabelOnlyDataset(val_data.values, val_labels.values)
    #     val_sampler = KShotBatchSampler(val_dataset, train_k_shot, include_query=True, query_size="rest", shuffle=False)
    #     val_loader = DataLoader(
    #         val_dataset,
    #         batch_sampler=val_sampler,
    #         num_workers=n_cpus,
    #         pin_memory=torch.cuda.is_available(),
    #         persistent_workers=torch.cuda.is_available(),
    #     )
    # else: 
    #     val_data = pd.DataFrame(columns=train_data.columns)
    #     val_metadata = pd.DataFrame(columns=train_metadata.columns)
    #     val_loader = None

    # # For testing: make limited data for testing of only 3 Groups
    # grouped = train_metadata.groupby("project")
    # train_metadata_new = pd.DataFrame()
    # for i, (group_name, group) in enumerate(grouped):
    #     if i < 3:
    #         train_metadata_new = pd.concat([train_metadata_new, group])
    #     else:
    #         break
    # train_data = train_data.loc[train_metadata_new.index]
    # train_metadata = train_metadata_new


    # filter data such that k_shot is possible
    # grouped_train_metadata = train_metadata.groupby("project", sort=False)
    # idx_to_remove = []
    # for group_name, group in grouped_train_metadata:
    #     grouped_by_label = group.groupby("label", sort=False)
    #     for label_name, label_group in grouped_by_label:
    #         if len(label_group) < train_k_shot * 2:
    #             logger.warning(
    #                 f"Group {group_name} with label {label_name} has only {len(label_group)} samples, which is less than k_shot ({train_k_shot})."
    #             )
    #             new_idx = train_metadata[train_metadata["project"] == group_name].index.tolist()
    #             logger.warning(
    #                 f"Removing {len(new_idx)} samples from group {group_name} with label {label_name}."
    #             )
    #             idx_to_remove.extend(new_idx)
    #             break
    # train_metadata = train_metadata.drop(index=idx_to_remove)
    # train_data = train_data.drop(index=idx_to_remove)

    # n unique groups
    n_unique_groups = train_metadata["project"].nunique()
    logger.info(
        f"Number of unique groups in train metadata: {n_unique_groups}"
    )

    if extra_configs["splitting_method"] == "normal":
        # order the metadata by the index of the data just to be sure
        train_metadata = train_metadata.loc[train_data.index]
        eval_metadata = eval_metadata.loc[eval_data.index]

        train_labels = encode_labels(LabelEncoder(), train_metadata["label"], extra_configs["positive_class_label"])
        eval_labels = encode_labels(LabelEncoder(), eval_metadata["label"], extra_configs["positive_class_label"])

        train_dataset = LabelOnlyDataset(train_data.values, train_labels.values, device=device)
        eval_dataset = LabelOnlyDataset(eval_data.values, eval_labels.values, device=device)
        train_sampler = KShotBatchSampler(train_dataset, train_k_shot, include_query=True)
        eval_sampler = KShotBatchSampler(eval_dataset, eval_k_shot, include_query=True, query_size="rest", shuffle=False)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_sampler=eval_sampler,
            num_workers=0
        )
    elif extra_configs["splitting_method"] == "study_wise":
        # Create Datasets for DataLoader
        train = MicrobiomeDataset(train_data, train_metadata, jitter_fraction=trial_config["jitter_fraction"], target_preprocessor=partial(pandas_label_encoder, positive_class_label=extra_configs["positive_class_label"]), device=device)
        eval = MicrobiomeDataset(
            eval_data, eval_metadata, preselected_support_set=eval_support_sets, jitter_fraction=0.0, target_preprocessor=partial(pandas_label_encoder, positive_class_label=extra_configs["positive_class_label"]), device=device
        )

        # Create DataLoaders
        sampler = BinaryFewShotBatchSampler(
            train,
            train_k_shot,
            include_query=True if algorithm != "Reptile" else False,
            shuffle=True,
        )
        train_loader = DataLoader(train, batch_sampler=sampler, num_workers=4, pin_memory=torch.cuda.is_available(), persistent_workers=torch.cuda.is_available(), collate_fn=partial(gpu_collate, device=device))

        sampler = BinaryFewShotBatchSampler(
            eval,
            eval_k_shot,
            include_query=True,
            shuffle=False,
            shuffle_once=False,
            training=False
        )
        eval_loader = DataLoader(eval, batch_sampler=sampler, num_workers=4, pin_memory=torch.cuda.is_available(), persistent_workers=torch.cuda.is_available(), collate_fn=partial(gpu_collate, device=device))
    else:
        raise ValueError(
            f"Unknown splitting method: {extra_configs['splitting_method']}"
        )

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
        make_output_binary=False if algorithm=="ProtoNet" else True,
    ).to(extra_configs["device"])

    train_labels_unordered = encode_labels(LabelEncoder(), train_metadata["label"], extra_configs["positive_class_label"])
    # class_weights_dict = train_labels_unordered.value_counts(normalize=True).to_dict()
    class_counts = train_labels_unordered.value_counts()
    class_weights_dict = {cls: len(train_labels_unordered) / count 
                     for cls, count in class_counts.items()}
    class_weights = []
    for i in range(len(class_weights_dict)):
        class_weights.append(class_weights_dict[i])
    if algorithm == "MAML":
        pos_class_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
        model = maml_with_l2l.MAML(
            model=model,
            device=extra_configs["device"],
            starting_lr=trial_config["model__starting_lr"],
            scheduler_step=trial_config["model__scheduler_step"],
            scheduler_gamma=trial_config["model__scheduler_gamma"],
            weight_decay=trial_config["model__weight_decay"],
            class_weights_loss_fn=class_weights,
            train_n_gradient_steps=extra_configs["n_gradient_steps"],
            eval_n_gradient_steps=extra_configs["n_gradient_steps"],
            inner_lr_range=trial_config["inner_lr_range"],
            inner_lr_reduction_factor=trial_config["inner_lr_reduction_factor"],
            outer_lr_range=trial_config["outer_lr_range"],
            train_k_shot=train_k_shot,
            eval_k_shot=eval_k_shot,
            loss_fn=extra_configs["loss_fn"](pos_weight=pos_class_weight) if extra_configs["loss_fn"] else None,
        )

    # Not converging at all with some tested hyperparams. Wrong implementation maybe. To be figured out when time allows.
    elif algorithm == "Reptile":
        if "betas" not in trial_config:
            logger.warning(
                "No betas found in trial_config. Using default values (0.9, 0.999)"
            )
            trial_config["betas"] = (0.9, 0.999)

        model = reptile_with_l2l.Reptile(
            model=model,
            train_n_gradient_steps=extra_configs["n_gradient_steps"],
            eval_n_gradient_steps=extra_configs["n_gradient_steps"],
            device=extra_configs["device"],
            inner_lr_range=trial_config["inner_lr_range"],
            inner_lr_reduction_factor=trial_config["inner_lr_reduction_factor"],
            outer_lr_range=trial_config["outer_lr_range"],
            train_k_shot=train_k_shot,
            eval_k_shot=eval_k_shot,
            betas=trial_config["betas"],
            loss_fn=extra_configs["loss_fn"],
            weight_decay=trial_config["model__weight_decay"],
        )
    elif algorithm == "ProtoNet":
        model = protonet.ProtonetTrainer(
            model=model,
            device=extra_configs["device"],
            train_k_shot=train_k_shot,
            eval_k_shot=eval_k_shot,
            starting_lr=trial_config["model__starting_lr"],
            scheduler_step=trial_config["model__scheduler_step"],
            scheduler_gamma=trial_config["model__scheduler_gamma"],
            weight_decay=trial_config["model__weight_decay"],
            class_weights_loss_fn=class_weights
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return model, train_loader, None, eval_loader


def hyp_param_val_for_metalearning(
    algorithm: str,
    inner_loop_splits: dict[int, list[str | list[str]]],
    orig_train_data: pd.DataFrame,
    orig_train_metadata: pd.DataFrame,
    # test_support_sets: dict[str : list[str]],
    train_k_shot: int,
    val_k_shot: int | None,
    search_space_sampler: callable,
    trial: optuna.Trial,
    extra_configs: dict,
    # early_stop_pat=None,
    # early_stop_metric="loss",
):
    if val_k_shot is None:
        val_k_shot = train_k_shot

    cross_val_results = {}
    for _, (val_study_name, val_support_sets) in inner_loop_splits.items():
        val_metadata = orig_train_metadata[
            orig_train_metadata["Project_1"] == val_study_name
        ]
        val_data = orig_train_data.loc[val_metadata.index]
        train_data = orig_train_data.drop(index=val_metadata.index)
        train_metadata = orig_train_metadata.drop(index=val_metadata.index)

        trial_config = search_space_sampler(trial)

        logger.info("Setting up model and dataloaders from trial")
        model, train_loader, val_loader, eval_loader = get_metalearning_model_from_trial(
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

        logger.info("Fitting model")
        train_results, val_results = model.fit(
            train_dataloader=train_loader,
            n_epochs=trial_config["max_epochs"],
            n_parallel_tasks=extra_configs["n_parallel_tasks"],
            val_dataloader=val_loader,
            eval_dataloader=eval_loader,
            early_stopping_patience=trial_config["early_stopping_patience"],
            early_stopping_fraction=trial_config["early_stopping_fraction"],
            # early_stopping_metric=early_stop_metric,
            log_metrics=False,  # Disable wandb logging during optimization
            track_best_f1=extra_configs["track_best_f1"],
        )

        # update keys to include train and val prefixes
        train_results = {f"train/{k}": v for k, v in train_results.items()} if train_results else {}
        val_results = {f"val/{k}": v for k, v in val_results.items()}

        train_results.update(val_results)

        for metric, val in train_results.items():
            if "predictions" in metric or "targets" in metric:
                continue
            if metric not in cross_val_results:
                cross_val_results[metric] = []
            cross_val_results[metric].append(val)
        if "actual_epochs" not in cross_val_results:
            cross_val_results["actual_epochs"] = []
        cross_val_results["actual_epochs"].append(model.current_epoch)

    # Get mean ansu std for all metrics
    wandb_data = {}

    # Add mean train metrics
    mean_train_data = {
        k.replace("train/", f"mean_hyp_param_opt_trains/"): np.mean(v)
        for k, v in cross_val_results.items()
        if "train" in k and "epoch" not in k
    }
    wandb_data.update(mean_train_data)

    # Add mean test metrics
    mean_test_data = {
        k.replace("val/", f"mean_hyp_param_opt_vals/"): np.mean(v)
        for k, v in cross_val_results.items()
        if "val" in k and "epoch" not in k
    }
    wandb_data.update(mean_test_data)

    # Add std train metrics
    std_train_data = {
        k.replace("train/", f"std_hyp_param_opt_trains/"): np.std(v)
        for k, v in cross_val_results.items()
        if "train" in k and "epoch" not in k
    }
    wandb_data.update(std_train_data)

    # Add std test metrics
    std_test_data = {
        k.replace("val/", f"std_hyp_param_opt_vals/"): np.std(v)
        for k, v in cross_val_results.items()
        if "val" in k and "epoch" not in k
    }
    wandb_data.update(std_test_data)
    wandb_data["trial"] = trial.number
    wandb_data["mean_actual_epochs"] = ceil(
        np.mean(cross_val_results["actual_epochs"]).item()
    )

    wandb.log(wandb_data)

    trial.set_user_attr(
        "actual_epochs", ceil(np.mean(cross_val_results["actual_epochs"]).item())
    )
    best_scorer_name = "val/best_" + extra_configs["best_fit_scorer"]
    best_scorer_name = best_scorer_name if best_scorer_name in cross_val_results else "val/" + extra_configs["best_fit_scorer"]
    logger.info(f"best scorer = {best_scorer_name}")
    return np.mean(cross_val_results[best_scorer_name])
