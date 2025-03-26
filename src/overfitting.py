import os
from importlib import import_module
from pathlib import Path

import fire
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from loguru import logger
from sklearn.preprocessing import Normalizer
from torch import nn

import wandb
from src.data.sun_et_al import KShotBatchSampler, LabelOnlyDataset
from src.global_vars import BASE_DATA_DIR
from src.helper_function import (
    column_rename_for_sun_et_al_metadata,
    encode_labels,
    get_run_dir_for_experiment,
)
from src.models import maml_with_l2l
from src.models.metalearning_helpers import get_metalearning_model_from_trial
from src.models.models import HighlyFlexibleModel


def main(
    # model_script: str,                    # optimization
    # model_name: str,                      # optimization
    datasource: str,
    algorithm: str,
    abundance_file: str | Path,
    metadata_file: str | Path,
    test_study: str,
    balanced_or_unbalanced: str,
    # val_study: list,                      # random selection done
    # outer_lr_range: tuple[float, float],  # optimization
    # inner_lr_range: tuple[float, float],  # optimization
    # inner_rl_reduction_factor: int,       # optimization
    n_gradient_steps: int,  # TODO Could be a hyperparam
    n_parallel_tasks: int,  # TODO Could be a hyperparam
    train_k_shot: int,
    positive_class_label: str,
    # eval_k_shot: int = None,              # skip
    # n_components_reduction_factor: int = 0,  # 0 or 1 for no PCA at all   # skip
    # use_cached_pca: bool = False,         # skip
    # do_normalization_before_scaling: bool = True, # optimization
    # scale_factor_before_training: int = 100,      # optimization
    loss_fn: str = "BCELog",
    use_wandb: bool = True,
    features_to_use: list[str] = None,
    early_stop_patience: int = None,
    early_stop_metric: str = "loss",
    track_best_f1: bool = True,
):
    config_script = "run_configs.metalearning"
    config_module = import_module(config_script)
    setup = config_module.get_setup(algorithm)
    (
        n_outer_splits,
        n_inner_splits,
        tuning_mode,
        best_fit_scorer,
        tuning_num_samples,
        search_space_sampler,
    ) = setup.values()

    if loss_fn == "BCELog":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Loss function not recognized.")

    data_root_dir = BASE_DATA_DIR / "sun_et_al_data"

    # Read data and metadata
    sun_et_al_abundance = pd.read_csv(
        f"{data_root_dir}/{abundance_file}",
        index_col=0,
        header=0,
    )
    sun_et_al_metadata = pd.read_csv(
        f"{data_root_dir}/{metadata_file}",
        index_col=0,
        header=0,
    )
    sun_et_al_metadata = sun_et_al_metadata.loc[sun_et_al_abundance.index]

    encoded_labels = encode_labels(
        LabelEncoder(), sun_et_al_metadata["Group"], positive_class_label=positive_class_label
    ).values

    train_data, test_data, train_labels, test_labels = train_test_split(
        sun_et_al_abundance,
        encoded_labels,
        test_size=0.2,
        random_state=42,
        stratify=encoded_labels if balanced_or_unbalanced == "balanced" else None,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Set up file logging
    # logger_path = get_run_dir_for_experiment(misc_config) / "log.log"
    # logger.add(logger_path, colorize=True, level="DEBUG")
    # logger.info("Setting up everything")

    # Set up wandb
    job_id = os.getenv("SLURM_JOB_ID")
    array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
    array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    tax_level = abundance_file.split("_")[1]
    config = {
        # "model_name": model_name,
        "datasource": datasource,
        "algorithm": algorithm,
        "abundance_file": abundance_file,
        "metadata_file": metadata_file,
        "test_study": test_study,
        "balanced_or_unbalanced": balanced_or_unbalanced,
        # "val_study": val_study,
        # "outer_lr_range": outer_lr_range,
        # "inner_lr_range": inner_lr_range,
        # "inner_rl_reduction_factor": inner_rl_reduction_factor,
        "n_gradient_steps": n_gradient_steps,
        "n_parallel_tasks": n_parallel_tasks,
        "train_k_shot": train_k_shot,
        # "eval_k_shot": eval_k_shot,
        # "n_components_reduction_factor": n_components_reduction_factor,
        # "use_cache_pca": use_cached_pca,
        # "do_normalization_before_scaling": do_normalization_before_scaling,
        # "scale_factor_before_training": scale_factor_before_training,
        "loss_fn": loss_fn,
        "use_wandb": use_wandb,
        "device": device,
        "features_to_use": features_to_use,
        # "model_script": model_script,
        "n_outer_splits": n_outer_splits,
        "n_inner_splits": n_inner_splits,
        "tuning_mode": tuning_mode,
        "best_fit_scorer": best_fit_scorer,
        "tuning_num_samples": tuning_num_samples,
        # "search_space_sampler": search_space_sampler,
        "job_id": job_id,
        "array_job_id": array_job_id,
        "array_task_id": array_task_id,
    }
    wandb_base_tags = [
        str(test_study),
        algorithm,
        tax_level,
        str(train_k_shot) + "_shot",
        datasource,
        balanced_or_unbalanced,
        # "e_k" + str(eval_k_shot),
    ]

    wandb_name = f"TS{test_study}_TK{train_k_shot}_{balanced_or_unbalanced}_{datasource}_{algorithm}_T{tax_level}_{array_job_id or job_id}"
    run_dir = get_run_dir_for_experiment(
        "metalearning", algorithm, test_study, wandb_name
    )

    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project="overfitting",
            name=wandb_name,
            config=config,
            notes=str(config),
            group=algorithm,
            tags=wandb_base_tags,
        )
    else:
        wandb.init(
            name=wandb_name,
            mode="disabled",
            config=config,
            notes=str(config),
            project="overfitting",
            group=algorithm,
            tags=wandb_base_tags,
        )

    logger.success("wandb init done")

    train_scores = []
    test_scores = []

    trial_config = search_space_sampler(None)

    do_normalization_before_scaling = trial_config["do_normalization_before_scaling"]
    scale_factor_before_training = trial_config["scale_factor_before_training"]
    
    if balanced_or_unbalanced == "balanced":
        raise NotImplementedError("Balanced data not implemented yet.")

    # normalize the data
    if do_normalization_before_scaling:
        train_data = Normalizer().fit_transform(train_data)
        test_data = Normalizer().fit_transform(test_data)

    train_data = train_data * scale_factor_before_training
    test_data = test_data * scale_factor_before_training

    # Get model
    n_input_features = train_data.shape[1]
    assert (
        n_input_features == test_data.shape[1]  # == test_data.shape[1]
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
    ).to(config["device"])

    if algorithm == "MAML":
        model = maml_with_l2l.MAML(
            model=model,
            train_n_gradient_steps=config["n_gradient_steps"],
            eval_n_gradient_steps=config["n_gradient_steps"],
            device=config["device"],
            inner_lr_range=trial_config["inner_lr_range"],
            inner_lr_reduction_factor=trial_config["inner_lr_reduction_factor"],
            outer_lr_range=trial_config["outer_lr_range"],
            train_k_shot=train_k_shot,
            eval_k_shot=train_k_shot,
            loss_fn=config["loss_fn"],
            weight_decay=trial_config["model__weight_decay"],
        )

    # tensorize and create dataloaders
    train_dataset = LabelOnlyDataset(train_data, train_labels)
    test_dataset = LabelOnlyDataset(test_data, test_labels)
    train_sampler = KShotBatchSampler(train_dataset, train_k_shot, include_query=True)
    test_sampler = KShotBatchSampler(test_dataset, train_k_shot, include_query=True, query_size="rest", shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=0,
        pin_memory=True,
    )

    train_res, test_res = model.fit(
        train_dataloader=train_loader,
        n_epochs=trial_config["max_epochs"],
        n_parallel_tasks=n_parallel_tasks,
        eval_dataloader=test_loader,
        val_or_test="test",
        log_metrics=True,
        score_name_prefix="fit",
        save_best_model_path=None,
        track_best_f1=track_best_f1
    )

    train_res = (
        {k: v for k, v in train_res.items() if k != "predictions" and k != "targets"}
        if train_res
        else {}
    )
    test_res = {
        k: v for k, v in test_res.items() if k != "predictions" and k != "targets"
    }

    train_scores.append(train_res)
    test_scores.append(test_res)

    # log overall results to wandb
    train_scores = pd.DataFrame(train_scores)
    test_scores = pd.DataFrame(test_scores)
    train_mean = train_scores.mean() if not train_scores.empty else pd.Series()
    test_mean = test_scores.mean()
    train_std = train_scores.std() if not train_scores.empty else pd.Series()
    test_std = test_scores.std()

    # Log bar plots for train and test metrics
    train_summary_df = pd.DataFrame(
        {"Metric": train_mean.index, "Mean": train_mean.values, "Std": train_std.values}
    ) if not train_scores.empty else pd.DataFrame()

    test_summary_df = pd.DataFrame(
        {"Metric": test_mean.index, "Mean": test_mean.values, "Std": test_std.values}
    )

    if not train_summary_df.empty:
        # wandb.log({"Train Metrics Summary table": wandb.Table(dataframe=train_summary_df)})
        train_summary_df.to_csv(
            run_dir / "train_metrics_summary.csv", index=False
        )
    # wandb.log({"Test Metrics Summary table": wandb.Table(dataframe=test_summary_df)})
    test_summary_df.to_csv(run_dir / "test_metrics_summary.csv", index=False)

    logger.success("Done!")
    wandb.finish()

if __name__ == "__main__":
    fire.Fire(main)

    # main(
    #     datasource="sun et al",
    #     algorithm="MAML",
    #     abundance_file="mpa4_species_profile_preprocessed.csv",
    #     metadata_file="sample_group_species_preprocessed.csv",
    #     test_study="JieZ_2017",
    #     balanced_or_unbalanced="unbalanced",
    #     n_gradient_steps=5,
    #     n_parallel_tasks=5,
    #     train_k_shot=10,
    #     use_wandb=True,
    # )
