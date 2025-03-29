import os
import sys
import time
import traceback
from functools import partial
from importlib import import_module
from pathlib import Path

import filelock
import numpy as np
import optuna
import optuna.storages
import optuna.terminator
from optuna.visualization import plot_param_importances

from src.data.dataloader import (
    get_cross_validation_sun_et_al_data_splits,
)
from src.helper_function import (
    checkpoint_updater_callback_optuna,
    get_resume_dir_for_experiment,
    get_run_dir_for_experiment,
    load_checkpoint,
    save_checkpoint,
)
from src.models.metalearning_helpers import (
    get_metalearning_model_from_trial,
    hyp_param_val_for_metalearning,
)

sys.path.append(".")

import fire
import pandas as pd
import torch
from loguru import logger
from torch import nn

import wandb
from src.global_vars import BASE_DATA_DIR, RANDOM_SEED


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
    splitting_method: str = "normal",  # "normal" or "study_wise"
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
    resume: bool = True,
    track_best_f1: bool = True,
    positive_class_label: str | None = None,
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
        tuning_num_samples_primary,
        tuning_num_samples_helper,
        search_space_sampler,
        initial_trial,
    ) = setup.values()

    if loss_fn == "BCELog":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Loss function not recognized.")

    if datasource == "sun et al":
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

        if features_to_use:
            sun_et_al_abundance = sun_et_al_abundance.loc[:, features_to_use]

        # Get the data splits: outer and inner cross val splits
        (
            test_loop_data_selection,
            val_loop_data_selection,
            train_data,
            train_metadata,
            test_data,
            test_metadata,
        ) = get_cross_validation_sun_et_al_data_splits(
            sun_et_al_abundance,
            sun_et_al_metadata,
            test_study=test_study,
            k_shot=train_k_shot,
            balanced_or_unbalanced=balanced_or_unbalanced,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
        )
    else:
        raise ValueError("Datasource not recognized.")

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

    wandb_base_tags = [
        str(test_study),
        algorithm,
        tax_level,
        str(train_k_shot) + "_shot",
        datasource,
        balanced_or_unbalanced,
        # "e_k" + str(eval_k_shot),
    ]

    wandb_name = f"TS{test_study}_TK{train_k_shot}_{balanced_or_unbalanced}_{datasource}_{algorithm}_T{tax_level}"
    # Set up checkpoint path and load checkpoint if resuming
    resume_dir = get_resume_dir_for_experiment(
        "metalearning", algorithm, test_study, wandb_name
    )
    checkpoint_path = str(resume_dir / "checkpoint.yaml")
    checkpoint = load_checkpoint(checkpoint_path) or {
        "completed_folds": [],
        "trials_done_per_job": {},
        "wandb_run_id": None,
        "fold_metrics": {},
        "warmup_completed": False,  # Flag for initial warmup phase
        "optimization_done": False,  # Flag for optimization completion
        "best_trial_params": None,
        "primary_job_id": None,
    }

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
        "resume": resume,
        "track_best_f1": track_best_f1,
        "splitting_method": splitting_method,
        "early_stop_patience": early_stop_patience,
        "early_stop_metric": early_stop_metric,
        "positive_class_label": positive_class_label,
        "job_id": job_id,
        "array_job_id": array_job_id,
        "array_task_id": array_task_id,
        "job_history": [f"{array_job_id or job_id}_{array_task_id or ''}"],
    }

    # Initialize wandb if enabled
    if use_wandb:
        # Update config with previous job IDs if resuming
        if resume and checkpoint["wandb_run_id"]:
            if job_id or array_job_id:
                current_job = f"{array_job_id or job_id}"
                if array_task_id:
                    current_job += f"_{array_task_id}"

                # Add current job ID to history
                config["job_history"].append(current_job)

                # Update run name to indicate multiple jobs
                if len(config["job_history"]) > 1:
                    wandb_name += f"_multi{len(config['job_history'])}"

        wandb.init(
            project="metalearning",
            name=wandb_name,
            config=config,
            notes=str(config),
            group=algorithm,
            tags=wandb_base_tags,
            id=checkpoint["wandb_run_id"] if resume else None,
            resume="allow" if resume and checkpoint["wandb_run_id"] else None,
        )
    else:
        wandb.init(
            name=wandb_name,
            mode="disabled",
            config=config,
            notes=str(config),
            project="metalearning",
            group=algorithm,
            tags=wandb_base_tags,
        )

    logger.success("wandb init done")

    run_dir = get_run_dir_for_experiment(
        "metalearning", algorithm, test_study, wandb.run.id
    )

    # Store wandb run ID in checkpoint
    if not checkpoint["wandb_run_id"]:
        checkpoint["wandb_run_id"] = wandb.run.id

    # Register this job
    job_identifier = f"{array_job_id or job_id}"
    if array_task_id:
        job_identifier += f"_{array_task_id}"

    # If first job, set as primary
    if not checkpoint.get("primary_job_id"):
        checkpoint["primary_job_id"] = job_identifier

    save_checkpoint(checkpoint_path, checkpoint)

    train_scores = []
    test_scores = []
    # split_config = []
    best_trial = None

    if tuning_num_samples > 0:
        # Create or load Optuna study with RDBStorage for parallel optimization
        storage_path = f"sqlite:///{resume_dir}/optuna_study.db"
        storage = optuna.storages.RDBStorage(
            url=storage_path,
            heartbeat_interval=60,
            grace_period=120,
            failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=1),
        )
        optuna_study = optuna.create_study(
            direction=tuning_mode,
            study_name=f"hyper-param_optimization_for_{checkpoint['wandb_run_id']}",
            storage=storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        )

        is_primary = job_identifier == checkpoint["primary_job_id"]
        logger.info(f"This job is {'primary' if is_primary else 'helper'}")

        # Calculate trials for this job
        n_warmup_trials = 5
        trials_to_do = (
            tuning_num_samples_primary if is_primary else tuning_num_samples_helper
        )
        trials_to_do = trials_to_do - checkpoint["trials_done_per_job"].get(
            job_identifier, 0
        )

        if trials_to_do > 0 and not checkpoint["optimization_done"]:
            # Primary job handles warmup phase
            if is_primary and not checkpoint.get("warmup_completed", False):
                optuna_study.enqueue_trial(initial_trial)
                logger.info(
                    f"Primary job running warmup phase: {n_warmup_trials} trials"
                )
                optuna_study.optimize(
                    lambda trial: hyp_param_val_for_metalearning(
                        algorithm,
                        val_loop_data_selection,
                        train_data,
                        train_metadata,
                        train_k_shot,
                        train_k_shot,
                        search_space_sampler,
                        trial,
                        config,
                        early_stop_pat=early_stop_patience,
                        early_stop_metric=early_stop_metric,
                    ),
                    n_trials=n_warmup_trials,
                    callbacks=[
                        partial(
                            checkpoint_updater_callback_optuna,
                            checkpoint_path=checkpoint_path,
                            job_id=job_identifier,
                        )
                    ],
                )
                with filelock.FileLock(checkpoint_path + ".lock", timeout=30):
                    checkpoint = load_checkpoint(checkpoint_path)
                    checkpoint["warmup_completed"] = True
                    save_checkpoint(checkpoint_path, checkpoint)
                logger.info("Warmup phase completed.")
                trials_to_do -= n_warmup_trials

            # Wait for warmup to complete if this is a helper job
            if not is_primary:
                max_wait = 60 * 30  # 30 minutes max wait
                wait_interval = 60  # check every 60 seconds
                waited = 0
                with filelock.FileLock(checkpoint_path + ".lock", timeout=30):
                    checkpoint = load_checkpoint(checkpoint_path)
                while (
                    not checkpoint.get("warmup_completed", False) and waited < max_wait
                ):
                    logger.info(
                        f"Helper job waiting for warmup to complete... ({waited}s)"
                    )
                    time.sleep(wait_interval)
                    waited += wait_interval
                    # Reload checkpoint
                    with filelock.FileLock(checkpoint_path + ".lock", timeout=30):
                        checkpoint = load_checkpoint(checkpoint_path)

                if not checkpoint.get("warmup_completed", False):
                    raise TimeoutError("Warmup phase not completed in time.")

            with filelock.FileLock(checkpoint_path + ".lock", timeout=30):
                checkpoint = load_checkpoint(checkpoint_path)

            # All jobs help with remaining trials
            if trials_to_do > 0:
                logger.info(f"Running {trials_to_do} trials for job {job_identifier}")
                optuna_study.optimize(
                    lambda trial: hyp_param_val_for_metalearning(
                        algorithm,
                        val_loop_data_selection,
                        train_data,
                        train_metadata,
                        train_k_shot,
                        train_k_shot,
                        search_space_sampler,
                        trial,
                        config,
                        early_stop_pat=early_stop_patience,
                        early_stop_metric=early_stop_metric,
                    ),
                    n_trials=trials_to_do,
                    callbacks=[
                        partial(
                            checkpoint_updater_callback_optuna,
                            checkpoint_path=checkpoint_path,
                            job_id=job_identifier,
                        )
                    ],
                )

            # if primary is finished, it should wait for helpers to finish
            if is_primary:
                logger.info(
                    f"Primary job {job_identifier} completed {trials_to_do} trials"
                )
                # Wait for helper jobs to finish
                while True:
                    with filelock.FileLock(checkpoint_path + ".lock", timeout=30):
                        checkpoint = load_checkpoint(checkpoint_path)
                        if (
                            sum([t for t in checkpoint["trials_done_per_job"].values()])
                            >= tuning_num_samples
                        ):
                            break
                    time.sleep(60)
            else:
                logger.info(
                    f"Helper job {job_identifier} completed {trials_to_do} trials"
                )
                return

            try:
                fig = plot_param_importances(optuna_study)
                wandb.log({"param_imp_fig": wandb.Plotly(fig)})
                param_importance = optuna.importance.get_param_importances(optuna_study)
                param_importance_df = pd.DataFrame(
                    {
                        "Parameter": list(param_importance.keys()),
                        "Importance": list(param_importance.values()),
                    }
                )
                # wandb.log({"param_imp": wandb.Table(dataframe=param_importance_df)})
                param_importance_df.to_csv(
                    run_dir / "param_importance.csv", index=False
                )
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error in plotting param importance: {e}")

        checkpoint = load_checkpoint(checkpoint_path)
        checkpoint["optimization_done"] = True

        best_trial = optuna_study.best_trial
        best_trial_params = best_trial.params
        best_trial_params = {k: str(v) for k, v in best_trial_params.items()}
        checkpoint["best_trial_params"] = best_trial_params
        save_checkpoint(checkpoint_path, checkpoint)

    best_trial_config = search_space_sampler(best_trial)

    for i, test_support_set in test_loop_data_selection.items():
        fold_id = str(i)

        if fold_id in checkpoint["completed_folds"]:
            logger.info(f"Skipping already completed fold {i}")

            # Load saved metrics
            if fold_id in checkpoint["fold_metrics"]:
                train_scores.append(checkpoint["fold_metrics"][fold_id]["train"])
                test_scores.append(checkpoint["fold_metrics"][fold_id]["test"])
            continue

        logger.info(f"Processing outer CV fold {i}")

        try:
            # Train the best model
            best_trial_config = search_space_sampler(best_trial)
            best_model, train_loader, test_loader = get_metalearning_model_from_trial(
                train_data,
                test_data,
                train_metadata,
                test_metadata,
                train_k_shot,
                train_k_shot,
                test_support_set,
                algorithm,
                best_trial_config,
                config,
            )

            # n_epochs = (
            #     int(best_trial.user_attrs["actual_epochs"])
            #     if best_trial and "actual_epochs" in best_trial.user_attrs
            #     else 100
            # )
            n_epochs = 1000

            train_res, test_res = best_model.fit(
                train_dataloader=train_loader,
                n_epochs=n_epochs,
                n_parallel_tasks=n_parallel_tasks,
                eval_dataloader=test_loader,
                val_or_test="test",
                log_metrics=True,
                log_gradients=True,
                score_name_prefix=f"outer_fold_{i}_fit",
                save_best_model_path=run_dir / f"best_model_outer_fold_{i}.pt",
                track_best_f1=track_best_f1,
            )

            train_res = (
                {
                    k: v.tolist() if hasattr(v, "tolist") else v
                    for k, v in train_res.items()
                    if k != "predictions" and k != "targets"
                }
                if train_res
                else {}
            )
            test_res = {
                k: v.tolist() if hasattr(v, "tolist") else v
                for k, v in test_res.items()
                if k != "predictions" and k != "targets"
            }

            train_scores.append(train_res)
            test_scores.append(test_res)

            # Update checkpoint
            checkpoint["completed_folds"].append(fold_id)
            if "fold_metrics" not in checkpoint:
                checkpoint["fold_metrics"] = {}
            checkpoint["fold_metrics"][fold_id] = {"train": train_res, "test": test_res}
            save_checkpoint(checkpoint_path, checkpoint)

        except Exception as e:
            logger.error(f"Error in outer CV fold {i}: {e}")
            traceback.print_exc()
            # Save checkpoint without marking this fold as complete
            save_checkpoint(checkpoint_path, checkpoint)
            raise e

    # log overall results to wandb
    if test_scores:
        train_scores = pd.DataFrame(train_scores)
        test_scores = pd.DataFrame(test_scores)
        train_mean = train_scores.mean() if not train_scores.empty else pd.Series()
        test_mean = test_scores.mean()
        train_std = train_scores.std() if not train_scores.empty else pd.Series()
        test_std = test_scores.std()

        # Log bar plots for train and test metrics
        train_summary_df = (
            pd.DataFrame(
                {
                    "Metric": train_mean.index,
                    "Mean": train_mean.values,
                    "Std": train_std.values,
                }
            )
            if not train_scores.empty
            else pd.DataFrame()
        )

        test_summary_df = pd.DataFrame(
            {
                "Metric": test_mean.index,
                "Mean": test_mean.values,
                "Std": test_std.values,
            }
        )

        if not train_summary_df.empty:
            # wandb.log({"Train Metrics Summary table": wandb.Table(dataframe=train_summary_df)})
            train_summary_df.to_csv(run_dir / "train_metrics_summary.csv", index=False)
        # wandb.log({"Test Metrics Summary table": wandb.Table(dataframe=test_summary_df)})
        test_summary_df.to_csv(run_dir / "test_metrics_summary.csv", index=False)

    # Save all outer CV splits and best trial parameters
    # results_df = pd.DataFrame(split_config)
    # results_path = run_dir / "outer_cv_splits_and_best_trial_params.csv"
    # results_df.to_csv(results_path, index=False)
    # wandb.log(
    #     {"outer_cv_splits_and_best_trial_params": wandb.Table(dataframe=results_df)}
    # )
    # logger.success(
    #     f"Saved all outer CV splits and best trial parameters to {results_path} and wandb."
    # )

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
    #     resume=False,
    #     positive_class_label="Disease",
    # )
