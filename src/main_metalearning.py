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

from src.data.dataloader import (
    get_cross_validation_sun_et_al_data_splits,
)
from src.helper_function import (
    checkpoint_updater_callback_optuna,
    get_resume_dir_for_experiment,
    get_run_dir_for_experiment,
    load_checkpoint,
    save_checkpoint,
    set_seed,
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
    datasource: str,
    algorithm: str,
    abundance_file: str | Path,
    metadata_file: str | Path,
    test_study: str,
    balanced_or_unbalanced: str,
    new_copied_run: bool,
    # val_study: list,                      # random selection done
    # outer_lr_range: tuple[float, float],  # optimization
    # inner_lr_range: tuple[float, float],  # optimization
    # inner_rl_reduction_factor: int,       # optimization
    n_gradient_steps: int,  # TODO Could be a hyperparam
    n_parallel_tasks: int,  # TODO Could be a hyperparam
    train_k_shot: int,
    splitting_method: str = "study_wise",  # "normal" or "study_wise"
    eval_k_shot: int = None,
    # n_components_reduction_factor: int = 0,  # 0 or 1 for no PCA at all   # skip
    # use_cached_pca: bool = False,         # skip
    # do_normalization_before_scaling: bool = True, # optimization
    # scale_factor_before_training: int = 100,      # optimization
    loss_fn: str = "BCELog",
    use_wandb: bool = True,
    features_to_use: list[str] = None,
    # early_stop_patience: int = None,
    # early_stop_metric: str = "loss",
    resume: bool = True,
    track_best_f1: bool = False,
    positive_class_label: str | None = None,
    feature_reduction_alg: str = None,
    project: str = None,
    random_seed: int = RANDOM_SEED,
    extra_str_indicator: str = "",
    wait_for_warmup: bool = False,
    new_primary: bool = False,
):
    set_seed(random_seed)
    project = project or "metalearning"
    eval_k_shot = eval_k_shot or train_k_shot
    
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
        loss_fn = nn.BCEWithLogitsLoss
    else:
        raise ValueError("Loss function not recognized.")

    if datasource == "sun et al":
        data_root_dir = BASE_DATA_DIR / "sun_et_al_data"
        if "stunt" in abundance_file:
            # Read data and metadata
            train_data = pd.read_csv(
                f"{data_root_dir}/{abundance_file}",
                index_col=0,
                header=0,
            ).reset_index(drop=True)
            train_metadata = pd.read_csv(
                f"{data_root_dir}/{metadata_file}",
                index_col=0,
                header=0,
            ).reset_index(drop=True)
            # They should already be ordered correctly

            if features_to_use:
                train_metadata = train_metadata.loc[:, features_to_use]


            orig_data = pd.read_csv(
                f"{data_root_dir}/mpa4_species_profile_preprocessed.csv",
                index_col=0,
                header=0,
            )

            orig_metadata = pd.read_csv(
                f"{data_root_dir}/sample_group_species_preprocessed.csv",
                index_col=0,
                header=0,
            )
            orig_metadata = orig_metadata.loc[orig_data.index, :]
            grouped = orig_metadata.groupby("Project_1")
            test_data = pd.DataFrame()
            test_metadata = pd.DataFrame()

            # extra_train_data = pd.DataFrame()
            # extra_train_metadata = pd.DataFrame()
            for group, idx in grouped.groups.items():
                if group == test_study:
                    test_metadata = pd.concat([test_metadata, orig_metadata.loc[idx, :]])
                    test_data = pd.concat([test_data, orig_data.loc[idx, :]])
                # else:
                #     extra_train_data = pd.concat([extra_train_data, orig_data.loc[idx, :]])
                #     extra_train_metadata = pd.concat([extra_train_metadata, orig_metadata.loc[idx, :]])

            # print(str(train_metadata["Project_1"].unique()))
            # train_data = pd.concat([train_data, extra_train_data])
            # train_metadata = pd.concat([train_metadata, extra_train_metadata])
            # print(str(train_metadata["Project_1"].unique()))

            sun_et_al_abundance = pd.concat([train_data, test_data])
            sun_et_al_metadata = pd.concat([train_metadata, test_metadata])
        else:
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
            k_shot=eval_k_shot,
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
        str(eval_k_shot) + "_e_shot",
        datasource,
        balanced_or_unbalanced,
        # "e_k" + str(eval_k_shot),
    ]

    wandb_name = f"TS{test_study}_TK{train_k_shot}_{balanced_or_unbalanced}_{datasource}_{algorithm}_T{tax_level}_{extra_str_indicator}"
    p = "metalearning2" if eval_k_shot != train_k_shot or new_copied_run else project
    if eval_k_shot != train_k_shot or new_copied_run:
        # copy resume directory if it exists
        resume_dir = get_resume_dir_for_experiment(
            p, algorithm, test_study, wandb_name
        )
        new_resume_dir = get_resume_dir_for_experiment(
            project, algorithm, test_study, wandb_name + "Ek" + str(eval_k_shot)
        )
        if resume_dir != new_resume_dir:
            if resume_dir.exists():
                logger.info(f"Copying resume directory from {resume_dir} to {new_resume_dir}")
                new_resume_dir.mkdir(parents=True, exist_ok=True)
                # copy with shutil
                import shutil
                shutil.copytree(resume_dir, new_resume_dir, dirs_exist_ok=True)
                
        resume_dir = new_resume_dir
        logger.info(f"Using resume directory: {resume_dir}")
    else:
        new_resume_dir = None

    # Set up checkpoint path and load checkpoint if resuming
    resume_dir = new_resume_dir or get_resume_dir_for_experiment(
        project, algorithm, test_study, wandb_name
    )
    checkpoint_path = str(resume_dir / "checkpoint.yaml")
    checkpoint = load_checkpoint(checkpoint_path, resume)

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
        "eval_k_shot": eval_k_shot,
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
        # "early_stop_patience": early_stop_patience,
        # "early_stop_metric": early_stop_metric,
        "positive_class_label": positive_class_label,
        "random_seed": random_seed,
        "extra_str_indicator": extra_str_indicator,
        "feature_reduction_alg": feature_reduction_alg,
        "job_id": job_id,
        "array_job_id": array_job_id,
        "array_task_id": array_task_id,
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
                if "job_history" not in config:
                    config["job_history"] = []
                config["job_history"].append(current_job)

                # Update run name to indicate multiple jobs
                if len(config["job_history"]) > 1:
                    wandb_name += f"_multi{len(config['job_history'])}"

        wandb.init(
            project=project,
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
            project=project,
            group=algorithm,
            tags=wandb_base_tags,
        )

    logger.success("wandb init done")

    run_dir = get_run_dir_for_experiment(
        project, algorithm, test_study, wandb.run.id
    )

    # Store wandb run ID in checkpoint
    if not checkpoint["wandb_run_id"]:
        checkpoint["wandb_run_id"] = wandb.run.id

    # Register this job
    job_identifier = f"{array_job_id or job_id}"
    if array_task_id:
        job_identifier += f"_{array_task_id}"

    # If first job, set as primary
    if not checkpoint.get("primary_job_id") or new_primary:
        checkpoint["primary_job_id"] = job_identifier
    else:
        if tuning_num_samples > 0 and sum([t for t in checkpoint["trials_done_per_job"].values()]) >= tuning_num_samples:
            checkpoint["primary_job_id"] = job_identifier
            checkpoint["optimization_done"] = True

    if eval_k_shot != train_k_shot or new_copied_run:
        checkpoint["completed_folds"] = []
        checkpoint["fold_metrics"] = {}

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
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED, multivariate=True),
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
        trials_to_do = trials_to_do - sum([t for t in checkpoint["trials_done_per_job"].values()])
        logger.info(f"trials_to_do: {trials_to_do}")

        if trials_to_do > 0 and not checkpoint["optimization_done"]:
            if is_primary and not checkpoint.get("warmup_completed", False):
                if initial_trial:
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
                        # early_stop_pat=early_stop_patience,
                        # early_stop_metric=early_stop_metric,
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
                max_wait = 60 * 120  # 120 minutes max wait
                wait_interval = 60  # check every 60 seconds
                waited = 0
                with filelock.FileLock(checkpoint_path + ".lock", timeout=30):
                    checkpoint = load_checkpoint(checkpoint_path)
                while (
                    not checkpoint.get("warmup_completed", False) and waited < max_wait
                ):
                    if not wait_for_warmup:
                        raise TimeoutError("Warmup phase happening.")
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
                        # early_stop_pat=early_stop_patience,
                        # early_stop_metric=early_stop_metric,
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

            # try:
            #     fig = plot_param_importances(optuna_study)
            #     wandb.log({"param_imp_fig": wandb.Plotly(fig)})
            #     param_importance = optuna.importance.get_param_importances(optuna_study)
            #     param_importance_df = pd.DataFrame(
            #         {
            #             "Parameter": list(param_importance.keys()),
            #             "Importance": list(param_importance.values()),
            #         }
            #     )
            #     # wandb.log({"param_imp": wandb.Table(dataframe=param_importance_df)})
            #     param_importance_df.to_csv(
            #         run_dir / "param_importance.csv", index=False
            #     )
            # except Exception as e:
            #     traceback.print_exc()
            #     logger.error(f"Error in plotting param importance: {e}")

        checkpoint = load_checkpoint(checkpoint_path)
        checkpoint["optimization_done"] = True

        best_trial = optuna_study.best_trial
        best_trial_params = best_trial.params
        best_trial_params = {k: str(v) for k, v in best_trial_params.items()}
        checkpoint["best_trial_params"] = best_trial_params
        save_checkpoint(checkpoint_path, checkpoint)

    best_trial_config = search_space_sampler(best_trial)
    wandb.log(best_trial_config)
    for i, s in enumerate(best_trial_config["model__layer_sizes"]):
        wandb.log(
            {f"model__layer_sizes_{i}": s})
    logger.info(f"best_trail_config:\n{best_trial_config}")


    logger.info("Testin on the validation sets to log the best model performance")

    val_scores = []
    val_train_scores = []
    for j, (val_study_name, val_support_sets) in val_loop_data_selection.items():
        val_metadata = train_metadata[
            train_metadata["Project_1"] == val_study_name
        ]
        val_data = train_data.loc[val_metadata.index]
        val_split_train_data = train_data.drop(index=val_metadata.index)
        val_split_train_metadata = train_metadata.drop(index=val_metadata.index)

        model, train_loader, val_loader, eval_loader = get_metalearning_model_from_trial(
            val_split_train_data,
            val_data,
            val_split_train_metadata,
            val_metadata,
            train_k_shot,
            eval_k_shot,
            val_support_sets,
            algorithm,
            best_trial_config,
            config,
        )

        logger.info("Fitting model")
        val_train_results, val_results = model.fit(
            train_dataloader=train_loader,
            n_epochs=best_trial_config["max_epochs"],
            n_parallel_tasks=config["n_parallel_tasks"],
            val_dataloader=val_loader,
            eval_dataloader=eval_loader,
            early_stopping_patience=best_trial_config["early_stopping_patience"],
            early_stopping_fraction=best_trial_config["early_stopping_fraction"],
            # early_stopping_metric=early_stop_metric,
            log_metrics=False,  # Disable wandb logging during optimization
            score_name_prefix=f"fold{j}",
            track_best_f1=config["track_best_f1"],
        )

        val_train_results = (
            {
                k: v.tolist() if hasattr(v, "tolist") else v
                for k, v in val_train_results.items()
                if k != "predictions" and k != "targets"
            }
            if val_train_results
            else {}
        )
        val_results = {
            k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in val_results.items()
            if k != "predictions" and k != "targets"
        }

        val_train_scores.append(val_train_results)
        val_scores.append(val_results)

        val_train_results = {"val_train/" + k: v for k, v in val_train_results.items()}
        val_results = {"val/" + k: v for k, v in val_results.items()}
        wandb.log(
            {"Outer fold": dict(val_train_results, **val_results)},
        )
        

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
            best_model, train_loader, val_loader, test_loader = get_metalearning_model_from_trial(
                train_data,
                test_data,
                train_metadata,
                test_metadata,
                train_k_shot,
                eval_k_shot,
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
            n_epochs = 200

            train_res, test_res = best_model.fit(
                train_dataloader=train_loader,
                n_epochs=n_epochs,
                n_parallel_tasks=n_parallel_tasks,
                val_loader=val_loader,
                eval_dataloader=test_loader,
                val_or_test="test",
                log_metrics=True,
                log_gradients=False,
                score_name_prefix=f"outer_fold_{i}_fit",
                save_best_model_path=run_dir / f"best_model_outer_fold_{i}.pt",
                track_best_f1=track_best_f1,
                early_stopping_patience=best_trial_config["early_stopping_patience"],
                early_stopping_fraction=best_trial_config["early_stopping_fraction"],
                # early_stopping_metric=early_stop_metric
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

            train_res = {"train/" + k: v for k, v in train_res.items()}
            test_res = {"test/" + k: v for k, v in test_res.items()}
            wandb.log(
                {"Outer fold": dict(train_res, **test_res)},
            )

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
    # fire.Fire(main)

    # --datasource="sun et al" \
    # --algorithm="${ALGORITHM}" \
    # --balanced_or_unbalanced "$BALANCED_OR_UNBALANCED" \
    # --test_study="$STUDY" \
    # --n_gradient_steps 5 \
    # --n_parallel_tasks 5 \
    # --train_k_shot 10 \
    # --positive_class_label "Disease" \
    # --splitting_method="${SPLITTING_METHOD}" \
    # --abundance_file="stunt/stunt_mpa4_species_profile_preprocessed.csv" \
    # --metadata_file="stunt/stunt_sample_group_species_preprocessed.csv" \
    # --project="stunt_small" \
    # --extra_str_indicator="stunt_larger_es_fraction" \
    # --new_primary=True

    main(
        datasource="sun et al",
        algorithm="ProtoNet",
        abundance_file="mpa4_species_profile_preprocessed.csv",
        metadata_file="sample_group_species_preprocessed.csv",
        test_study="ZhongH_2019",
        balanced_or_unbalanced="unbalanced",
        n_gradient_steps=5,
        n_parallel_tasks=5,
        train_k_shot=10,
        use_wandb=False,
        resume=False,
        positive_class_label="Disease",
        splitting_method="study_wise",
        new_primary=True,
        track_best_f1=True,
        feature_reduction_alg="PCA",
    )


