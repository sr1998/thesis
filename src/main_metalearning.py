import os
import sys
from importlib import import_module
from pathlib import Path
import traceback

import optuna
from optuna.visualization import plot_param_importances

from src.data.dataloader import (
    get_cross_validation_sun_et_al_data_splits,
)
from src.helper_function import (
    get_cluster_save_directory,
    get_run_dir_for_experiment,
    optuna_wandb_callback,
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
from src.global_vars import BASE_DATA_DIR


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

    wandb_name = f"TS{test_study}_TK{train_k_shot}_{balanced_or_unbalanced}_{datasource}_{algorithm}_T{tax_level}_J{job_id}"
    run_dir = get_run_dir_for_experiment({"wandb_params": {"name": wandb_name}})
    model_save_dir = get_cluster_save_directory({"wandb_params": {"name": wandb_name}}) or run_dir
    logger.info(f"Model save dir: {model_save_dir}")

    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project="metalearning",
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
            project="metalearning",
            group=algorithm,
            tags=wandb_base_tags,
        )

    logger.success("wandb init done")

    train_scores = []
    test_scores = []
    # split_config = []

    if tuning_num_samples > 0:
        optuna_study = optuna.create_study(
            direction=tuning_mode,
            study_name=f"hyper-param_optimization_for_{wandb.run.name}",
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
            n_trials=tuning_num_samples,
            callbacks=[optuna_wandb_callback],
        )
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
            wandb.log({"param_imp": wandb.Table(dataframe=param_importance_df)})
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error in plotting param importance: {e}")

    for i, test_support_set in test_loop_data_selection.items():
        if tuning_num_samples > 0:
            best_trial = optuna_study.best_trial
            # save best trial parameters + split for this loop
            best_trial_params = best_trial.params
            best_trial_params = {k: str(v) for k, v in best_trial_params.items()}
        else:
            best_trial = None
        # split_config.append(
        #     {
        #         "outer_cv_split": i,
        #         **best_trial_params,
        #     }
        # )

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

        train_res, test_res = best_model.fit(
            train_dataloader=train_loader,
            n_epochs=int(
                best_trial.user_attrs["actual_epochs"]
            ), # 10% more epochs
            n_parallel_tasks=n_parallel_tasks,
            eval_dataloader=test_loader,
            val_or_test="test",
            log_metrics=True,
            score_name_prefix=f"outer_fold_{i}_fit",
            save_best_model_path=model_save_dir / f"best_model_outer_fold_{i}.pt",
        )

        train_res = {
            k: v for k, v in train_res.items() if k != "predictions" and k != "targets"
        } if train_res else {}
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
        wandb.log({"Train Metrics Summary table": wandb.Table(dataframe=train_summary_df)})
    wandb.log({"Test Metrics Summary table": wandb.Table(dataframe=test_summary_df)})

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
    #     algorithm="Reptile",
    #     abundance_file="mpa4_species_profile_preprocessed.csv",
    #     metadata_file="sample_group_species_preprocessed.csv",
    #     test_study="JieZ_2017",
    #     balanced_or_unbalanced="balanced",
    #     n_gradient_steps=5,
    #     n_parallel_tasks=5,
    #     train_k_shot=10,
    #     use_wandb=True,
    # )
