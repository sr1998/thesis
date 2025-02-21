import os
import sys
from importlib import import_module

sys.path.append(".")

import fire
import pandas as pd
import torch
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

import src.data.sun_et_al as hf
import src.models.maml_with_l2l as maml_with_l2l
import src.models.reptile as rp
import wandb
from src.global_vars import BASE_DATA_DIR


def get_studies_desired_from_sun_et_al(
    data: pd.DataFrame, metadata: pd.DataFrame, study: list
):
    """Get the studies desired from the Sun et al data.

    Args:
        data: The data to filter. Index should be samples.
        metadata: The metadata to filter. Index should be samples.
        studies: The studies to keep. Should be in the Project_1 column of the metadata.

    Returns:
        tuple: data, metadata dataframes with only the studies of interest
    """
    # Filter metadata to only include the studies of interest
    metadata = metadata[metadata["Project_1"].isin(study)]

    # Filter data to only include samples that are in the metadata
    data = data.loc[metadata.index]

    return data, metadata


def split_sun_et_al_data(data: pd.DataFrame, metadata: pd.DataFrame, test, val):
    """Split the data into train, test and validation sets.

    Args:
        data: The data to split. Index should be samples.
        metadata: The metadata to split. Index should be samples.
        test: The studies to use for testing.
        val: The studies to use for validation.

    Returns:
        tuple: train, test, val dataframes
    """
    if not isinstance(test, list):
        test = [test]
    if not isinstance(val, list):
        val = [val]

    test_data, test_metadata = get_studies_desired_from_sun_et_al(data, metadata, test)
    val_data, val_metadata = get_studies_desired_from_sun_et_al(data, metadata, val)

    train_data = data.drop(test_data.index)
    train_data = train_data.drop(val_data.index)

    train_metadata = metadata.drop(index=test_metadata.index)
    train_metadata = train_metadata.drop(index=val_metadata.index)

    return train_data, test_data, val_data, train_metadata, test_metadata, val_metadata


def column_rename_for_sun_et_al_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata[["Group", "Project_1"]]
    metadata = metadata.rename(columns={"Group": "label", "Project_1": "project"})
    return metadata


def pca_reduction(
    train_data,
    test_data,
    val_data,
    n_components_reduction_factor: int,
    use_cache: bool = False,
):
    if not use_cache:
        pca = PCA(
            n_components=int(train_data.shape[1] // n_components_reduction_factor)
        )
        print("fitting and transforming")
        train_data = pd.DataFrame(pca.fit_transform(train_data), index=train_data.index)
        print("transforming")
        test_data = pd.DataFrame(pca.transform(test_data), index=test_data.index)
        print("transforming")
        val_data = pd.DataFrame(pca.transform(val_data), index=val_data.index)
        train_data.to_csv("train_data_PCA.csv")
        test_data.to_csv("test_data_PCA.csv")
        val_data.to_csv("val_data_PCA.csv")
    else:
        train_data = pd.read_csv("train_data_PCA.csv", index_col=0)
        test_data = pd.read_csv("test_data_PCA.csv", index_col=0)
        val_data = pd.read_csv("val_data_PCA.csv", index_col=0)

    return train_data, test_data, val_data


def main(
    model_script: str,
    model_name: str,
    abundance_file: pd.DataFrame,
    metadata_file: pd.DataFrame,
    test_study: list,
    val_study: list,
    outer_lr_range: tuple[float, float],
    inner_lr_range: tuple[float, float],
    inner_rl_reduction_factor: int,
    n_gradient_steps: int,
    n_parallel_tasks: int,
    n_epochs: int,
    train_k_shot: int,
    eval_k_shot: int = None,
    n_components_reduction_factor: int = 0,  # 0 or 1 for no PCA at all
    use_cached_pca: bool = False,
    do_normalization_before_scaling: bool = True,
    scale_factor_before_training: int = 100,
    loss_fn: str = "BCELog",
    use_wandb: bool = True,
    features_to_use: list[str] = None,
    algorithm: str = "MAML",
):
    if loss_fn == "BCELog":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        raise ValueError("Loss function not recognized.")

    data_root_dir = f"{BASE_DATA_DIR}/sun_et_al_data/"

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up file logging
    # logger_path = get_run_dir_for_experiment(misc_config) / "log.log"
    # logger.add(logger_path, colorize=True, level="DEBUG")
    # logger.info("Setting up everything")

    job_id = os.getenv("SLURM_JOB_ID")
    tax_level = abundance_file.split("_")[1]
    config = {
        "model_name": model_name,
        "algorithm": algorithm,
        "abundance_file": abundance_file,
        "metadata_file": metadata_file,
        "test_study": test_study,
        "val_study": val_study,
        "outer_lr_range": outer_lr_range,
        "inner_lr_range": inner_lr_range,
        "n_gradient_steps": n_gradient_steps,
        "n_parallel_tasks": n_parallel_tasks,
        "n_epochs": n_epochs,
        "train_k_shot": train_k_shot,
        "eval_k_shot": eval_k_shot,
        "n_components_reduction_factor": n_components_reduction_factor,
        "use_cache_pca": use_cached_pca,
        "do_normalization_before_scaling": do_normalization_before_scaling,
        "scale_factor_before_training": scale_factor_before_training,
        "loss_fn": loss_fn,
        "use_wandb": use_wandb,
        "device": device,
        "job_id": job_id,
        "features_to_use": features_to_use,
        "model_script": model_script,
    }
    wandb_base_tags = [
        "t_s" + str(test_study),
        "v_s" + str(val_study),
        "m_" + model_name,
        "a_" + algorithm,
        "j_" + job_id if job_id else "j_local",
        "tax_" + tax_level,
        "t_k" + str(train_k_shot),
        "e_k" + str(eval_k_shot),
    ]

    wand_name = f"{model_name}_{algorithm}_TS{test_study}_VS{val_study}_J{job_id}_T{tax_level}_TK{train_k_shot}_EK{eval_k_shot}"

    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project="meta-learning",
            name=wand_name,
            config=config,
            group=algorithm,
            tags=wandb_base_tags,
        )
    else:
        wandb.init(
            name=wand_name,
            mode="disabled",
            config=config,
            project="meta-learning",
            group=algorithm,
            tags=wandb_base_tags,
        )

    logger.success("wandb init done")

    if features_to_use:
        sun_et_al_abundance = sun_et_al_abundance.loc[:, features_to_use]

    sun_et_al_metadata = sun_et_al_metadata.sort_index()
    sun_et_al_abundance = sun_et_al_abundance.sort_index()

    train_data, test_data, val_data, train_metadata, test_metadata, val_metadata = (
        split_sun_et_al_data(
            sun_et_al_abundance, sun_et_al_metadata, test_study, val_study
        )
    )

    if n_components_reduction_factor != 0 and n_components_reduction_factor != 1:
        train_data, test_data, val_data = pca_reduction(
            train_data, test_data, val_data, use_cache=use_cached_pca
        )

    # tts
    # train_data = preprocessing_functions.total_sum_scaling(train_data)
    # test_data = preprocessing_functions.total_sum_scaling(test_data)
    # val_data = preprocessing_functions.total_sum_scaling(val_data)

    # centered log ratio transform
    # replace_zero_with = train_data[train_data > 0].min().min() / 100
    # train_data = preprocessing_functions.centered_log_ratio(
    #     train_data, replace_zero_with=replace_zero_with
    # )
    # test_data = preprocessing_functions.centered_log_ratio(
    #     test_data, replace_zero_with=replace_zero_with
    # )
    # val_data = preprocessing_functions.centered_log_ratio(
    #     val_data, replace_zero_with=replace_zero_with
    # )

    # normalize the data for deep learning
    if do_normalization_before_scaling:
        train_data = pd.DataFrame(
            Normalizer().fit_transform(train_data),
            index=train_data.index,
            columns=train_data.columns,
        )
        if test_study:
            test_data = pd.DataFrame(
                Normalizer().fit_transform(test_data),
                index=test_data.index,
                columns=test_data.columns,
            )
        val_data = pd.DataFrame(
            Normalizer().fit_transform(val_data),
            index=val_data.index,
            columns=val_data.columns,
        )

    train_data = train_data * scale_factor_before_training
    test_data = test_data * scale_factor_before_training
    val_data = val_data * scale_factor_before_training

    train_metadata = column_rename_for_sun_et_al_metadata(train_metadata)
    test_metadata = column_rename_for_sun_et_al_metadata(test_metadata)
    val_metadata = column_rename_for_sun_et_al_metadata(val_metadata)

    # Create Datasets for DataLoader
    train = hf.MicrobiomeDataset(train_data, train_metadata)
    test = hf.MicrobiomeDataset(test_data, test_metadata)
    val = hf.MicrobiomeDataset(val_data, val_metadata)

    # Create DataLoaders
    sampler = hf.BinaryFewShotBatchSampler(
        train, train_k_shot, include_query=True, shuffle=True
    )
    train_loader = DataLoader(train, batch_sampler=sampler)

    if eval_k_shot is None:
        eval_k_shot = train_k_shot

    sampler = hf.BinaryFewShotBatchSampler(
        test,
        train_k_shot,   # Has to be train_k_shot for now. Adjust MAML and Reptile algorithms (evaluation part) to be able to use eval_k_shot
        include_query=True,
        shuffle=False,
        shuffle_once=False,
        training=False,
    )
    test_loader = DataLoader(test, batch_sampler=sampler)

    sampler = hf.BinaryFewShotBatchSampler(
        val,
        train_k_shot,
        include_query=True,
        shuffle=False,
        shuffle_once=False,
        training=False,
    )
    val_loader = DataLoader(val, batch_sampler=sampler)

    # Get model
    model_module = import_module(model_script)
    n_features = train_data.shape[1]
    assert (
        n_features == test_data.shape[1] == val_data.shape[1]
    ), "Number of features of train, test and val must be the same."

    # Simple model to test
    model = model_module.get_model(model_name)(n_features).to(device)

    if algorithm == "MAML":
        MAML = maml_with_l2l.MAML(
            model=model,
            train_n_gradient_steps=n_gradient_steps,
            eval_n_gradient_steps=n_gradient_steps,
            device=device,
            inner_lr_range=inner_lr_range,
            inner_rl_reduction_factor=inner_rl_reduction_factor,
            outer_lr_range=outer_lr_range,
            k_shot=train_k_shot,
            loss_fn=loss_fn,
        )

        MAML.fit(
            train_dataloader=train_loader,
            n_epochs=n_epochs,
            n_parallel_tasks=n_parallel_tasks,
            evaluate_train=True,
            val_dataloader=val_loader,
        )
    elif algorithm == "Reptile":
            # Instantiate the Reptile meta-learner.
        reptile = rp.Reptile(
            model=model,
            train_n_gradient_steps=n_gradient_steps,
            eval_n_gradient_steps=n_gradient_steps,
            device=device,
            # loss_function=loss_fn,
            inner_lr=max(inner_lr_range),
            outer_lr=max(outer_lr_range),
            k_shot=train_k_shot,
        )

        reptile.fit(
            train_dataloader=train_loader,
            n_epochs=n_epochs,
            n_parallel_tasks=n_parallel_tasks,
            evaluate_train=True,
            val_dataloader=val_loader,
        )


if __name__ == "__main__":
    fire.Fire(main)

    # main(
    #     "src.models.models",
    #     "model3",
    #     "mpa4_species_profile_preprocessed.csv",
    #     "sample_group_species_preprocessed.csv",
    #     "",
    #     "JieZ_2017",
    #     outer_lr_range=(1, 0.01),
    #     inner_lr_range=(0.5, 0.001),
    #     inner_rl_reduction_factor=2,
    #     n_epochs=100,
    #     train_k_shot=10,
    #     n_gradient_steps=5,
    #     n_parallel_tasks=5,
    #     n_components_reduction_factor=0,
    #     use_cached_pca=False,
    #     do_normalization_before_scaling=True,
    #     scale_factor_before_training=100,
    #     loss_fn="BCELog",
    #     algorithm="MAML"
    # )
