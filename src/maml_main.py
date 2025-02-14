import sys

sys.path.append("../..")

import pandas as pd
from loguru import logger

import src.preprocessing.functions as preprocessing_functions
from src.global_vars import BASE_DATA_DIR

data_root_dir = f"{BASE_DATA_DIR}/sun_et_al_data/"
columns_to_keep = ["Sample", "Group", "Project", "Project_1"]
studies_to_remove = ["LiS_2021a", "LiS_2021b"]

import os
import sys

sys.path.append("../../")

import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.decomposition import PCA

import src.data.sun_et_al as hf
import src.models.maml as maml
import src.models.reptile as rp
import wandb

from umap import UMAP


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


def change_sun_et_al_metadata_for_metalearning(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata[["Group", "Project_1"]]
    metadata = metadata.rename(columns={"Group": "label", "Project_1": "project"})
    return metadata


def main(
    sun_et_al_abundance: pd.DataFrame,
    sun_et_al_metadata: pd.DataFrame,
    config_script: str,
    test_study: list,
    val_study: list,
    outer_lr_range=(0.5, 0.01),
    inner_lr_range=(0.5, 0.01),
    loss_fn=nn.BCELoss(),
    n_gradient_steps=5,
    n_parallel_tasks=5,
    n_epochs=10,
    k_shot=30,
):
    # config_module = import_module(config_script)
    # setup = config_module.get_setup()
    # (
    #     misc_config,
    #     outer_cv_config,
    #     inner_cv_config,
    #     standard_pipeline,
    #     label_preprocessor,
    #     scoring,
    #     best_fit_scorer,
    #     tuning_mode,
    #     search_space_sampler,
    #     tuning_num_samples,
    # ) = setup.values()
    setup = ""
    wandb_params = {}

    # use_wandb = misc_config["wandb"]
    use_wandb = False
    # wandb_params = misc_config["wandb_params"]

    # Set up file logging
    # logger_path = get_run_dir_for_experiment(misc_config) / "log.log"
    # logger.add(logger_path, colorize=True, level="DEBUG")
    # logger.info("Setting up everything")

    job_id = os.getenv("SLURM_JOB_ID")
    wandb_base_tags = [
        "test_s" + str(test_study),
        "val_s" + str(val_study),
        "m_" + "Reptile",
        "j_" + job_id if job_id else "j_local",
    ]

    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            notes=str(setup),
            config=setup,
            **wandb_params,
            tags=wandb_base_tags,
        )
    else:
        wandb.init(
            mode="disabled",
            notes=str(setup),
            config=setup,
            **wandb_params,
            tags=wandb_base_tags,
        )

    logger.success("wandb init done")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sun_et_al_metadata = sun_et_al_metadata.sort_index()
    sun_et_al_abundance = sun_et_al_abundance.sort_index()

    train_data, test_data, val_data, train_metadata, test_metadata, val_metadata = (
        split_sun_et_al_data(
            sun_et_al_abundance, sun_et_al_metadata, test_study, val_study
        )
    )

    # U-map dimensionality reduction
    # umapper = UMAP(n_components=int(sun_et_al_abundance.shape[1]//1.5), n_neighbors=int(sun_et_al_abundance.shape[0]//1000), min_dist=0.1, metric="euclidean")
    # print("fitting and transforming")
    # train_data = pd.DataFrame(umapper.fit_transform(train_data), index=train_data.index, columns=train_data.columns)
    # train_data.to_csv("train_data.csv")
    # print('transforming')
    # test_data = pd.DataFrame(umapper.transform(test_data), index=test_data.index, columns=test_data.columns)
    # test_data.to_csv("test_data.csv")
    # print('transforming')
    # val_data = pd.DataFrame(umapper.transform(val_data), index=val_data.index, columns=val_data.columns)
    # val_data.to_csv("val_data.csv")

    # PCA
    pca = PCA(n_components=int(sun_et_al_abundance.shape[1]//20))
    print("fitting and transforming")
    train_data = pd.DataFrame(pca.fit_transform(train_data), index=train_data.index)
    print("transforming")
    test_data = pd.DataFrame(pca.transform(test_data), index=test_data.index)
    print("transforming")
    val_data = pd.DataFrame(pca.transform(val_data), index=val_data.index)
    # train_data.to_csv("train_data_PCA.csv")
    # test_data.to_csv("test_data_PCA.csv")
    # val_data.to_csv("val_data_PCA.csv")
    
    # train_data = pd.read_csv("train_data_PCA.csv", index_col=0)
    # test_data = pd.read_csv("test_data_PCA.csv", index_col=0)
    # val_data = pd.read_csv("val_data_PCA.csv", index_col=0)

    # normalize the data for deep learning
    from sklearn.preprocessing import Normalizer
    train_data = pd.DataFrame(Normalizer().fit_transform(train_data * 1000), index=train_data.index)
    test_data = pd.DataFrame(Normalizer().fit_transform(test_data * 1000), index=test_data.index)
    val_data = pd.DataFrame(Normalizer().fit_transform(val_data * 1000), index=val_data.index)

    train_metadata = change_sun_et_al_metadata_for_metalearning(train_metadata)
    test_metadata = change_sun_et_al_metadata_for_metalearning(test_metadata)
    val_metadata = change_sun_et_al_metadata_for_metalearning(val_metadata)

    train = hf.MicrobiomeDataset(train_data, train_metadata)
    test = hf.MicrobiomeDataset(test_data, test_metadata)
    val = hf.MicrobiomeDataset(val_data, val_metadata)

    sampler = hf.BinaryFewShotBatchSampler(
        train, k_shot, include_query=True, shuffle=True
    )
    train_loader = DataLoader(train, batch_sampler=sampler)

    sampler = hf.BinaryFewShotBatchSampler(
        test, k_shot, include_query=True, shuffle=False, shuffle_once = False, training=False
    )
    test_loader = DataLoader(test, batch_sampler=sampler)

    sampler = hf.BinaryFewShotBatchSampler(
        val, k_shot, include_query=True, shuffle=False, shuffle_once = False, training=False
    )
    val_loader = DataLoader(val, batch_sampler=sampler)

    n_features = train_data.shape[1]
    assert (
        n_features == test_data.shape[1] == val_data.shape[1]
    ), "Number of features of train, test and val must be the same."

    # Simple model to test
    model = rp.Model(n_features).to(device)

    meta_optimizer = SGD(model.parameters(), lr=max(outer_lr_range))

    # Instantiate the Reptile meta-learner.
    # reptile = rp.Reptile(
    #     model=model,
    #     train_n_gradient_steps=n_gradient_steps,
    #     eval_n_gradient_steps=n_gradient_steps,
    #     device=device,
    #     # loss_function=loss_fn,
    #     meta_optimizer=meta_optimizer,
    #     inner_lr=inner_lr,
    #     outer_lr=outer_lr,
    #     k_shot=k_shot,
    # )

    # reptile.fit(
    #     train_dataloader=train_loader,
    #     n_epochs=n_epochs,
    #     n_parallel_tasks=n_parallel_tasks,
    #     evaluate_train=True,
    #     val_dataloader=val_loader,
    # )

    # Do MAML
    MAML = maml.MAML(
        model=model,
        train_n_gradient_steps=n_gradient_steps,
        eval_n_gradient_steps=n_gradient_steps,
        device=device,
        meta_optimizer=meta_optimizer,
        inner_lr_range=inner_lr_range,
        outer_lr_range=outer_lr_range,
        k_shot=k_shot,
    )

    MAML.fit(train_dataloader=train_loader, n_epochs=n_epochs, n_parallel_tasks=n_parallel_tasks, evaluate_train=True, val_dataloader=val_loader)

if __name__ == "__main__":
    sun_et_al_abundance = pd.read_csv(
    f"{data_root_dir}/mpa4_species_profile_preprocessed.csv",
    index_col=0,
    header=0,
    )

    sun_et_al_metadata = pd.read_csv(
        f"{data_root_dir}/sample_group_preprocessed.csv",
        index_col=0,
        header=0,
    )

    main(
    sun_et_al_abundance,
    sun_et_al_metadata,
    None,
    "HanL_2021",
    "JieZ_2017",
    outer_lr_range=(1, 0.00001),
    inner_lr_range=(0.5, 0.00001),
    k_shot=10,
    n_gradient_steps=5,
    n_epochs=1000,
)