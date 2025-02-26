from pathlib import Path

import pandas as pd
from loguru import logger

import src.data.mgnify_helper as mhf
from src.global_vars import BASE_DATA_DIR
from src.helper_function import df_str_for_loguru


def load_mgnify_data(
    base_data_dir: str | Path,
    study_accessions: list[str],  # | set[str] | None?
    summary_type: str,
    pipeline: str | None,
    metdata_cols_to_use_as_features: list[str] | None | str,
    label_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load summary data and metadata for a study and return a combined dataframe + a label dataframe.

    Args:
        base_data_dir: Base directory for data.
        study_accessions: List of study accessions.
        summary_type: Type of summary data. Possible values:
            - GO_abundances
            - GO-slim_abundances
            - phylum_taxonomy_abundances_SSU
            - taxonomy_abundances_SSU
            - IPR_abundances
            - ... (see MGnify API for more)
        pipeline: Pipeline version. Possible values:
            - v3.0
            - v4.0
            - v4.1
            - v5.0
        metdata_cols_to_use_as_features: Metadata columns to use as features
        label_col: Label column in metadata

    Returns:
        Combined DataFrame of summary and metadata (except labels) and labels DataFrame.
    """
    data_dir = Path(base_data_dir) / "mgnify_data"
    mgnify = mhf.MGnifyData(cache_folder=data_dir)

    # FIXME if necessary: for now only one study is supported
    study_id = study_accessions[0]

    # get secondary_id for study
    id_to_secondary_id = mgnify.get_secondary_accessions(study_accessions)
    secondary_id = id_to_secondary_id[study_id]

    summary_file = (
        Path(data_dir) / f"{study_id}__{secondary_id}_{summary_type}_{pipeline}.tsv"
    )

    if "GO" in summary_type:
        logger.debug("Loading GO data")
        data = pd.read_table(summary_file, sep="\t", header=0, index_col=[0, 1, 2])
        data = data.droplevel(["description", "category"])
        logger.debug("Summary data: " + df_str_for_loguru(data))
        logger.debug(f"Summary data shape: {data.shape}")
    else:
        logger.debug("Loading taxonomy data")
        data = pd.read_table(summary_file, sep="\t", header=0)

        logger.debug("Summary data: " + df_str_for_loguru(data))
        logger.debug(f"Summary data shape: {data.shape}")
        data = data[
            data["#SampleID"].apply(lambda x: x.split(";")[-1].startswith("g__"))
        ]
        data = data.set_index("#SampleID")

        logger.debug(f"Summary data @ genus level: {df_str_for_loguru(data)}")
        logger.debug(f"Summary data shape - genus level: {data.shape}")

    data = data.T

    metadata = mgnify.get_metadata_for_study(study_id)

    # what metadata cols to use
    if metdata_cols_to_use_as_features is None:
        metdata_cols_to_use_as_features = []
    elif (
        metdata_cols_to_use_as_features == "all"
        or metdata_cols_to_use_as_features == ["all"]
        or metdata_cols_to_use_as_features == "All"
        or metdata_cols_to_use_as_features == ["All"]
    ):
        metdata_cols_to_use_as_features = list(metadata.columns)

    logger.debug(f"Metadata: {df_str_for_loguru(metadata)}")
    metadata_with_only_sample_id = metadata.loc[["sample_id"]]
    metadata = metadata.loc[metdata_cols_to_use_as_features + [label_col]]
    logger.debug(
        f"Metadata with cols to be used as features: {df_str_for_loguru(metadata)}"
    )
    logger.debug(f"Metadata shape: {metadata.shape}")

    # getting run_id to sample_id to map sample_id to run_id in metadata index
    run_id_to_sample_id = mgnify.get_run_id_to_sample_id_dict(study_id)

    # mapping them
    column_mapper = {}
    for key, value in run_id_to_sample_id.items():
        if value not in column_mapper:
            column_mapper[value] = []
        column_mapper[value].append(key)

    new_columns = {
        run_id: metadata_with_only_sample_id[sample_id]
        for sample_id, run_ids in column_mapper.items()
        for run_id in run_ids
        if sample_id in metadata_with_only_sample_id
    }
    metadata_with_run_id_as_index = metadata_with_only_sample_id.drop(
        columns=list(column_mapper.keys()), errors="ignore"
    )
    metadata_with_run_id_as_index = metadata_with_run_id_as_index.join(
        pd.DataFrame(new_columns)
    )
    metadata_with_run_id_as_index = metadata_with_run_id_as_index.dropna(
        axis=1, how="all"
    )
    metadata_with_run_id_as_index = metadata_with_run_id_as_index.T
    logger.debug(
        f"metadata_with_run_id_as_index: {df_str_for_loguru(metadata_with_run_id_as_index)}"
    )

    if (len(set(data.index) - set(metadata_with_run_id_as_index.index))) == 0:
        logger.info("All run_ids in summary data are in metadata")
    else:
        logger.info("Some run_ids in summary data are not in metadata")

    # group features of data by sample_id and sum them up
    # sample_id from metadata
    combined_data = data.join(metadata_with_run_id_as_index[["sample_id"]], how="inner")
    combined_data = combined_data.groupby(
        "sample_id"
    ).sum()  # FIXME if wanted: we can do other things than summation or skip this
    # add metadata labels to combined data
    combined_data = combined_data.join(metadata.T, how="inner")

    # drop rows without label
    combined_data = combined_data.dropna(subset=[label_col])

    logger.debug(f"Combined data: {df_str_for_loguru(combined_data)}")
    logger.debug(f"Combined data shape: {combined_data.shape}")

    labels = combined_data[label_col]
    combined_data = combined_data.drop(columns=[label_col])

    return combined_data, labels


def get_mgnify_data(
    study_accessions: str | list[str] | None,
    summary_type: str,
    pipeline_version: str,
    label_col: str,
    metdata_cols_to_use_as_features: list[str] = [],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get the data and labels for the MGnify dataset for learning.

    Args:
        study_accessions: List of study accessions.
        summary_type: Type of summary data. Possible values:
            - GO_abundances
            - GO-slim_abundances
            - phylum_taxonomy_abundances_SSU
            - taxonomy_abundances_SSU
            - IPR_abundances
            - ... (see MGnify API for more)
        pipeline_version: Pipeline version. Possible values:
            - v3.0
            - v4.0
            - v4.1
            - v5.0
        label_col: Label column in metadata
        metdata_cols_to_use_as_features: Metadata columns to use as features.

    Returns:
        The data and labels for the MGnify dataset.
    """
    if not isinstance(study_accessions, list) and study_accessions is not None:
        study_accessions = [study_accessions]

    data, labels = load_mgnify_data(
        BASE_DATA_DIR,
        study_accessions,
        summary_type,
        pipeline_version,
        metdata_cols_to_use_as_features,
        label_col,
    )

    return data, labels


def get_sun_et_al_study_data(
    study: str, abundance_file: str | Path, metadata_file: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get the data and labels for the Sun et al. of a specific study and taxonomic level for learning.

    Args:
        study: The study to get the data for.
        tax_level: The taxonomic level to get the data for.

    Returns:
        The data and labels for the Sun et al. dataset.
    """
    data = pd.read_csv(
        BASE_DATA_DIR / "sun_et_al_data" / abundance_file,
        index_col=0,
        header=0,
    )
    metadata = pd.read_csv(
        BASE_DATA_DIR / "sun_et_al_data" / metadata_file,
        header=0,
    )

    # Filter metadata to only include the study of interest
    metadata = metadata[metadata["Project_1"] == study]
    metadata = metadata.set_index("Sample")
    labels = metadata["Group"]

    # Filter data to only include samples that are in the metadata
    data = data.loc[labels.index]

    return data, labels


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