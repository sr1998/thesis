import pandas as pd
from loguru import logger

import src.preprocessing.functions as preprocessing_functions

from src.global_vars import BASE_DATA_DIR


data_root_dir = f"{BASE_DATA_DIR}/sun_et_al_data/"
columns_to_keep = ["Sample", "Group", "Project", "Project_1"]
studies_to_remove = ["LiS_2021a", "LiS_2021b"]
abundance_filtering_threshold = 0.0001
tax_level = "species"

# Get sample group data
sample_group = pd.read_table(f"{data_root_dir}/sample.group", sep="\t", header=0)
# Remove studies
logger.info(f"sample_group.shape before removal of studies: {sample_group.shape}")
sample_group = sample_group[~sample_group["Project_1"].isin(studies_to_remove)]
logger.info(f"sample_group.shape after removal of studies: {sample_group.shape}")

# Keep recommended columns
logger.info(f"sample_group.shape before column removal: {sample_group.shape}")
sample_group = sample_group[columns_to_keep]
logger.info(f"sample_group_useful.shape after column removal: {sample_group.shape}")
# Set index to Sample
sample_group = sample_group.set_index("Sample")
logger.info(f"sample_group_useful.shape after setting index: {sample_group.shape}")

# Get {tax_level} profile data
mpa4_profile = pd.read_table(
    f"{data_root_dir}/mpa4_{tax_level}.profile", sep="\t", header=0, index_col=0
)
# Remove {tax_level} with no reads
mpa4_profile = mpa4_profile.loc[
    :, mpa4_profile.sum(axis=0) >= 1
]

## Remove repeated samples
logger.info(f"sample_group_useful.shape before removal: {sample_group.shape}")
sample_group = sample_group[~sample_group.index.duplicated(keep="first")]
logger.info(f"sample_group_useful.shape after removal: {sample_group.shape}")

# remove samples not in sample_group
logger.info(
    f"mpa4_{tax_level}_profile.shape before filtering out samples without metadata: {mpa4_profile.shape}"
)
samples_to_keep = list(
    set(sample_group.index.tolist()) & set(mpa4_profile.columns.tolist())
)
mpa4_profile = mpa4_profile[samples_to_keep]
logger.info(
    f"mpa4_{tax_level}_profile.shape after filtering out samples without metadata: {mpa4_profile.shape}"
)
mpa4_profile = mpa4_profile.T
logger.info(
    f"mpa4_{tax_level}_profile.shape after transposing: {mpa4_profile.shape}"
)

# remove samples from sample_group that are not in mpa4_{tax_level}_profile
logger.info(
    f"sample_group_useful.shape before filtering out samples not in mpa4_{tax_level}_profile: {sample_group.shape}"
)
sample_group = sample_group.loc[samples_to_keep]
logger.info(
    f"sample_group_useful.shape after filtering out samples not in mpa4_{tax_level}_profile: {sample_group.shape}"
)

# Normalize the data
logger.info(
    f"mpa4_{tax_level}_profile summation before normalization: {mpa4_profile.sum(axis=1)}"
)
mpa4_profile = preprocessing_functions.total_sum_scaling(mpa4_profile)
logger.info(
    f"mpa4_{tax_level}_profile summation after normalization: {mpa4_profile.sum(axis=1)}"
)


# prevalence and abundance filtering
# low abundance filtering per study
# Should we skip filtering?
grouped_sample_group = sample_group.groupby("Project_1")

for project, samples in grouped_sample_group.groups.items():
    logger.info(f"Project: {project}")
    rows_to_update = mpa4_profile.loc[samples]
    feature_prevalence = (rows_to_update > abundance_filtering_threshold).sum(axis=0) / rows_to_update.shape[0]
    low_abundance_features = feature_prevalence < 0.1

    df_masked = rows_to_update.mask(
        low_abundance_features | (rows_to_update <= abundance_filtering_threshold), 0
    )
    mpa4_profile.update(df_masked)


# save it all
mpa4_profile.to_csv(
    f"{data_root_dir}/mpa4_{tax_level}_profile_after_abundane_filtering.csv"
)

# normalize and transform
logger.info(
    f"mpa4_{tax_level}_profile summation before normalization: {mpa4_profile.sum(axis=1)}"
)
mpa4_profile = preprocessing_functions.total_sum_scaling(mpa4_profile)
logger.info(
    f"mpa4_{tax_level}_profile summation after normalization: {mpa4_profile.sum(axis=1)}"
)

# Centered arcsine transform
logger.info(
    f"mpa4_{tax_level}_profile summation before centered arcsine transform: {mpa4_profile.sum(axis=1)}"
)
mpa4_profile = preprocessing_functions.centered_arcsine_transform(
    mpa4_profile
)
logger.info(
    f"mpa4_{tax_level}_profile summation after centered arcsine transform: {mpa4_profile.sum(axis=1)}"
)

# Save the data
mpa4_profile.to_csv(f"{data_root_dir}/mpa4_{tax_level}_profile_preprocessed.csv")
sample_group.to_csv(f"{data_root_dir}/sample_group_{tax_level}_preprocessed.csv")
