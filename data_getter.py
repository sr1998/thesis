from pathlib import Path
from typing import List, Optional, Union
from loguru import logger
from os.path import join as os_path_join

from global_vars import BASE_DIR
from mgnify_helper_functions import MGnifyData


def get_mgnify_data(study_download_label_start: str, study_accessions: Optional[List[str]] = None, biomes_desired: Optional[List[str]] = None,
                    download_metadata: bool = False, use_metadata: bool = False, base_data_dir: Union[str, Path] = os_path_join(BASE_DIR, "data")) -> None:    
    # move to ml pipeline
    # train_test_split = config.get("train_test_split", [0.8, 0.2])
    # cross_val_k = config.get("cross_val_k", False)
    # val_size = config.get("val_size", 0.1)

    data_dir = os_path_join(base_data_dir, "mgnify_data")
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    mgnify = MGnifyData(cache_folder=data_dir)
    download_links = mgnify.get_download_links_for_studies(label_start_str=study_download_label_start,
                                                           desired_biomes=biomes_desired,
                                                           study_ids=study_accessions
                                                           )

    download_links = mgnify.filter_download_links_for_most_recent_link(download_links)
    mgnify.download_summary_for_studies(download_links)

    if download_metadata:
        mgnify.download_metadata_for_studies(list(download_links.keys()))

    # combining metadata and summary data
    if use_metadata:
        for study, summary_link in download_links.items():
            mgnify.combine_metadata_with_corresponding_study_summary(study, summary_link)


def main():
    get_mgnify_data("Complete GO", ["MGYS00005628"], None, download_metadata=True, base_data_dir="data")

if __name__ == "__main__":
    main()