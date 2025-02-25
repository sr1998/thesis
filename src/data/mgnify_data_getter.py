from pathlib import Path
from typing import List, Optional, Union
from os.path import join as os_path_join
import fire

from src.global_vars import BASE_DIR
from src.data.mgnify_helper import MGnifyData


def get_mgnify_data(study_download_label_start: str,
                    study_accessions: Optional[List[str]] = None,
                    biomes_desired: Optional[List[str]] = None,
                    download_metadata: bool = False,
                    base_data_dir: Union[str, Path] = os_path_join(BASE_DIR, "data"),
                    external_metadata: Optional[Union[str, Path]] = None
                    ) -> None:  
    """
    Function to download and if desired merge data and metadata and save it in the data directory
    
    """  
    # TODO - make join type parametrized
    print(study_accessions)
    # move to ml pipeline
    # train_test_split = config.get("train_test_split", [0.8, 0.2])
    # cross_val_k = config.get("cross_val_k", False)
    # val_size = config.get("val_size", 0.1)

    data_dir = os_path_join(base_data_dir, "mgnify_data")
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    mgnify = MGnifyData(cache_folder=data_dir)
    download_links = mgnify.get_download_links_for_studies(label_start_str=study_download_label_start,
                                                        #    desired_biomes=biomes_desired,
                                                           study_ids=study_accessions
                                                           )
    download_links = mgnify.filter_download_links_for_most_recent_link(download_links)
    mgnify.download_summary_for_studies(download_links)

    if download_metadata:
        mgnify.download_metadata_for_studies(list(download_links.keys()), external_metadata=external_metadata)


def main():
    fire.Fire(get_mgnify_data)
    # # None if all studies desired; if provided, only the summaries of those studies are downloaded
    # # if studies are given, they have to contain the desired summary given by study_download_label_start
    # study_accessions = "[..."]

    # # indicates what summary file to download for the studies
    # # possible values:
    #     # - Complete GO
    #     # - GO slim
    #     # - Phylum level (this results in downloading all possibilities, e.g. SSU, LSU, ...)
    #     # - Phylum level taxonomies SSU
    #     # - Taxonomic assignments SSU
    #     # - InterPro
    #     # - Taxonomic diversity metrics SSU
    #     # ... (see MGnify API)
    # study_download_label_start = "Taxonomic assignments SSU"

    # # The biome to get the studies for.
    # # None if all studies are desired
    # # The strings can also just be a part of the biome name.
    # biomes_desired = "..."

    # # Whether to download metadata for the studies
    # download_metadata =  True

    # # relative path of parent data directory from the root of the project
    # base_data_dir = "data"

    # get_mgnify_data(study_download_label_start=study_download_label_start, study_accessions=study_accessions, biomes_desired=biomes_desired, download_metadata=True, base_data_dir="data")

if __name__ == "__main__":
    main()