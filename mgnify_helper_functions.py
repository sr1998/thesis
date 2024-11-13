from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from typing import Union
import pandas as pd
from jsonapi_client import Session
                                

class MGnifyData:
    def __init__(self, base_folder: Union[str, Path]):
        self.base_folder = Path(base_folder)
        self.base_api = "https://www.ebi.ac.uk/metagenomics/api/latest/"
        self._session = Session(self.base_api)



    def load_checkpoint(self, file_path: Union[str, Path]) -> dict:
        """
        Load progress from checkpoint if it exists and is still valid (within the last 24 hours).
        """
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                checkpoint = json.load(file)
                # Check if the cache is within the last 24 hours
                last_saved_time = datetime.fromisoformat(checkpoint.get("timestamp"))
                if datetime.now() - last_saved_time < timedelta(hours=24):
                    return {k: v for k, v in checkpoint.items() if k != "timestamp"}
        return {}  # Return empty data

    def save_checkpoint(self, data: dict, file_path: Union[str, Path]):
        """
        Save progress to checkpoint file, with the current timestamp.
        """
        data["timestamp"] = datetime.now().isoformat()  # Store exact timestamp
        with open(file_path, "w") as file:
            json.dump(data, file)


    def _check_download_for_id_start(self, session, study_id, label_start_str):
        """
        Helper function to find the pipelines that have a specific download with id_start_str.
        """
        downloads_path = f"studies/{study_id}/downloads"
        downloads = session.get(downloads_path).resources
        pipelines_giving_desired_downloads = []
        for download in downloads:
            if download.description.label.startswith(label_start_str):
                pipelines_giving_desired_downloads.append(download.pipeline.id)
        return {study_id: pipelines_giving_desired_downloads}

    def studies_with_desired_summaries(self, label_start_str: str = "Complete GO", max_workers: int = 10):
        """
        Returns all studies (MGnify IDs) with a specific analysis summary available

        examples of label_start_str:
        - Complete GO
        - Phylum level
        - phylum_taxonomy_abundances_SSU
        - Phylum level taxonomies SSU
        - Taxonomic assignments SSU
        - InterPro

        for an example, look at "label"s in api for downloads of a study: e.g. https://www.ebi.ac.uk/metagenomics/api/v1/studies/MGYS00001980/downloads?format=api
        """
        # num_studies = 10

        with Session(self.base_api) as mgnify:
            result = {}
            study_ids = []
            num_studies = 5
            

            # Retrieve study IDs up to the specified limit
            for i, resource in enumerate(mgnify.iterate("studies")):
                if i >= num_studies:
                    break
                study_ids.append(resource.id)

            # Use ThreadPoolExecutor to fetch downloads concurrently
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._check_download_for_id_start, mgnify, study_id, label_start_str): study_id for study_id in study_ids}
                
                for future in as_completed(futures):
                    pipelines_giving_desired_downloads = future.result()
                    result.update(pipelines_giving_desired_downloads)

        return result


    # def get_study_sample_sizes():
    #     """
    #     Get the number of samples in each study
    #     """
    #     with Session(BASE_API) as mgnify:
    #         study_sizes = {}
            

    #         # Retrieve study IDs up to the specified limit
    #         for i, resource in enumerate(mgnify.iterate("studies")):
    #             study_id = resource.id
    #             sample_size = resource.attributes["samples-count"]
    #             study_sizes[study_id] = sample_size

    #     return study_sizes
    
    def get_study_sample_sizes(self, page_size=1000):
        """
        Get the number of samples in each study, saving progress at the end of each page.
        """
        checkpoint_path = self.base_folder / "study_sizes.json"
        with Session(self.base_api) as mgnify:
            # Load existing progress if available and valid for today
            checkpoint = self.load_checkpoint(checkpoint_path)
            last_page = checkpoint.get("last_page", 1)
            study_sizes = checkpoint.get("study_sizes", {})
            
            page = last_page  # Start from last saved page
            while True:
                # Retrieve studies ordered by `last_update`, limited to specified fields
                studies_endpoint = (
                    f"studies?ordering=last_update&page={page}&page_size={page_size}&"
                    "fields[studies]=id,samples_count"
                )
                try:
                    response = mgnify.get(studies_endpoint)
                    studies = response.resources
                except Exception as e:
                    if "404" in str(e):
                        print("No more pages available.")
                        break
                    else:
                        raise e  # Re-raise other exceptions
                
                # Exit if there are no more studies to process
                if not studies:
                    break

                # Process each study
                for resource in studies:
                    study_id = resource.id
                    
                    # Get and store the sample size for the study
                    sample_size = resource.attributes["samples-count"]
                    study_sizes[study_id] = sample_size

                # Save progress after each page, including the current date
                self.save_checkpoint({"last_page": page, "study_sizes": study_sizes}, checkpoint_path)

                # Move to the next page
                page += 1

        return study_sizes
