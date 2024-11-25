from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from loguru import logger
from matplotlib.axes import Axes
import pandas as pd
from requests import Session as requests_session
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from biosamples_helper_functions import get_metadata_of_samples
from global_vars import HTTP_ADAPTER_FOR_REQUESTS
from helper_function import hasher
                                

def plot_jaccard_similarities(dir_path: Union[str, Path], title: str, column_name: str, biome_dict: Optional[Dict] = None, file_ending: str = ".tsv", axis: Optional[Axes] = None):
    # get dataframe for each file in dir_path ending with file_ending and only column column_name
    data = {}
    for file in os.listdir(dir_path):
        if file.endswith(file_ending):
            data[file] = pd.read_table(os.path.join(dir_path, file), usecols=[column_name])

    # triangular jaccard similarity of descriptions
    jaccard_similarities = np.zeros((len(data), len(data)))
    for i, (file1, df1) in enumerate(data.items()):
        for j, (file2, df2) in enumerate(data.items()):
            if i > j:
                continue
            descriptions1 = set(df1[column_name])
            descriptions2 = set(df2[column_name])
            jaccard_similarities[i, j] = len(descriptions1.intersection(descriptions2)) / len(descriptions1.union(descriptions2))

    # plot heatmap
    xticks = [file.split("_")[0] for file in data.keys()]
    if biome_dict:
        try:
            biome_dict = {k: v.split(":")[-1] for k, v in biome_dict.items()}
            xticks = [x + " (" + biome_dict[x] + ")" for x in xticks]
        except KeyError:
            print("Biome_dict not compatible with xticks")

    sns.heatmap(jaccard_similarities, xticklabels=xticks, yticklabels=xticks, cmap="YlGnBu", annot=True, ax=axis)
    plt.title(title)


class MGnifyData:
    # TODO Testing needs to be done
    # TODO add time_threshold to all functions as parameter
    # TODO check whether retry and timeout are added to all requests
    # TODO repeated samples in the same study but different runs are not handled
    
    # BE MINDFUL: id == primary_accession in context of studies

    def __init__(self, cache_folder: Union[str, Path]):
        self.cache_folder = Path(cache_folder)
        self.base_api = "https://www.ebi.ac.uk/metagenomics/api/latest"

    def load_checkpoint(self, file_path: Union[str, Path], time_threshold: int = 24):
        """
        Load progress from checkpoint if it exists and is still valid (within the last {time_threshold} hours).
        """
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                checkpoint = json.load(file)
                # Check if the cache is within the last 24 hours
                last_saved_time = datetime.fromisoformat(checkpoint.get("timestamp"))
                if datetime.now() - last_saved_time < timedelta(hours=time_threshold):
                    return {k: v for k, v in checkpoint.items() if k != "timestamp"}
        return {}  # Return empty data

    def save_checkpoint(self, data: Dict, file_path: Union[str, Path]):
        """
        Save progress to checkpoint file, with the current timestamp.
        """
        data["timestamp"] = datetime.now().isoformat()  # Store exact timestamp
        with open(file_path, "w") as file:
            json.dump(data, file)

    
    def filter_download_links_for_most_recent_link(self, links: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Get the most recent link from a list of download links based on the pipeline version in the link.
        """
        # Get the version of each link
        versions = {study_id: max([float(link.split("/")[-3]) for link in study_links]) for study_id, study_links in links.items() if study_links and not print(study_links)}
        # Get the most recent link for each study
        most_recent_links = {study_id: [link for link in study_links if float(link.split("/")[-3]) == versions[study_id]][:1] for study_id, study_links in links.items() if study_links}
        return most_recent_links

    def _get_download_urls_for_label_with_start(self, session: requests_session, study_id: str, label_start_str: str) -> Dict[str, List[str]]:
        """
        Helper function to get the links for downloading specific "study-download" with label starting with label_start_str
        """
        downloads_path = f"{self.base_api}/studies/{study_id}/downloads"
        downloads = session.get(downloads_path).json()["data"]
        urls = []
        for download in downloads:
            if download["attributes"]["description"]["label"].startswith(label_start_str):
                urls.append(download["links"]["self"])
        return {study_id: urls}

    def get_study_accessions(self, page_size: int = 250, cache_time_threshold: int = 24*7, desired_biomes: Optional[str] = None) -> List[str]:
        """
        Get all study accessions from MGnify API that have a specific biome (if specified else all studies).

        Args:
            page_size (int): The number of studies to retrieve per page. Max allowed: 250.
            cache_time_threshold (int): The number of hours the previous cache is valid for.
            desired_biomes (str): The biomes to filter the studies by. If None, all studies are returned.
                The strings can also just be a part of the biome name.

        Returns:
            List[str]: A list of study accessions.
        """
        cache_file = self.cache_folder / "study_accessions.json"
        cached_data = self.load_checkpoint(cache_file, time_threshold=cache_time_threshold)
        study_accessions = cached_data.get("study_accessions", set())
        next_page = cached_data.get("next_page_url")

        with requests_session() as session:
            url = next_page or f"{self.base_api}/studies?page_size={page_size}&page=1"
            session.mount("http://", HTTP_ADAPTER_FOR_REQUESTS)
            session.mount("https://", HTTP_ADAPTER_FOR_REQUESTS)

            while url:
                response = session.get(url, timeout=5).json()
                studies = response.get("data", [])
                for study in studies:
                    biome = study["relationships"]["biomes"]["data"][0]["id"]
                    if desired_biomes is None or any(d in biome for d in desired_biomes):
                            study_accessions.add(study["id"])

                url = response.get("links", {}).get("next")
                self.save_checkpoint({"study_accessions": study_accessions, "next_page_url": url}, cache_file)

        return study_accessions

    def get_download_links_for_studies(self, study_ids, label_start_str: str = "Complete GO", desired_biomes: Optional[str] = None, max_workers: int = 10) -> Dict[str, List[str]]:
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

        Args:
            label_start_str (str): The start of the label of the analysis summary ("study-download") to download.
            desired_biomes (str): The biomes to filter the studies by. If None, all studies are returned.   # TODO not used yet
            max_workers (int): The maximum number of workers to use for concurrent requests.
            study_ids (list): The list of study IDs to download the links for. If None, all studies are considered by getting their accession id first.

        Returns:
            dict: A dictionary of study IDs and the desired corresponding download links.
        """
        if not study_ids:
            study_ids = self.get_study_accessions(desired_biomes=desired_biomes)

        with requests_session() as session:
            session.mount("http://", HTTP_ADAPTER_FOR_REQUESTS)
            session.mount("https://", HTTP_ADAPTER_FOR_REQUESTS)
            
            checkpoint_file = os.path.join(self.cache_folder, f"download_links_{label_start_str}_{desired_biomes}_{hasher(frozenset(study_ids))}.json")
            checkpoint_data = self.load_checkpoint(checkpoint_file)

            # Load existing progress if available and still valid
            result = checkpoint_data.get("result", {})
            print("cached:\n",result)
            processed_ids = set(result.keys())

            # Filter out already processed studies
            study_ids_to_process = [study_id for study_id in study_ids if study_id not in processed_ids]
            
            # Save "result" checkpoint every `save_interval` studies
            save_interval = 250

            # Use ThreadPoolExecutor to fetch download links concurrently
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._get_download_urls_for_label_with_start, session, study_id, label_start_str): study_id
                    for study_id in study_ids_to_process
                }
                
                completed = 0
                for future in as_completed(futures):
                    pipelines_giving_desired_downloads = future.result()
                    result.update(pipelines_giving_desired_downloads)
                    completed += 1

                    # Save checkpoint every `save_interval` studies
                    if completed % save_interval == 0:
                        self.save_checkpoint({"result": result}, checkpoint_file, time_threshold=720)


        # Final checkpoint save to capture any remaining progress
        self.save_checkpoint({"result": result}, checkpoint_file)

        return result
    
    def download_summary_for_studies(self, download_links: Dict[str, List[str]]) -> None:
        """
        Download some summaries of some studies given by download_links.

        Args:
            download_links (dict): A dictionary of study IDs and the corresponding desired download links.

        Returns:
            None
        """
        with requests_session() as session:
            session.mount("http://", HTTP_ADAPTER_FOR_REQUESTS)
            session.mount("https://", HTTP_ADAPTER_FOR_REQUESTS)

            for study_id, download_links in download_links.items():
                for link in download_links:
                    file_name = link.split("/")[-1]
                    file_path = os.path.join(self.cache_folder, study_id + "__" + file_name)
                    
                    # Skip if file already exists
                    if os.path.exists(file_path):
                        continue
                    
                    # Download the file within the session
                    response = session.get(link, stream=True)  # Use stream for efficient large file download
                    
                    # Check if the request was successful
                    if response.status_code == 200:
                        with open(file_path, "wb") as file:
                            for chunk in response.iter_content(chunk_size=8192):
                                file.write(chunk)
                    else:
                        print(f"Failed to download {link}. Status code: {response.status_code}")

    def get_study_sample_sizes(self, page_size: int = 250) -> Dict[str, int]:
        """
        Get the number of samples in each study, saving progress at the end of each page.

        Args:
            page_size (int): The number of studies to retrieve per page. Max allowed: 250.

        Returns:
            dict: A dictionary of study IDs and the corresponding sample sizes.
        """
        checkpoint_path = os.path.join(self.cache_folder, "study_sizes.json")
        with requests_session(self.base_api) as session:
            # Load existing progress if available and valid for today
            checkpoint = self.load_checkpoint(checkpoint_path, time_threshold=24*7)
            next_url = checkpoint.get("next_url")
            study_sizes = checkpoint.get("study_sizes", {})
            
            while True:
                # Retrieve studies ordered by `last_update`, limited to specified fields
                url = next_url or (
                    f"{self.base_api}/studies?ordering=last_update&page=1&page_size={page_size}&"
                    "fields[studies]=id,samples_count"
                )
                
                studies = session.get(url, timeout=10).json()

                # Process each study
                for resource in studies["data"]:
                    study_id = resource["id"]
                    
                    # Get and store the sample size for the study
                    sample_size = int(resource["attributes"]["samples-count"])
                    study_sizes[study_id] = sample_size

                # Save progress after each page, including the current date
                self.save_checkpoint({"next_url": url, "study_sizes": study_sizes}, checkpoint_path)

                url = studies["links"].get("next")

        return study_sizes
    
    def get_secondary_accessions(self, primary_study_accessions: list):
        """
        Get secondary accessions for each study in the list of study_ids.
        """
        # TODO caching is not used optimally
        chechpoint_file = os.path.join(self.cache_folder, f"primary_to_secondary_mappings.json")
        sec_accessions = self.load_checkpoint(chechpoint_file)["primary_to_secondary_mappings"]

        url = "{self.base_api}/studies/{}?fields[studies]=accession,secondary_accession"

        with requests_session(self.base_api) as session:
            session.mount("http://", HTTP_ADAPTER_FOR_REQUESTS)
            session.mount("https://", HTTP_ADAPTER_FOR_REQUESTS)

            for primary_accession in primary_study_accessions:
                if primary_accession not in sec_accessions:
                    study = session.get(url.format(primary_accession)).json()
                    sec_accessions[primary_accession] = study.secondary_accession

        self.save_checkpoint({"primary_to_secondary_mappings": sec_accessions}, chechpoint_file)

        # Return only the secondary accessions of the input primary accessions
        return {k: v for k, v in sec_accessions.items() if k in primary_study_accessions}
    
    def get_primary_from_secondary_accession(self, secondary_study_accessions: list):
        """
        Get the primary accession for each secondary accession in the list of secondary_accessions.
        """
        # TODO caching is not used optimally
        checkpoint_file = os.path.join(self.cache_folder, f"secondary_to_primary_mappings.json")
        primary_accessions = self.load_checkpoint(checkpoint_file)["secondary_to_primary_mappings"]

        url = "{self.base_api}/studies/{}?fields[studies]=accession,secondary_accession"

        with requests_session(self.base_api) as session:
            session.mount("http://", HTTP_ADAPTER_FOR_REQUESTS)
            session.mount("https://", HTTP_ADAPTER_FOR_REQUESTS)

            for secondary_accession in secondary_study_accessions:
                if secondary_accession not in primary_accessions:
                    study = session.get(url.format(secondary_accession)).json()
                    primary_accessions[secondary_accession] = study["id"]

        self.save_checkpoint({"secondary_to_primary_mappings": primary_accessions}, checkpoint_file)

        # Return only the primary accessions of the input secondary accessions
        return {k: v for k, v in primary_accessions.items() if k in secondary_study_accessions}

    def get_run_id_to_sample_id_dict(self, study_id: str, page_size=250, cache_time_threshold=24*7*365*10) -> Dict[str, str]:
        """
        Get the run_id-sample_id pair of the runs of a specific study. The run id can also be the assembly id depending on the analysis.

        Args:
            study_id (str): The MGnify study ID to get the run ID to sample ID mapping for.
            page_size (int): The number of analyses to retrieve per page. Max allowed: 250.
            cache_time_threshold (int): The number of hours the previous cache is valid for.

        Returns:
            dict: A dictionary of run IDs and their corresponding sample ID.
                Be mindful that different run IDs can have the same sample ID.
        """
        cache_file = os.path.join(self.cache_folder, f"runId_sampleId_pairs_{study_id}.json")
        cached_data = self.load_checkpoint(cache_file, time_threshold=cache_time_threshold)
        result_dict = cached_data.get("id_pairs", {})
        next_page = cached_data.get("next_page_url")

        with requests_session() as session:
            url = next_page or f"{self.base_api}/studies/{study_id}/analyses?page_size={page_size}&page=1&fields[analysis-jobs]=id,assembly,run,sample"

            while url:
                response = session.get(url).json()
                analyses = response["data"]

                # Check whether we should get assembly id or run id
                # The hope is that first analysis will be the same as the other ones. We'll see after using this function a lot
                if analyses[0]["relationships"]["run"] is None:
                    print("No run id for this analysis")
                    what_id = "assembly"
                else:
                    what_id = "run"

                for a in analyses:
                    run_id = a["relationships"][what_id]["data"]["id"]
                    result_dict[run_id] = a["relationships"]["sample"]["data"]["id"]
                
                url = response["links"]["next"]
                self.save_checkpoint({"id_pairs": result_dict, "next_page_url": url}, cache_file)

        return result_dict

    def download_metadata_for_studies(self, study_ids: List[str], page_size: int = 250, cache_time_threshold=24*7*365*10) -> None:
        """
        Download metadata for a list of studies. It includes the metadata from MGnify, BioSamples, and ELIXIR Contextual Data ClearingHouse
        """
        for study_id in study_ids:
            self.get_metadata_for_study(study_id, page_size, cache_time_threshold)

    # TODO using csv for metadata is more efficient, but we need new caching mechanism (maybe flexible)
    def get_metadata_for_study(self, study_id: str, page_size: int = 250, cache_time_threshold=24*7*365*10) -> pd.DataFrame: # TODO add additinal functionality of getting the runs and change get_assembly_id_to_sample_id_dict based on this as input
        """
        Get metadata for a specific study. It includes the metadata from MGnify, BioSamples, and ELIXIR Contextual Data ClearingHouse
        """
        cache_file = self.cache_folder / f"{study_id}_metadata.json"
        cached_metadata = self.load_checkpoint(cache_file, time_threshold=cache_time_threshold)
        metadata = cached_metadata.get("metadata", {})
        next_page = cached_metadata.get("next_page_url") 

        # get all sample ids for the study + mgnify metadata
        with requests_session() as session:
            url = next_page or f"{self.base_api}/studies/{study_id}/samples?fields[samples]=id,biosample,latitude,longitude,species,sample_metadata&page_size={page_size}&page=1"

            while url:
                biosamples_ids = {}
                response = session.get(url)
                if response.status_code != 200: # TODO remove if this function never gives error
                    print(f"Failed to get metadata for {url}\n. Status code: {response.status_code}")
                    break
                response = response.json()
                samples = response["data"]
                samples = [sample for sample in samples if sample["id"] not in metadata]
                additional_metadata = {}
                for sample in samples:
                    attributes = sample.get("attributes", {})
                    if "biosample" in attributes:
                        biosamples_ids[sample["id"]] = attributes["biosample"]

                    # get mgnify metadata
                    additional_metadata[sample["id"]] = {d["key"]: d["value"] for d in attributes.get("sample_metadata", [])}
                    additional_metadata[sample["id"]]["biosample_id"] = attributes.get("biosample", None)
                    additional_metadata[sample["id"]]["sample_id"] = sample["id"]
                    
                    additional_metadata[sample["id"]]["latitude"] = attributes.get("latitude", None)
                    additional_metadata[sample["id"]]["longitude"] = attributes.get("longitude", None)
                    additional_metadata[sample["id"]]["species"] = attributes.get("species", None)

                # get all BioSamples metadata
                biosamples_metadata = get_metadata_of_samples(biosamples_ids)
                for sample_id, sample_metadata in biosamples_metadata.items():
                    additional_metadata[sample_id].update(sample_metadata)

                # get elixir metadata
                elixir_metadata = get_elixir_contextual_data_clearinghouse_metadata(list(biosamples_ids.keys()))
                for sample_id, sample_metadata in elixir_metadata.items():
                    additional_metadata[sample_id].update(sample_metadata)

            
                metadata.update(additional_metadata)

                # Get the next page URL (if available)
                url = response.get("links", {}).get("next")

                # Save progress after each page
                self.save_checkpoint({"metadata": metadata, "next_page_url": url}, cache_file)

        metadata = pd.DataFrame(metadata)   # rows are features, columns are samples (as in summary data)

        return metadata
    
    def combine_metadata_with_corresponding_study_summary(self,
                                                          study_id: str, 
                                                          summary_link: str,
                                                          metadata_features: Optional[List[str]] = None,
                                                          join_type: str = "inner",
                                                          cache_time_threshold=24*7*365*10) -> pd.DataFrame:
        """
        Combine the metadata of a study with the corresponding study summary

        Args:
            study_id (str): The MGnify study ID to combine the metadata with the summary for.
            summary_link (str): The link to the summary data of the study.
            metadata_features (list): The metadata features to include in the combined data.
            join_type (str): The type of join to use when combining the metadata with the summary.
            cache_time_threshold (int): The number of hours the previous cache is valid for.
        """
        # TODO not finished as we may not even want this. One reason can be that not all samples in the summary are in the metadata and so our dataset will 
        # be either incomplete or we need to remove some samples from the summary reducing the size of the dataset

        # TODO We may want to drop some samples in metadata if they don't have a desired feature (e.g. Health state);
        # and based on join_type, this may reduce the summary data size too.

        cache_file = os.path.join(self.cache_folder, f"combined_data_{study_id}.csv")
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file)

        # Get and filter metadata
        metadata = self.get_metadata_for_study(study_id)
        if metadata_features:
            metadata = metadata.loc[metadata_features]

        # Download summary if not downloaded yet
        self.download_summary_for_studies({study_id: [summary_link]})
        summary_file_name = summary_link.split("/")[-1]
        summary = pd.read_table(os.path.join(self.cache_folder, summary_file_name))

        run_id_to_sample_id = self.get_run_id_to_sample_id_dict(study_id)

        # Change sample id to run id in metadata
        column_mapper = {}
        for key, value in run_id_to_sample_id.items():
            if value not in column_mapper:
                column_mapper[value] = []
            column_mapper[value].append(key)

        new_columns = {new_name: metadata[old_name] for old_name, new_names in column_mapper.items() for new_name in new_names}
        metadata = metadata.drop(columns=list(column_mapper.keys()))
        metadata = metadata.join(pd.DataFrame(new_columns))
        
        if "GO" in summary_link or "InterPro" in summary_link:  # TODO make this optional
            summary = summary.droplevel(["GO", "category"])
        summary = summary.T
        metadata = metadata.T

        combined_data = summary.join(metadata, how=join_type)   # rows are samples, columns are features
        combined_data.to_csv(cache_file)

        return combined_data

        

@logger.catch
def get_elixir_contextual_data_clearinghouse_metadata(sample_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Get the metadata of a list of samples from ELIXIR Contextual Data ClearingHouse
    """
    base_api = "https://www.ebi.ac.uk/ena/clearinghouse/api"
    to_return = {}
    with requests_session() as session:
        for id in sample_ids:
            metadata = session.get(f"{base_api}/curations/{id}").json().get("curations", [])
            metadata = {d["attributePost"] + "__elixir": d["valuePost"] for d in metadata}
            to_return[id] = metadata
    
    return to_return







            
            

