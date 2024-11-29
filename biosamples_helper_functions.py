from typing import Any, Dict
from loguru import logger
from requests import Session as RequestsSession

from helper_function import config_session

BIOSAMPLES_BASE_API_URL = "https://www.ebi.ac.uk/biosamples"

@logger.catch(reraise=True, onerror=lambda _: logger.error("An error occurred while getting the metadata of the samples from the BioSamples."))
def get_metadata_of_samples(ids: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    Get the metadata of a list of samples from the BioSamples.

    Args:
    sample_ids: A dictionary with the sample ids as keys and the BioSamples ids as values.

    Returns:
    A dictionary with the sample ids as keys and the attribute related metadata of the samples as values.
    """
    to_return = {}
    
    with RequestsSession() as sessions:
        config_session(sessions)

        for sample_id, biosmaple_id in ids.items():
            response = sessions.get(f"{BIOSAMPLES_BASE_API_URL}/samples/{biosmaple_id}", timeout=10)
            characteristics = response.json().get("characteristics", {})
            metadata = {x + "__biosamples": characteristics[x][0]["text"] for x in characteristics if characteristics[x][0].get("tag", "") == "attribute"} 
            to_return[sample_id] = metadata
    return to_return
    