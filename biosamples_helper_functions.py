from typing import Any, Dict
from loguru import logger
from requests import Session as RequestsSession

from global_vars import HTTP_ADAPTER_FOR_REQUESTS

BASE_API_URL = "https://www.ebi.ac.uk/biosamples"

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
    
    with RequestsSession() as biosample_session:
        biosample_session.mount("http://", HTTP_ADAPTER_FOR_REQUESTS)
        biosample_session.mount("https://", HTTP_ADAPTER_FOR_REQUESTS)

        for sample_id, biosmaple_id in ids.items():
            response = biosample_session.get(f"{BASE_API_URL}/samples/{biosmaple_id}", timeout=10)
            characteristics = response.json().get("characteristics", {})
            metadata = {x + "__biosamples": characteristics[x][0]["text"] for x in characteristics if characteristics[x][0].get("tag", "") == "attribute"} 
            to_return[sample_id] = metadata
    return to_return
    