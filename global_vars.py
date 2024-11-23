from pathlib import Path
from requests.adapters import HTTPAdapter, Retry

# Adatapter for requests to retry on some http errors up to 3 times
RETRIES_FOR_REQUESTS = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
HTTP_ADAPTER_FOR_REQUESTS = HTTPAdapter(max_retries=RETRIES_FOR_REQUESTS)

BASE_DIR = Path(__file__).resolve().parent