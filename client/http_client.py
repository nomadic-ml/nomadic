from pydantic import BaseModel
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Any


MAX_RETRY_COUNT = 5


class HttpClientOptions(BaseModel):
    api_key: str
    base_url: str


class HttpClient:
    api_key: str
    base_url: str
    session: requests.Session

    @staticmethod
    def sanitize_base_url(url: str) -> str:
        return url.strip().rstrip("/")

    @staticmethod
    def create_session() -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=MAX_RETRY_COUNT,
            allowed_methods=["GET", "POST", "DELETE", "PATCH", "HEAD", "OPTIONS"],
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def __init__(self, config: HttpClientOptions) -> None:
        self.api_key = config.api_key
        self.base_url = HttpClient.sanitize_base_url(config.base_url)
        self.session = HttpClient.create_session()

    def request(self, method: str, url: str, **kwargs) -> Any:
        resp = self.session.request(
            method=method,
            url=self.base_url + url,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": f"{self.api_key}",
                # "Authorization": f"Bearer {self.api_key}",
            },
            **kwargs,
        )
        resp.raise_for_status()
        if resp.content:
            return resp.json()
        else:
            return None
