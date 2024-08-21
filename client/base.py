import os

from pydantic import BaseModel
from typing import (
    Optional,
)

from .http_client import HttpClient, HttpClientOptions


DEFAULT_BASE_API_URL = "https://app.nomadicml.com/api/"


class ClientOptions(BaseModel):
    api_key: str
    base_url: Optional[str] = None


class NomadicClient(HttpClient):
    def __init__(self, config: ClientOptions) -> None:
        super().__init__(
            HttpClientOptions(
                api_key=config.api_key, base_url=config.base_url or _get_base_url()
            )
        )
        _set_client(self)


def get_client() -> NomadicClient:
    global _CLIENT
    if _CLIENT is None:
        raise ValueError("NomadicClient not initialized")
    return _CLIENT


def _set_client(client: NomadicClient) -> None:
    global _CLIENT
    _CLIENT = client


def _get_base_url() -> str:
    return os.environ["NOMADIC_BASE_API_URL"] or DEFAULT_BASE_API_URL
