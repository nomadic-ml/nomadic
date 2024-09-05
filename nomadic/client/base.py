import os
from pydantic import BaseModel
from typing import Optional

from .http_client import HttpClient, HttpClientOptions

DEFAULT_BASE_API_URL = "https://app.nomadicml.com/api/"

_CLIENT = None


class ClientOptions(BaseModel):
    api_key: str
    base_url: Optional[str] = None


class NomadicClient(HttpClient):
    auto_sync_enabled: bool = True

    def __init__(self, config: ClientOptions, auto_sync_enabled: bool = True) -> None:
        super().__init__(
            HttpClientOptions(
                api_key=config.api_key, base_url=config.base_url or _get_base_url()
            )
        )
        from nomadic.client.resources import Models, Experiments, ExperimentResults

        self.auto_sync_enabled = auto_sync_enabled

        self.models = Models(self)
        self.experiments = Experiments(self)
        self.experiment_results = ExperimentResults(self)
        _set_client(self)


def get_client() -> NomadicClient:
    global _CLIENT
    if _CLIENT is None:
        print("NomadicClient not initialized. Configuring placeholder client")
        _set_client(
            NomadicClient(
                ClientOptions(api_key="", base_url=""), auto_sync_enabled=False
            )
        )
    return _CLIENT


def _set_client(client: NomadicClient) -> None:
    global _CLIENT
    _CLIENT = client


def _get_base_url() -> str:
    return (
        DEFAULT_BASE_API_URL
        if "NOMADIC_BASE_API_URL" not in os.environ
        else os.environ["NOMADIC_BASE_API_URL"]
    )
