from __future__ import annotations

from nomadic.client import NomadicClient


class APIResource:
    _client: NomadicClient

    def __init__(self, client: NomadicClient):
        self._client = client
