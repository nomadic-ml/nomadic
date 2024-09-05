from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nomadic.client.base import NomadicClient


class APIResource:
    _client: "NomadicClient"

    def __init__(self, client: "NomadicClient"):
        self._client = client
