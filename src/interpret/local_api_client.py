import contextlib
from typing import Any, Optional

import httpx
import orjson
from neuron_explainer.api_client import (
    ApiClient,
    exponential_backoff,
    is_api_error,
    API_HTTP_HEADERS,
)


class LocalApiClient(ApiClient):
    def __init__(
        self, model_name: str, max_concurrent: Optional[int] = None, cache: bool = False
    ):
        super().__init__(model_name, max_concurrent, cache)
        self.base_url = "http://localhost:5001/v1"  # depending on llama_server.py

    @exponential_backoff(retry_on=is_api_error)
    async def make_request(
        self, timeout_seconds: Optional[int] = None, **kwargs: Any
    ) -> dict[str, Any]:
        if self._cache is not None:
            key = orjson.dumps(kwargs)
            if key in self._cache:
                return self._cache[key]
        async with contextlib.AsyncExitStack() as stack:
            if self._concurrency_check is not None:
                await stack.enter_async_context(self._concurrency_check)
            http_client = await stack.enter_async_context(
                httpx.AsyncClient(timeout=timeout_seconds)
            )
            # Send request to the local API
            url = self.base_url + "/completions"
            response = await http_client.post(
                url, headers=API_HTTP_HEADERS, json=kwargs
            )
        # The response json has useful information but the exception doesn't include it, so print it
        # out then reraise.
        try:
            response.raise_for_status()
        except Exception as e:
            print(response.json())
            raise e
        if self._cache is not None:
            self._cache[key] = response.json()
        return response.json()
