from __future__ import annotations

import httpx
import json
from typing import Any


class HTTPXRecorder:
    """Capture the last HTTPX request and response.

    Can be used either to create custom HTTPX clients with event hooks or to
    temporarily patch the global ``httpx`` clients via ``install`` and
    ``uninstall``.
    """

    def __init__(self) -> None:
        self.last_request: httpx.Request | None = None
        self.last_response: httpx.Response | None = None
        self.last_request_str: str | None = None
        self.last_response_str: str | None = None
        self._orig_async_send = None
        self._orig_send = None

    def create_async_client(self, **kwargs: Any) -> httpx.AsyncClient:
        async def record_request(request: httpx.Request) -> None:
            self.last_request = request
            self.last_request_str = self._format_request(request)

        async def record_response(response: httpx.Response) -> None:
            await response.aread()
            self.last_response = response
            self.last_response_str = self._format_response(response)

        hooks = {"request": [record_request], "response": [record_response]}
        return httpx.AsyncClient(event_hooks=hooks, **kwargs)

    def create_client(self, **kwargs: Any) -> httpx.Client:
        def record_request(request: httpx.Request) -> None:
            self.last_request = request
            self.last_request_str = self._format_request(request)

        def record_response(response: httpx.Response) -> None:
            response.read()
            self.last_response = response
            self.last_response_str = self._format_response(response)

        hooks = {"request": [record_request], "response": [record_response]}
        return httpx.Client(event_hooks=hooks, **kwargs)

    # ------------------------------------------------------------------
    # Global patch helpers
    # ------------------------------------------------------------------
    def install(self) -> None:
        """Monkeypatch ``httpx`` clients to record requests and responses."""
        if self._orig_async_send is not None:
            return  # Already installed

        self._orig_async_send = httpx.AsyncClient.send
        self._orig_send = httpx.Client.send

        async def async_send(client: httpx.AsyncClient, request: httpx.Request, *args: Any, **kwargs: Any) -> httpx.Response:
            self.last_request = request
            self.last_request_str = self._format_request(request)
            response = await self._orig_async_send(client, request, *args, **kwargs)
            await response.aread()
            self.last_response = response
            self.last_response_str = self._format_response(response)
            return response

        def send(client: httpx.Client, request: httpx.Request, *args: Any, **kwargs: Any) -> httpx.Response:
            self.last_request = request
            self.last_request_str = self._format_request(request)
            response = self._orig_send(client, request, *args, **kwargs)
            response.read()
            self.last_response = response
            self.last_response_str = self._format_response(response)
            return response

        httpx.AsyncClient.send = async_send  # type: ignore[assignment]
        httpx.Client.send = send  # type: ignore[assignment]

    def uninstall(self) -> None:
        """Revert ``httpx`` monkeypatching."""
        if self._orig_async_send is None:
            return
        httpx.AsyncClient.send = self._orig_async_send  # type: ignore[assignment]
        httpx.Client.send = self._orig_send  # type: ignore[assignment]
        self._orig_async_send = None
        self._orig_send = None

    @staticmethod
    def _format_request(request: httpx.Request) -> str:
        """Return JSON string with method, URL, headers and body."""
        body: str
        if isinstance(request.content, (bytes, bytearray)):
            try:
                body = request.content.decode()
            except Exception:
                body = repr(request.content)
        else:
            body = str(request.content)
        try:
            body_json = json.loads(body)
        except Exception:
            body_json = body

        data = {
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "body": body_json,
        }
        return json.dumps(data, indent=4)

    @staticmethod
    def _format_response(response: httpx.Response) -> str:
        """Return JSON string with status code, URL, headers and body."""
        text = response.text
        try:
            body_json = json.loads(text)
        except Exception:
            body_json = text

        data = {
            "status_code": response.status_code,
            "url": str(response.request.url),
            "headers": dict(response.headers),
            "body": body_json,
        }
        return json.dumps(data, indent=4)
