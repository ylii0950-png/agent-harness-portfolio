"""Fixed-origin HTTPS, no redirects/proxies/retries, bounded JSON and SSE."""

import json
import time
from collections.abc import Generator, Iterator, Mapping
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, OpenerDirector, ProxyHandler, Request, build_opener

from agent_harness_lab.integration import FailureKind, api_failure, failure

_ALLOWED = {
    "https://auth.openai.com/api/accounts/oauth/token",
    "https://auth.openai.com/.well-known/jwks.json",
    "https://auth.openai.com/api/accounts/oauth/revoke",
    "https://api.openai.com/v1/models",
    "https://api.openai.com/v1/responses",
}
_LIMIT = 8 * 1024 * 1024


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


class HTTPTransport:
    def __init__(self, *, timeout: float = 30.0) -> None:
        if timeout <= 0 or timeout > 60:
            raise ValueError("HTTP timeout must be between 0 and 60 seconds")
        self.timeout = timeout
        self._opener: OpenerDirector = build_opener(ProxyHandler({}), _NoRedirect())

    def _open(self, method: str, url: str, data: bytes | None, headers: Mapping[str, str]) -> Any:
        if url not in _ALLOWED:
            raise ValueError("endpoint is outside the integration allowlist")
        try:
            return self._opener.open(
                Request(url, data=data, headers=dict(headers), method=method), timeout=self.timeout
            )
        except HTTPError as exc:
            with exc:
                try:
                    body = json.loads(exc.read(65536))
                except (ValueError, UnicodeError):
                    body = None
                raise api_failure(exc.code, body, exc.headers.get("x-request-id")) from None
        except (URLError, OSError, TimeoutError):
            raise failure(FailureKind.API, "connection_failed") from None

    def json(
        self,
        method: str,
        url: str,
        *,
        data: bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        if url not in _ALLOWED:
            raise ValueError("endpoint is outside the integration allowlist")
        try:
            with self._open(method, url, data, headers or {}) as response:
                raw = response.read(_LIMIT + 1)
                if len(raw) > _LIMIT:
                    raise failure(FailureKind.PROTOCOL, "response_too_large")
                result = json.loads(raw)
                if not isinstance(result, dict):
                    raise ValueError("expected object")
                return result
        except (ValueError, UnicodeError):
            raise failure(FailureKind.PROTOCOL, "invalid_json") from None
        except (OSError, TimeoutError):
            raise failure(FailureKind.API, "connection_failed") from None

    def events(
        self, url: str, body: dict[str, object], token: str
    ) -> Generator[dict[str, Any], None, None]:
        headers = {
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        try:
            with self._open("POST", url, json.dumps(body).encode(), headers) as response:
                content_type = response.headers.get("Content-Type", "").strip().lower()
                if content_type and content_type.split(";", 1)[0] != "text/event-stream":
                    raise failure(FailureKind.PROTOCOL, "expected_event_stream")
                yield from parse_sse(response)
        except (OSError, TimeoutError):
            raise failure(FailureKind.INTERRUPTED, "stream_interrupted") from None


def parse_sse(lines: Any) -> Iterator[dict[str, Any]]:
    """Honor SSE data framing; terminal success is checked by the provider."""
    pending: list[str] = []
    consumed = 0
    started = time.monotonic()
    for _ in range(50000):
        raw = lines.readline(_LIMIT + 1)
        if not raw:
            break
        consumed += len(raw)
        if consumed > _LIMIT or time.monotonic() - started > 120:
            raise failure(FailureKind.INTERRUPTED, "stream_limit")
        try:
            line = raw.decode("utf-8").rstrip("\r\n")
        except UnicodeError:
            raise failure(FailureKind.PROTOCOL, "invalid_sse") from None
        if line.startswith("data:"):
            pending.append(line[5:].removeprefix(" "))
        elif not line and pending:
            data = "\n".join(pending)
            pending.clear()
            if data == "[DONE]":
                return
            try:
                event = json.loads(data)
            except ValueError:
                raise failure(FailureKind.PROTOCOL, "invalid_sse") from None
            if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                raise failure(FailureKind.PROTOCOL, "invalid_sse")
            yield event
    else:
        raise failure(FailureKind.INTERRUPTED, "stream_limit")
    if pending:
        raise failure(FailureKind.INTERRUPTED, "truncated_sse")
