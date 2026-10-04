"""Dependency-free HTTP adapter for chat-completions-compatible endpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import socket
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ..errors import (
    AuthenticationError, InvalidResponseError, ProviderError, ProviderTimeoutError,
    RateLimitError, TransportError,
)
from ..models import ProviderConfig
from ..validation import parse_json_object
from .base import ProviderAdapter


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ChatProvider(ProviderAdapter):
    def __init__(self, config: ProviderConfig):
        self.config = config
        # Do not follow redirects carrying credentials or submitted content.
        self._opener = build_opener(_NoRedirects())

    def provenance(self) -> dict[str, str]:
        settings = {"endpoint": self.config.endpoint, "model": self.config.model,
                    "temperature": self.config.temperature,
                    "structured_output": self.config.structured_output,
                    "extra_body": dict(self.config.extra_body)}
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:20]
        return {"provider": self.config.base_url.rstrip("/"), "model": self.config.model,
                "configuration_id": "config_" + digest}

    def complete(self, messages: list[dict[str, str]], *, schema: dict[str, Any]) -> str:
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "User-Agent": "importance-agent/0.1.0"}
        if self.config.api_key_env is not None:
            key = os.environ.get(self.config.api_key_env, "")
            if not key.strip():
                raise AuthenticationError("The configured API-key environment variable is missing or empty.")
            if any(ord(c) < 32 or ord(c) > 126 for c in key):
                raise AuthenticationError("The configured API key contains invalid header characters.")
            headers["Authorization"] = "Bearer " + key
        body = dict(self.config.extra_body)
        body.update(model=self.config.model, messages=messages, stream=False)
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        if self.config.structured_output == "json_object":
            body["response_format"] = {"type": "json_object"}
        elif self.config.structured_output == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "importance_response", "strict": True, "schema": schema},
            }
        request = Request(self.config.endpoint, data=json.dumps(body, allow_nan=False).encode("utf-8"),
                          headers=headers, method="POST")
        for attempt in range(self.config.max_retries + 1):
            delay = None
            try:
                raw = self._request(request)
            except HTTPError as exc:
                status = exc.code
                retry_header = exc.headers.get("Retry-After") if exc.headers else None
                exc.close()
                if status in (401, 403):
                    raise AuthenticationError("Provider rejected authentication or access.", status_code=status) from None
                if status == 413:
                    raise ProviderError("Provider rejected the request size; reduce the input.", status_code=status) from None
                retryable = status in (408, 429, 500, 502, 503, 504)
                error_type = RateLimitError if status == 429 else ProviderError
                error = error_type(f"Provider returned HTTP {status}.", status_code=status)
                if not retryable or attempt == self.config.max_retries:
                    raise error from None
                delay = self._retry_after(retry_header)
                # Never retry earlier than the server requested. If that wait
                # exceeds our configured bound, surface the error immediately.
                if delay is not None and delay > self.config.max_retry_delay_seconds:
                    raise error from None
            except (URLError, OSError, HTTPException) as exc:
                reason = exc.reason if isinstance(exc, URLError) else exc
                if isinstance(reason, (TimeoutError, socket.timeout)):
                    error = ProviderTimeoutError("Provider request timed out.")
                else:
                    error = TransportError("Could not communicate with the configured provider.")
                if attempt == self.config.max_retries:
                    raise error from None
            else:
                return self._extract(raw)
            if delay is None:
                delay = min(2 ** attempt + random.uniform(0, 0.25), self.config.max_retry_delay_seconds)
            time.sleep(delay)
        raise AssertionError("Unreachable retry state")

    def _request(self, request: Request) -> bytes:
        deadline = time.monotonic() + self.config.timeout_seconds
        with self._opener.open(request, timeout=self.config.timeout_seconds) as response:
            chunks = []
            size = 0
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError
                chunk = response.read1(min(65_536, self.config.max_response_bytes + 1 - size))
                if not chunk:
                    return b"".join(chunks)
                size += len(chunk)
                if size > self.config.max_response_bytes:
                    raise InvalidResponseError("Provider response exceeds max_response_bytes.")
                chunks.append(chunk)

    @staticmethod
    def _retry_after(value: str | None) -> float | None:
        if not value:
            return None
        try:
            delay = float(value)
            return max(0.0, delay) if math.isfinite(delay) else None
        except ValueError:
            try:
                target = parsedate_to_datetime(value)
                if target.tzinfo is None:
                    target = target.replace(tzinfo=timezone.utc)
                return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None

    @staticmethod
    def _extract(raw: bytes) -> str:
        try:
            envelope = parse_json_object(raw.decode("utf-8"))
            choice = envelope["choices"][0]
            if choice.get("finish_reason") in ("length", "content_filter", "tool_calls", "function_call"):
                raise InvalidResponseError("Provider output was truncated, filtered, or was not a text answer.")
            content = choice["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise InvalidResponseError("Provider returned no generated text.")
            return content
        except (UnicodeError, KeyError, IndexError, TypeError, AttributeError):
            raise InvalidResponseError("Provider response is not a supported chat-completions envelope.") from None
