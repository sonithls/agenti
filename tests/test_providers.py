import io
import json
import os
import unittest
from datetime import datetime, timedelta, timezone
from email.message import Message
from email.utils import format_datetime
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from importance_agent import (
    AuthenticationError, ChatProvider, InvalidResponseError, ProviderConfig, ProviderError,
    ProviderTimeoutError, RateLimitError, TransportError,
)
from importance_agent.prompts import EVALUATION_SCHEMA


def wire_response(content='{"ok": true}', finish_reason="stop"):
    return io.BytesIO(json.dumps({"choices": [{"message": {"content": content},
                                              "finish_reason": finish_reason}]}).encode())


def http_error(code, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return HTTPError("https://example.com", code, "sensitive server message", headers,
                     io.BytesIO(b"secret response body"))


class ProviderTests(unittest.TestCase):
    def provider(self, **kwargs):
        args = {"base_url": "https://example.com/v1/", "model": "example", "api_key_env": None,
                "max_retries": 0}
        args.update(kwargs)
        provider = ChatProvider(ProviderConfig(**args))
        provider._opener = Mock()
        return provider

    def complete(self, provider):
        return provider.complete([{"role": "user", "content": "Evaluate supplied news"}], schema=EVALUATION_SCHEMA)

    def test_post_body_auth_and_response_extraction(self):
        provider = self.provider(api_key_env="TEST_IMPORTANCE_KEY", extra_body={"max_tokens": 1000})
        provider._opener.open.return_value = wire_response()
        with patch.dict(os.environ, {"TEST_IMPORTANCE_KEY": "test-secret"}):
            self.assertEqual(self.complete(provider), '{"ok": true}')
        request = provider._opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.com/v1/chat/completions")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
        body = json.loads(request.data)
        self.assertEqual(body["max_tokens"], 1000)
        self.assertFalse(body["stream"])
        self.assertNotIn("response_format", body)
        self.assertNotIn("test-secret", json.dumps(provider.provenance()))

    def test_structured_modes_and_temperature_omission(self):
        for mode in ("json_object", "json_schema"):
            provider = self.provider(structured_output=mode, temperature=None)
            provider._opener.open.return_value = wire_response()
            self.complete(provider)
            body = json.loads(provider._opener.open.call_args.args[0].data)
            self.assertEqual(body["response_format"]["type"], mode)
            self.assertNotIn("temperature", body)
            if mode == "json_schema":
                self.assertEqual(body["response_format"]["json_schema"]["schema"], EVALUATION_SCHEMA)

    def test_missing_or_invalid_key_fails_before_network(self):
        for key in (None, "", "bad\nheader"):
            provider = self.provider(api_key_env="TEST_IMPORTANCE_KEY")
            with patch.dict(os.environ, {} if key is None else {"TEST_IMPORTANCE_KEY": key}, clear=True):
                with self.assertRaises(AuthenticationError):
                    self.complete(provider)
            provider._opener.open.assert_not_called()

    def test_authentication_and_non_retryable_errors_are_sanitized(self):
        for status, error_type in ((401, AuthenticationError), (403, AuthenticationError),
                                   (400, ProviderError), (302, ProviderError), (413, ProviderError)):
            provider = self.provider(max_retries=2)
            provider._opener.open.side_effect = http_error(status)
            with self.assertRaises(error_type) as caught:
                self.complete(provider)
            self.assertNotIn("sensitive", str(caught.exception))
            self.assertNotIn("secret", str(caught.exception))
            self.assertEqual(caught.exception.status_code, status)
            self.assertEqual(provider._opener.open.call_count, 1)

    @patch("importance_agent.providers.chat.time.sleep")
    def test_rate_limit_retry_after_and_recovery(self, sleep):
        provider = self.provider(max_retries=2)
        provider._opener.open.side_effect = [http_error(429, "2"), wire_response()]
        self.complete(provider)
        sleep.assert_called_once_with(2)
        self.assertEqual(provider._opener.open.call_count, 2)

    @patch("importance_agent.providers.chat.time.sleep")
    def test_excessive_retry_after_does_not_retry_early(self, sleep):
        provider = self.provider(max_retries=2, max_retry_delay_seconds=5)
        provider._opener.open.side_effect = http_error(429, "60")
        with self.assertRaises(RateLimitError):
            self.complete(provider)
        sleep.assert_not_called()
        self.assertEqual(provider._opener.open.call_count, 1)

    @patch("importance_agent.providers.chat.time.sleep")
    def test_retry_exhaustion(self, sleep):
        for status in (429, 500, 503):
            provider = self.provider(max_retries=2)
            provider._opener.open.side_effect = [http_error(status) for _ in range(3)]
            with self.assertRaises(ProviderError):
                self.complete(provider)
            self.assertEqual(provider._opener.open.call_count, 3)

    def test_timeout_and_transport_failures(self):
        for cause, error_type in ((URLError(TimeoutError()), ProviderTimeoutError),
                                   (TimeoutError(), ProviderTimeoutError),
                                   (URLError("private error"), TransportError)):
            provider = self.provider()
            provider._opener.open.side_effect = cause
            with self.assertRaises(error_type) as caught:
                self.complete(provider)
            self.assertNotIn("private", str(caught.exception))

    def test_response_size_limit_and_invalid_envelopes(self):
        provider = self.provider(max_response_bytes=8)
        provider._opener.open.return_value = wire_response()
        with self.assertRaises(InvalidResponseError):
            self.complete(provider)
        for response in (io.BytesIO(b"not JSON"), io.BytesIO(b"{}"), wire_response(None),
                          wire_response(finish_reason="length"), wire_response(finish_reason="tool_calls")):
            provider = self.provider()
            provider._opener.open.return_value = response
            with self.assertRaises(InvalidResponseError):
                self.complete(provider)

    def test_retry_after_http_date(self):
        value = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=10), usegmt=True)
        self.assertTrue(8 <= ChatProvider._retry_after(value) <= 10)
        self.assertIsNone(ChatProvider._retry_after("invalid"))
        self.assertIsNone(ChatProvider._retry_after("nan"))

    def test_score_settings_change_provenance(self):
        first = self.provider(temperature=0).provenance()
        self.assertEqual(first, self.provider(temperature=0).provenance())
        self.assertNotEqual(first["configuration_id"], self.provider(temperature=1).provenance()["configuration_id"])
