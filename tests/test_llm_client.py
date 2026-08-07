from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List

from data_synthetic import build_generation_error
from synthetic.infrastructure.llm_client import (
    LLMClientConfig,
    LLMResponseError,
    VLLMClient,
    VLLMHTTPError,
    extract_message_text,
)
from synthetic.infrastructure.retry import RetryConfig


class StubResponse:
    def __init__(
        self,
        status_code: int,
        payload: Any,
        *,
        reason_phrase: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.reason_phrase = reason_phrase or (
            "Bad Request" if status_code == 400 else "Service Unavailable"
        )
        self.url = "http://localhost:8000/v1/chat/completions"
        self.text = (
            payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True)
        )

    def json(self) -> Any:
        if isinstance(self._payload, str):
            raise ValueError("not JSON")
        return self._payload


class StubAsyncClient:
    def __init__(self, responses: List[StubResponse]) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    async def post(self, path: str, **kwargs: Any) -> StubResponse:
        self.calls.append({"path": path, **kwargs})
        if not self.responses:
            raise AssertionError("StubAsyncClient has no response left.")
        return self.responses.pop(0)

    async def aclose(self) -> None:
        return None


class VLLMClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_length_limited_reasoning_response_has_safe_metadata(self) -> None:
        response = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": None, "role": "assistant"},
                }
            ],
            "usage": {"completion_tokens": 2048},
        }

        with self.assertRaises(LLMResponseError) as raised:
            extract_message_text(response)

        error = raised.exception
        self.assertEqual(error.finish_reason, "length")
        self.assertEqual(error.completion_tokens, 2048)
        self.assertEqual(error.message_fields, ["content", "role"])
        self.assertIn("incomplete JSON prefix", str(error))

        error_row = build_generation_error(error, unit_index=2, unit_id="unit-2")
        self.assertEqual(error_row["finish_reason"], "length")
        self.assertEqual(error_row["completion_tokens"], 2048)
        self.assertEqual(error_row["message_fields"], ["content", "role"])

    async def test_length_limited_partial_json_is_rejected_before_parsing(self) -> None:
        response = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": '{"answer":"unterminated',
                        "role": "assistant",
                    },
                }
            ],
            "usage": {"completion_tokens": 4096},
        }

        with self.assertRaises(LLMResponseError) as raised:
            extract_message_text(response)

        self.assertEqual(raised.exception.finish_reason, "length")
        self.assertEqual(raised.exception.completion_tokens, 4096)
        self.assertIn("incomplete JSON prefix", str(raised.exception))
        self.assertNotIn("unterminated", str(raised.exception))

    async def test_reasoning_only_response_has_actionable_error(self) -> None:
        response = {
            "choices": [
                {"message": {"content": None, "reasoning_content": "private"}}
            ]
        }

        with self.assertRaisesRegex(ValueError, "exhausted max_tokens"):
            extract_message_text(response)

    async def test_400_preserves_vllm_detail_and_does_not_retry(self) -> None:
        transport = StubAsyncClient(
            [
                StubResponse(
                    400,
                    {
                        "error": {
                            "message": "The model `gemma4` does not exist.",
                            "type": "NotFoundError",
                            "param": "model",
                            "code": 400,
                        }
                    },
                )
            ]
        )
        client = make_client(transport)

        with self.assertRaises(VLLMHTTPError) as raised:
            await client.chat_text(
                messages=[{"role": "user", "content": "private prompt"}],
                response_format={"type": "json_schema", "json_schema": {}},
                retry_config=no_delay_retry(max_attempts=3),
            )

        error = raised.exception
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(error.status_code, 400)
        self.assertFalse(error.retryable)
        self.assertIn("The model `gemma4` does not exist.", str(error))
        self.assertEqual(error.request_summary["model"], "gemma4")
        self.assertEqual(error.request_summary["response_format"], "json_schema")
        self.assertNotIn("private prompt", str(error))

        error_row = build_generation_error(
            error,
            unit_index=7,
            unit_id="unit-7",
        )
        self.assertEqual(error_row["http_status"], 400)
        self.assertEqual(error_row["vllm_error"], error.response_detail)
        self.assertFalse(error_row["retryable"])
        self.assertEqual(error_row["request_summary"]["model"], "gemma4")

    async def test_transient_503_retries_and_returns_success(self) -> None:
        transport = StubAsyncClient(
            [
                StubResponse(503, "temporarily unavailable"),
                StubResponse(
                    200,
                    {"choices": [{"message": {"content": "ready"}}]},
                    reason_phrase="OK",
                ),
            ]
        )
        client = make_client(transport)

        text = await client.chat_text(
            messages=[{"role": "user", "content": "hello"}],
            retry_config=no_delay_retry(max_attempts=2),
        )

        self.assertEqual(text, "ready")
        self.assertEqual(len(transport.calls), 2)


def make_client(transport: StubAsyncClient) -> VLLMClient:
    client = VLLMClient(
        LLMClientConfig(
            model="gemma4",
            extra_body={"reasoning_effort": "medium", "include_reasoning": False},
        )
    )
    client._async_client = transport
    return client


def no_delay_retry(max_attempts: int) -> RetryConfig:
    return RetryConfig(
        max_attempts=max_attempts,
        initial_delay_s=0,
        max_delay_s=0,
        backoff_multiplier=1,
        jitter_s=0,
    )


if __name__ == "__main__":
    unittest.main()
