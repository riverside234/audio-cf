"""OpenAI-compatible client for vLLM-served local models."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .retry import RetryConfig, retry_async


ChatMessage = Dict[str, Any]
MAX_ERROR_DETAIL_CHARS = 4000


class VLLMHTTPError(RuntimeError):
    """Actionable error returned by an OpenAI-compatible vLLM endpoint."""

    def __init__(
        self,
        *,
        status_code: int,
        reason_phrase: str,
        url: str,
        response_detail: str,
        request_summary: Mapping[str, Any],
    ) -> None:
        self.status_code = status_code
        self.reason_phrase = reason_phrase
        self.url = url
        self.response_detail = response_detail
        self.request_summary = dict(request_summary)
        self.retryable = status_code in {408, 425, 429} or status_code >= 500
        summary_text = ", ".join(
            f"{key}={value!r}" for key, value in self.request_summary.items()
        )
        super().__init__(
            f"vLLM request failed with HTTP {status_code} {reason_phrase} for "
            f"{url}: {response_detail}. Request summary: {summary_text}"
        )


class LLMResponseError(ValueError):
    """A successful HTTP response that lacks usable assistant content."""

    def __init__(
        self,
        message: str,
        *,
        finish_reason: Any,
        completion_tokens: Any,
        message_fields: Sequence[str],
    ) -> None:
        self.finish_reason = finish_reason
        self.completion_tokens = completion_tokens
        self.message_fields = list(message_fields)
        details = (
            f"finish_reason={finish_reason!r}, "
            f"completion_tokens={completion_tokens!r}, "
            f"message_fields={self.message_fields!r}"
        )
        super().__init__(f"{message} ({details})")


@dataclass
class LLMClientConfig:
    """Runtime settings for an OpenAI-compatible vLLM endpoint."""

    base_url: str = "http://localhost:8000/v1"
    model: str = ""
    api_key: str = "EMPTY"
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 512
    timeout_s: float = 120.0
    client_max_inflight_requests: int = 16
    response_format: Optional[Dict[str, Any]] = None
    extra_body: Dict[str, Any] = field(default_factory=dict)
    extra_headers: Dict[str, str] = field(default_factory=dict)
    verify_ssl: bool = True


class VLLMClient:
    """Thin async wrapper around vLLM's OpenAI-compatible chat endpoint."""

    def __init__(self, config: LLMClientConfig):
        if not config.model:
            raise ValueError("LLMClientConfig.model must be set.")
        self.config = config
        self._async_client: Any = None
        self._semaphore = asyncio.Semaphore(config.client_max_inflight_requests)

    async def close(self) -> None:
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    async def __aenter__(self) -> "VLLMClient":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close()

    async def _client(self) -> Any:
        if self._async_client is None:
            try:
                import httpx  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "httpx is required for VLLMClient. Install it with `pip install httpx`."
                ) from exc

            self._async_client = httpx.AsyncClient(
                base_url=self.config.base_url.rstrip("/"),
                timeout=self.config.timeout_s,
                verify=self.config.verify_ssl,
            )
        return self._async_client

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.config.extra_headers)
        return headers

    def _payload(
        self,
        messages: Sequence[ChatMessage],
        temperature: Optional[float],
        top_p: Optional[float],
        max_tokens: Optional[int],
        response_format: Optional[Dict[str, Any]],
        extra_body: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": self.config.temperature if temperature is None else temperature,
            "top_p": self.config.top_p if top_p is None else top_p,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
        }

        active_response_format = (
            self.config.response_format if response_format is None else response_format
        )
        if active_response_format:
            payload["response_format"] = active_response_format

        payload.update(self.config.extra_body)
        if extra_body:
            payload.update(dict(extra_body))
        return payload

    async def chat_completion(
        self,
        messages: Sequence[ChatMessage],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        retry_config: Optional[RetryConfig] = None,
    ) -> Dict[str, Any]:
        """Call `/chat/completions` and return the raw JSON response."""

        payload = self._payload(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_format=response_format,
            extra_body=extra_body,
        )

        async def operation() -> Dict[str, Any]:
            async with self._semaphore:
                client = await self._client()
                response = await client.post(
                    "/chat/completions",
                    json=payload,
                    headers=self._headers(),
                )
                if not 200 <= response.status_code < 300:
                    raise VLLMHTTPError(
                        status_code=response.status_code,
                        reason_phrase=response.reason_phrase,
                        url=str(response.url),
                        response_detail=_response_error_detail(response),
                        request_summary=_request_summary(payload),
                    )
                return response.json()

        return await retry_async(
            operation,
            retry_config,
            should_retry=lambda error: not isinstance(error, VLLMHTTPError)
            or error.retryable,
        )

    async def chat_text(
        self,
        messages: Sequence[ChatMessage],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        retry_config: Optional[RetryConfig] = None,
    ) -> str:
        """Call chat completion and return the first message content."""

        response = await self.chat_completion(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_format=response_format,
            extra_body=extra_body,
            retry_config=retry_config,
        )
        return extract_message_text(response)

    async def batch_chat_text(
        self,
        message_batches: Sequence[Sequence[ChatMessage]],
        retry_config: Optional[RetryConfig] = None,
    ) -> List[str]:
        """Run multiple chat requests concurrently, preserving input order."""

        tasks = [
            self.chat_text(messages=messages, retry_config=retry_config)
            for messages in message_batches
        ]
        return list(await asyncio.gather(*tasks))


def extract_message_text(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not choices:
        raise ValueError("LLM response does not contain choices.")
    first = choices[0]
    message = first.get("message") or {}
    content = message.get("content")
    if content is None:
        finish_reason = first.get("finish_reason")
        usage = response.get("usage") or {}
        completion_tokens = (
            usage.get("completion_tokens") if isinstance(usage, Mapping) else None
        )
        message_fields = sorted(str(field) for field in message.keys())
        has_reasoning = any(
            message.get(field) not in (None, "")
            for field in ("reasoning", "reasoning_content")
        )
        if finish_reason == "length":
            raise LLMResponseError(
                "LLM exhausted the completion token limit before producing final "
                "message.content. For reasoning models, set thinking_token_budget "
                "below max_tokens or increase the completion/context limits.",
                finish_reason=finish_reason,
                completion_tokens=completion_tokens,
                message_fields=message_fields,
            )
        if has_reasoning:
            raise LLMResponseError(
                "LLM response contains reasoning but no final message.content. "
                "The model may have exhausted max_tokens before leaving its "
                "thinking block, or the configured reasoning parser may not match "
                "the served model.",
                finish_reason=finish_reason,
                completion_tokens=completion_tokens,
                message_fields=message_fields,
            )
        raise LLMResponseError(
            "LLM response choice does not contain message.content. Check the "
            "served model's reasoning parser and completion token budget.",
            finish_reason=finish_reason,
            completion_tokens=completion_tokens,
            message_fields=message_fields,
        )
    return str(content)


def _response_error_detail(response: Any) -> str:
    try:
        payload = response.json()
    except (ValueError, TypeError):
        payload = None

    detail = ""
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping):
            parts = []
            for key in ("message", "type", "param", "code"):
                value = error.get(key)
                if value is not None and value != "":
                    parts.append(f"{key}={value!r}")
            detail = ", ".join(parts)
        payload_detail = payload.get("detail")
        if not detail and payload_detail is not None and payload_detail != "":
            detail = _json_text(payload_detail)
        if not detail:
            detail = _json_text(payload)
    elif payload is not None:
        detail = _json_text(payload)

    if not detail:
        detail = str(getattr(response, "text", "")).strip()
    if not detail:
        detail = "vLLM returned an empty error response"
    if len(detail) > MAX_ERROR_DETAIL_CHARS:
        return detail[:MAX_ERROR_DETAIL_CHARS] + "... [truncated]"
    return detail


def _request_summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    standard_fields = {
        "model",
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "response_format",
    }
    response_format = payload.get("response_format")
    response_format_type = (
        response_format.get("type") if isinstance(response_format, Mapping) else None
    )
    messages = payload.get("messages")
    return {
        "model": payload.get("model"),
        "message_count": len(messages) if isinstance(messages, Sequence) else None,
        "response_format": response_format_type,
        "extra_fields": sorted(set(payload) - standard_fields),
    }


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
