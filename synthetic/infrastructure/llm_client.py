"""OpenAI-compatible client for vLLM-served local models."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .retry import RetryConfig, retry_async


ChatMessage = Dict[str, Any]


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
                response.raise_for_status()
                return response.json()

        return await retry_async(operation, retry_config)

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
        raise ValueError("LLM response choice does not contain message.content.")
    return str(content)

