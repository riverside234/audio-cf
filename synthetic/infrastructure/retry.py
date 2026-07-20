"""Retry helpers for transient infrastructure failures."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Tuple, Type, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RetryConfig:
    """Small exponential-backoff retry policy."""

    max_attempts: int = 3
    initial_delay_s: float = 1.0
    max_delay_s: float = 30.0
    backoff_multiplier: float = 2.0
    jitter_s: float = 0.25

    def delay_for_attempt(self, attempt_index: int) -> float:
        base = self.initial_delay_s * (self.backoff_multiplier ** max(0, attempt_index - 1))
        delay = min(base, self.max_delay_s)
        if self.jitter_s > 0:
            delay += random.uniform(0, self.jitter_s)
        return delay


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    config: Optional[RetryConfig] = None,
    retry_exceptions: Tuple[Type[BaseException], ...] = (Exception,),
) -> T:
    """Run an async operation with retry/backoff."""

    policy = config or RetryConfig()
    last_error: Optional[BaseException] = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await operation()
        except retry_exceptions as exc:
            last_error = exc
            if attempt >= policy.max_attempts:
                break
            await asyncio.sleep(policy.delay_for_attempt(attempt))

    assert last_error is not None
    raise last_error


def retry_sync(
    operation: Callable[[], T],
    config: Optional[RetryConfig] = None,
    retry_exceptions: Tuple[Type[BaseException], ...] = (Exception,),
) -> T:
    """Run a sync operation with retry/backoff."""

    policy = config or RetryConfig()
    last_error: Optional[BaseException] = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation()
        except retry_exceptions as exc:
            last_error = exc
            if attempt >= policy.max_attempts:
                break
            time.sleep(policy.delay_for_attempt(attempt))

    assert last_error is not None
    raise last_error

