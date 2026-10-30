"""Async OpenAI-compatible chat client with retries and budget control."""

import asyncio
import dataclasses
import logging
import os
import random
from typing import Any


class BudgetExceededError(RuntimeError):
    """Raised when a reservation would exceed the sweep budget."""


class Budget:
    """Upfront cost reservation (protocol: no unbounded API spend).

    Each call reserves its worst-case cost (prompt estimate + max_tokens)
    before it is sent and settles to the actual usage afterwards, so
    concurrent calls can never overshoot the cap.
    """

    def __init__(self, limit_usd: float):
        self.limit = float(limit_usd)
        self.spent = 0.0
        self.reserved = 0.0
        self._lock = asyncio.Lock()

    async def reserve(self, amount: float) -> None:
        async with self._lock:
            if self.spent + self.reserved + amount > self.limit:
                raise BudgetExceededError(
                    f"budget {self.limit:.2f} USD exhausted (spent "
                    f"{self.spent:.2f}, reserved {self.reserved:.2f})"
                )
            self.reserved += amount

    async def settle(self, reserved: float, actual: float) -> None:
        async with self._lock:
            self.reserved -= reserved
            self.spent += actual


@dataclasses.dataclass
class ChatResult:
    text: str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    raw_finish_reason: str | None = None


class ChatClient:
    """Minimal async client for /chat/completions."""

    def __init__(
        self,
        base_url: str,
        api_key_env: str | None,
        timeout: float = 300.0,
        max_retries: int = 6,
    ):
        import httpx  # pylint: disable=import-outside-toplevel

        key = os.environ.get(api_key_env or "", "") or "EMPTY"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
        self._retries = max_retries
        self._httpx = httpx

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        seed: int = 0,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResult:
        body = {
            "model": model,
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        body.update(extra_body or {})
        delay = 2.0
        for attempt in range(self._retries + 1):
            try:
                response = await self._client.post(
                    "/chat/completions", json=body
                )
                if response.status_code in (429, 500, 502, 503, 504):
                    raise self._httpx.HTTPStatusError(
                        f"HTTP {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                if response.status_code >= 400:
                    return ChatResult(
                        None,
                        error=f"HTTP {response.status_code}"
                        f": {response.text[:300]}",
                    )
                payload = response.json()
                choice = (payload.get("choices") or [{}])[0]
                usage = payload.get("usage") or {}
                return ChatResult(
                    text=(choice.get("message") or {}).get("content"),
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    raw_finish_reason=choice.get("finish_reason"),
                )
            except (self._httpx.HTTPError, ValueError) as exc:
                if attempt == self._retries:
                    return ChatResult(
                        None, error=f"{type(exc).__name__}: {exc}"
                    )
                await asyncio.sleep(delay + random.random())
                delay = min(delay * 2, 60)
        return ChatResult(None, error="unreachable")


def sync_chat_fn(endpoint_cfg: dict[str, Any] | None):
    """Returns prompt -> text using a blocking client (query synthesis)."""
    if not endpoint_cfg:
        raise ValueError(
            "llm.query_endpoint must be configured for "
            "cohort.query_generator=llm"
        )

    def chat(prompt: str) -> str:
        async def _once():
            client = ChatClient(
                endpoint_cfg["base_url"], endpoint_cfg.get("api_key_env")
            )
            try:
                result = await client.chat(
                    endpoint_cfg["model"],
                    prompt,
                    int(endpoint_cfg.get("max_tokens", 512)),
                    extra_body=endpoint_cfg.get("extra_body"),
                )
            finally:
                await client.close()
            if result.error:
                logging.warning("query LLM error: %s", result.error)
            return result.text or ""

        return asyncio.run(_once())

    return chat
