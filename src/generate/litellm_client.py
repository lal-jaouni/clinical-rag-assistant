"""LiteLLM-backed LLM client: unified interface for Ollama, Anthropic, OpenAI, Groq, etc.

The model string (e.g., ``"ollama/biomistral:7b"``, ``"anthropic/claude-3-5-sonnet-20241022"``,
``"openai/gpt-4o-mini"``) tells LiteLLM which provider to route to. API keys come
from environment variables (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.).

Usage:
    client = LLMClient.from_config(load_config().model.llm)
    result = client.complete(system_prompt="...", user_prompt="...")
    print(result.text, result.usage_tokens)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import litellm


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    raw: Any  # full LiteLLM ModelResponse, kept for debugging


class LLMClient:
    """Thin wrapper around LiteLLM.completion with sensible clinical defaults."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 500,
        top_p: float = 0.9,
        provider_kwargs: dict[str, Any] | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.provider_kwargs = provider_kwargs or {}
        self.provider = model.split("/", 1)[0] if "/" in model else "unknown"

    @classmethod
    def from_config(cls, llm_cfg) -> "LLMClient":
        return cls(
            model=llm_cfg.model,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
            top_p=llm_cfg.top_p,
            provider_kwargs=llm_cfg.provider_kwargs,
        )

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        start = time.perf_counter()
        resp = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            **self.provider_kwargs,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        choice = resp.choices[0]
        usage = resp.usage if hasattr(resp, "usage") else None
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        return LLMResponse(
            text=choice.message.content or "",
            model=self.model,
            provider=self.provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            raw=resp,
        )

    def ping(self) -> bool:
        """Smoke test: ensure the configured model responds."""
        try:
            resp = self.complete("You are a test.", "Reply with the word 'ok' only.")
            return "ok" in resp.text.lower()
        except Exception:
            return False
