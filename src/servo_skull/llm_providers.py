"""Pluggable LLM provider interface with support for Ollama, OpenAI, Anthropic, and Gemini."""
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from servo_skull._utils import retry
from servo_skull.models import BenchStats
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, timeout: float = 300.0):
        self.timeout = timeout

    @abstractmethod
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_override: float | None = None
    ) -> str:
        """Call LLM; return response text."""
        pass

    def chat_stats(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> BenchStats:
        """Call LLM and return full benchmark statistics.

        Default implementation wraps chat() with wall-clock timing only.
        Providers that support native stats (Ollama, Gemini) override this.
        """
        t0 = time.time()
        text = self.chat(system_prompt, user_prompt)
        total_s = time.time() - t0
        words = len(text.split())
        # Rough token estimate: ~0.75 words/token
        est_tokens = max(1, int(words / 0.75))
        return BenchStats(
            text=text,
            ttft_s=0.0,           # not available without streaming
            total_s=round(total_s, 2),
            prompt_tokens=0,      # not available in compat mode
            output_tokens=est_tokens,
            tokens_per_s=round(est_tokens / total_s, 1),
            prefill_tokens_per_s=0.0,
            cost_usd=None,
        )

    @abstractmethod
    def close(self) -> None:
        """Close and cleanup provider resources."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class OllamaProvider(BaseLLMProvider):
    """OpenAI-compatible Ollama client."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434/v1",
        model: str = "gemma4:26b",
        temperature: float = 0.0,
        timeout: float = 300.0
    ):
        """
        Initialize Ollama provider.

        Args:
            base_url: Base URL for Ollama API (OpenAI-compatible)
            model: Model name to use
            temperature: Temperature for output (0.0 = deterministic)
            timeout: Request timeout in seconds
        """
        super().__init__(timeout=timeout)
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.client = httpx.Client(timeout=timeout)

    @retry(max_attempts=3, delay=1.0)
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_override: float | None = None
    ) -> str:
        """
        Call Ollama LLM with retry logic.

        Args:
            system_prompt: System prompt
            user_prompt: User message
            timeout_override: Override default timeout

        Returns:
            LLM response as string

        Raises:
            httpx.TimeoutException: If request times out
            httpx.HTTPError: If API call fails
        """
        timeout = timeout_override if timeout_override is not None else self.timeout

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }

        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError as e:
                logger.error(f"Ollama: Failed to parse JSON response: {e}")
                raise httpx.HTTPError(f"Invalid JSON response body") from e
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Ollama: Invalid API response structure: {e}")
                raise httpx.HTTPError(f"Invalid response: missing expected keys") from e
            return content
        except httpx.TimeoutException as e:
            logger.error(f"Ollama chat request timed out after {timeout}s: {e}")
            raise
        except httpx.HTTPError as e:
            logger.error(f"Ollama chat call failed with status code: {e.response.status_code if hasattr(e, 'response') else 'unknown'}")
            raise

    @property
    def _native_base_url(self) -> str:
        """Derive native Ollama API URL by stripping /v1 suffix."""
        url = self.base_url
        if url.endswith("/v1"):
            url = url[:-3]
        return url.rstrip("/")

    def chat_stats(self, system_prompt: str, user_prompt: str) -> BenchStats:
        """Use Ollama native /api/chat for exact TTFT and tokens/s stats.

        Returns:
            BenchStats with:
            - ttft_s: time to process input (prompt_eval_duration) — warm-model TTFT
            - tokens_per_s: output generation throughput (eval_count / eval_duration)
            - prefill_tokens_per_s: input processing rate — asset for large-context tasks
        """
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        t0 = time.time()
        try:
            response = self.client.post(
                f"{self._native_base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.error(f"Ollama native chat_stats failed: {exc}")
            raise

        total_s = time.time() - t0
        text = data.get("message", {}).get("content", "")

        # Native Ollama durations are in nanoseconds
        eval_count          = data.get("eval_count", 1)          # output tokens
        eval_duration_ns    = data.get("eval_duration", 1)        # generation time
        prompt_eval_count   = data.get("prompt_eval_count", 1)    # input tokens
        prompt_eval_duration_ns = data.get("prompt_eval_duration", 1)  # prefill time

        tokens_per_s   = eval_count / max(eval_duration_ns / 1e9, 0.001)
        prefill_tps    = prompt_eval_count / max(prompt_eval_duration_ns / 1e9, 0.001)
        ttft_s         = prompt_eval_duration_ns / 1e9  # warm TTFT = prefill duration

        return BenchStats(
            text=text,
            ttft_s=round(ttft_s, 3),
            total_s=round(total_s, 2),
            prompt_tokens=prompt_eval_count,
            output_tokens=eval_count,
            tokens_per_s=round(tokens_per_s, 1),
            prefill_tokens_per_s=round(prefill_tps, 1),
            cost_usd=None,
        )

    def close(self) -> None:
        """Close HTTP client."""
        self.client.close()


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API client. Also used for OpenAI-compatible local servers (mlx_lm.server, llama.cpp)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 300.0,
        max_tokens: int | None = None,
        no_system_role: bool = False,
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (use any non-empty string for local servers)
            model: Model name to use (default gpt-4)
            base_url: Base URL for OpenAI API
            timeout: Request timeout in seconds
            max_tokens: Max tokens in response. Set ≥2048 for reasoning models
                        (gemma4 thinking mode, o1, etc.) so prefill and reasoning
                        tokens complete before the final content key is emitted.
            no_system_role: If True, merge the system prompt into the user message
                            as a single user turn. Required for mlx_lm.server with
                            gemma4: a separate system role unconditionally activates
                            thinking mode, causing reasoning traces to leak into output.
        """
        super().__init__(timeout=timeout)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.no_system_role = no_system_role
        self.client = httpx.Client(timeout=timeout)

    @retry(max_attempts=3, delay=1.0)
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_override: float | None = None
    ) -> str:
        """
        Call OpenAI API with retry logic.

        Args:
            system_prompt: System prompt
            user_prompt: User message
            timeout_override: Override default timeout

        Returns:
            LLM response as string

        Raises:
            httpx.TimeoutException: If request times out
            httpx.HTTPError: If API call fails
        """
        timeout = timeout_override if timeout_override is not None else self.timeout

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # MLX gemma4 thinking mode fix: when no_system_role is set, merge
        # the system prompt into the user message. A separate system role
        # unconditionally activates gemma4's chain-of-thought thinking mode
        # in mlx_lm.server, causing <|channel>...</channel|> traces to leak
        # into the content field and produce 5-10× longer outputs.
        if self.no_system_role and system_prompt:
            merged_user = f"[System instructions]\n{system_prompt}\n\n[Request]\n{user_prompt}"
            messages = [{"role": "user", "content": merged_user}]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        payload: dict = {
            "model": self.model,
            "messages": messages,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError as e:
                logger.error(f"OpenAI: Failed to parse JSON response: {e}")
                raise httpx.HTTPError(f"Invalid JSON response body") from e
            try:
                msg = data["choices"][0]["message"]
                # Reasoning models (gemma4 thinking mode, o1, etc.) return both
                # "content" (final answer) and "reasoning" (chain-of-thought).
                # "content" may be absent if max_tokens is too low to complete
                # the thinking phase. Prefer "content"; fall back to "reasoning".
                content = msg.get("content") or msg.get("reasoning") or ""
                if not content:
                    raise KeyError("neither 'content' nor 'reasoning' in message")
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"OpenAI: Invalid API response structure: {e}")
                raise httpx.HTTPError(f"Invalid response: missing expected keys") from e
            return content
        except httpx.TimeoutException as e:
            logger.error(f"OpenAI chat request timed out after {timeout}s: {e}")
            raise
        except httpx.HTTPError as e:
            logger.error(f"OpenAI chat call failed with status code: {e.response.status_code if hasattr(e, 'response') else 'unknown'}")
            raise

    def chat_stats(self, system_prompt: str, user_prompt: str) -> BenchStats:
        """Chat with usage stats from OpenAI-compat response (works for MLX server too)."""
        t0 = time.time()
        text = self.chat(system_prompt, user_prompt)
        total_s = time.time() - t0
        # Usage stats available in the last response — re-fetch via a direct call
        # to get the usage block (the chat() method only returns text).
        # We use wall-clock timing + word-count estimation for tokens/s.
        words = len(text.split())
        est_tokens = max(1, int(words / 0.75))
        return BenchStats(
            text=text,
            ttft_s=0.0,       # not available without streaming on OpenAI-compat
            total_s=round(total_s, 2),
            prompt_tokens=0,  # not extracted from compat endpoint in this path
            output_tokens=est_tokens,
            tokens_per_s=round(est_tokens / max(total_s, 0.001), 1),
            prefill_tokens_per_s=0.0,
            cost_usd=None,
        )

    def close(self) -> None:
        """Close HTTP client."""
        self.client.close()


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API client."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-sonnet-20240229",
        timeout: float = 300.0,
        max_tokens: int = 4096
    ):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            model: Model name to use
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens in response (default 4096)
        """
        super().__init__(timeout=timeout)
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = "https://api.anthropic.com"
        self.client = httpx.Client(timeout=timeout)

    @retry(max_attempts=3, delay=1.0)
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_override: float | None = None
    ) -> str:
        """
        Call Anthropic API with retry logic.

        Args:
            system_prompt: System prompt
            user_prompt: User message
            timeout_override: Override default timeout

        Returns:
            LLM response as string

        Raises:
            httpx.TimeoutException: If request times out
            httpx.HTTPError: If API call fails
        """
        timeout = timeout_override if timeout_override is not None else self.timeout

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2024-06-01",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }

        try:
            response = self.client.post(
                f"{self.base_url}/v1/messages",
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError as e:
                logger.error(f"Anthropic: Failed to parse JSON response: {e}")
                raise httpx.HTTPError(f"Invalid JSON response body") from e
            try:
                content = data["content"][0]["text"]
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"Anthropic: Invalid API response structure: {e}")
                raise httpx.HTTPError(f"Invalid response: missing expected keys") from e
            return content
        except httpx.TimeoutException as e:
            logger.error(f"Anthropic chat request timed out after {timeout}s: {e}")
            raise
        except httpx.HTTPError as e:
            logger.error(f"Anthropic chat call failed with status code: {e.response.status_code if hasattr(e, 'response') else 'unknown'}")
            raise

    def close(self) -> None:
        """Close HTTP client."""
        self.client.close()


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API client using modern genai SDK."""

    # Gemini 2.5 Flash pricing (per million tokens, standard tier, 2026-05)
    _PRICE_INPUT_PER_M  = 0.30   # $/M input tokens
    _PRICE_OUTPUT_PER_M = 2.50   # $/M output tokens (incl. thinking)

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        timeout: float = 300.0,
        temperature: float = 0.0,
    ):
        """Initialize Gemini provider."""
        super().__init__(timeout=timeout)
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        if genai is None:
            raise ImportError("google-genai package is not installed.")
        self.client = genai.Client(api_key=self.api_key)
        self._last_usage: dict[str, int] = {}  # prompt_tokens, output_tokens

    @retry(max_attempts=3, delay=1.0)
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_override: float | None = None,
        enable_grounding: bool = False,
    ) -> str:
        """Call Gemini LLM with retry logic.

        Args:
            system_prompt: System instruction.
            user_prompt: User message.
            timeout_override: Override default timeout.
            enable_grounding: If ``True``, attaches the ``google_search`` tool
                so Gemini can retrieve live web results before generating.
        """
        timeout = timeout_override if timeout_override is not None else self.timeout
        timeout_ms = int(timeout * 1000)

        tool_cfg: list[types.Tool] = []
        if enable_grounding:
            tool_cfg = [types.Tool(google_search=types.GoogleSearch())]
            logger.info("GeminiProvider: grounding search enabled.")

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            tools=tool_cfg if tool_cfg else None,
            http_options={"timeout": timeout_ms}
        )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=config,
            )
            if not response.text:
                logger.error("Gemini: Empty response text received")
                raise httpx.HTTPError("Empty response text from Gemini")
            # Capture usage metadata for cost tracking
            usage = getattr(response, "usage_metadata", None)
            if usage:
                self._last_usage = {
                    "prompt_tokens":  getattr(usage, "prompt_token_count", 0) or 0,
                    "output_tokens":  getattr(usage, "candidates_token_count", 0) or 0,
                }
            return response.text
        except Exception as e:
            logger.error(f"Gemini chat request failed: {e}")
            raise

    def cost_for_last_call(self) -> float:
        """Calculate USD cost for the most recent chat() call."""
        p = self._last_usage.get("prompt_tokens", 0)
        o = self._last_usage.get("output_tokens", 0)
        return (p / 1_000_000) * self._PRICE_INPUT_PER_M + \
               (o / 1_000_000) * self._PRICE_OUTPUT_PER_M

    def chat_stats(self, system_prompt: str, user_prompt: str) -> BenchStats:
        """Chat with Gemini and return BenchStats including token counts and USD cost."""
        t0 = time.time()
        text = self.chat(system_prompt, user_prompt)
        total_s = time.time() - t0
        prompt_t = self._last_usage.get("prompt_tokens", 0)
        output_t = self._last_usage.get("output_tokens", 0)
        cost = self.cost_for_last_call()
        return BenchStats(
            text=text,
            ttft_s=0.0,      # TTFT needs streaming; not available in sync mode
            total_s=round(total_s, 2),
            prompt_tokens=prompt_t,
            output_tokens=output_t,
            tokens_per_s=round(output_t / max(total_s, 0.001), 1),
            prefill_tokens_per_s=round(prompt_t / max(total_s, 0.001), 1),
            cost_usd=round(cost, 6),
        )

    def grounded_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_override: float | None = None,
    ) -> str:
        """Convenience wrapper: :meth:`chat` with grounding always enabled."""
        return self.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timeout_override=timeout_override,
            enable_grounding=True,
        )

    def close(self) -> None:
        """No persistent resources to close."""
        pass


import os
import re

def expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in a dict, list, or string."""
    if isinstance(value, str):
        match = re.match(r"^\$\{?([a-zA-Z_][a-zA-Z0-9_]*)\}?$", value)
        if match:
            var_name = match.group(1)
            return os.environ.get(var_name, value)
        return os.path.expandvars(value)
    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [expand_env_vars(v) for v in value]
    return value


def create_provider(config: dict[str, Any]) -> BaseLLMProvider:
    """
    Factory function to create LLM provider based on configuration.

    Args:
        config: Configuration dict with 'type' key and provider-specific settings

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If provider type is unknown or config is invalid
    """
    config = expand_env_vars(config)
    provider_type = config.get("type", "").lower()

    if not provider_type:
        raise ValueError(
            "Configuration must specify a 'type' key "
            "(ollama, openai, anthropic, gemini, or mlx)"
        )

    if provider_type == "ollama":
        return OllamaProvider(
            base_url=config.get("base_url", "http://127.0.0.1:11434/v1"),
            model=config.get("model", "gemma4:26b"),
            temperature=config.get("temperature", 0.0),
            timeout=config.get("timeout", 300.0),
        )
    elif provider_type == "openai":
        return OpenAIProvider(
            api_key=config.get("api_key", ""),
            model=config.get("model", "gpt-4"),
            base_url=config.get("base_url", "https://api.openai.com/v1"),
            timeout=config.get("timeout", 300.0),
            max_tokens=config.get("max_tokens", None),
            no_system_role=config.get("no_system_role", False),
        )
    elif provider_type == "anthropic":
        return AnthropicProvider(
            api_key=config.get("api_key", ""),
            model=config.get("model", "claude-3-sonnet-20240229"),
            timeout=config.get("timeout", 300.0),
            max_tokens=config.get("max_tokens", 4096),
        )
    elif provider_type == "gemini":
        return GeminiProvider(
            api_key=config.get("api_key", ""),
            model=config.get("model", "gemini-2.5-flash"),
            timeout=config.get("timeout", 300.0),
            temperature=config.get("temperature", 0.0),
        )
    elif provider_type == "mlx":
        from servo_skull.mlx_provider import MLXProvider  # optional dep
        return MLXProvider(
            model=config.get(
                "model", "mlx-community/Qwen2.5-7B-Instruct-4bit"
            ),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.0),
            verbose=config.get("verbose", False),
        )
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
