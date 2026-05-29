"""MLX inference backend for Apple Silicon native LLM execution.

Implements :class:`~servo_skull.llm_providers.BaseLLMProvider` using the
``mlx-lm`` Python API. Models are loaded lazily on the first :meth:`chat`
call and cached for the lifetime of the provider instance, so subsequent
calls within the same process incur no model-load overhead.

Quantization is handled by ``mlx-lm`` at load-time; specify a pre-quantized
HuggingFace repository (e.g. ``mlx-community/Qwen2.5-7B-Instruct-4bit``) or
any base model — ``mlx-lm`` will auto-quantize on the fly.

This module is **optional**: it requires ``mlx-lm`` which is mac-only.
Import guards ensure the rest of the pipeline degrades gracefully when
``mlx-lm`` is not installed.
"""
from __future__ import annotations

import logging
from typing import Any

from servo_skull.llm_providers import BaseLLMProvider

logger = logging.getLogger(__name__)

# Default chat template applied when the tokenizer does not supply one
_DEFAULT_SYSTEM_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


class MLXProvider(BaseLLMProvider):
    """Apple Silicon native inference via ``mlx-lm``.

    Args:
        model: HuggingFace model ID or local path. Pre-quantized models from
            ``mlx-community`` are recommended for zero-overhead startup.
        max_tokens: Maximum tokens to generate per response.
        temperature: Sampling temperature (0.0 = deterministic greedy decode).
        timeout: Ignored for MLX (inference is synchronous and local). Kept
            for interface compatibility with :class:`BaseLLMProvider`.
        verbose: If ``True``, ``mlx-lm`` logs token-by-token output to stdout.
    """

    def __init__(
        self,
        model: str = "mlx-community/Qwen2.5-7B-Instruct-4bit",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        timeout: float = 300.0,
        verbose: bool = False,
    ) -> None:
        super().__init__(timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose

        self._mlx_model: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the MLX model and tokenizer (once per process)."""
        if self._mlx_model is not None:
            return

        try:
            from mlx_lm import load as mlx_load  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "mlx-lm is required for the MLX backend. "
                "Install it with: uv pip install 'mlx-lm>=0.21.0'\n"
                "Note: mlx-lm is Apple Silicon / macOS only."
            ) from exc

        logger.info("MLXProvider: loading model '%s' …", self.model)
        self._mlx_model, self._tokenizer = mlx_load(self.model)
        logger.info("MLXProvider: model loaded.")

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Assemble a chat prompt using the tokenizer's template if available."""
        if self._tokenizer is None:
            return _DEFAULT_SYSTEM_TEMPLATE.format(
                system=system_prompt, user=user_prompt
            )

        # Use the tokenizer's apply_chat_template if present
        apply_fn = getattr(self._tokenizer, "apply_chat_template", None)
        if apply_fn is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            try:
                return apply_fn(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass  # fall through to default template

        return _DEFAULT_SYSTEM_TEMPLATE.format(
            system=system_prompt, user=user_prompt
        )

    # ------------------------------------------------------------------
    # BaseLLMProvider interface
    # ------------------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_override: float | None = None,
    ) -> str:
        """Run inference synchronously on Apple Silicon.

        Args:
            system_prompt: System instruction for the model.
            user_prompt: User message / document content.
            timeout_override: Not used for local MLX inference.

        Returns:
            Generated response string.

        Raises:
            ImportError: If ``mlx-lm`` is not installed.
            RuntimeError: If the model produces an empty response.
        """
        self._ensure_loaded()

        from mlx_lm import generate as mlx_generate  # type: ignore[import]

        prompt = self._build_prompt(system_prompt, user_prompt)

        logger.debug(
            "MLXProvider: generating (model=%s, max_tokens=%d, temp=%.2f)…",
            self.model,
            self.max_tokens,
            self.temperature,
        )

        response: str = mlx_generate(
            self._mlx_model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temp=self.temperature,
            verbose=self.verbose,
        )

        if not response or not response.strip():
            raise RuntimeError(
                f"MLXProvider: empty response from model '{self.model}'"
            )

        logger.debug("MLXProvider: generated %d chars.", len(response))
        return response.strip()

    def close(self) -> None:
        """Release model references.

        MLX manages its own memory pools; setting references to ``None``
        allows Python's garbage collector to free the weights.
        """
        self._mlx_model = None
        self._tokenizer = None
        logger.debug("MLXProvider: model references released.")

    def __repr__(self) -> str:
        loaded = self._mlx_model is not None
        return (
            f"MLXProvider(model='{self.model}', "
            f"max_tokens={self.max_tokens}, loaded={loaded})"
        )
