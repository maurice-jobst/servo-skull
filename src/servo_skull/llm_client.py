"""Unified LLM client with primary -> fallback provider chain."""
import logging
from typing import TYPE_CHECKING, Optional

import httpx

from servo_skull.llm_providers import BaseLLMProvider

if TYPE_CHECKING:
    from servo_skull.pii_vault import PIIVault

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM client managing primary and fallback providers.

    On timeout or connection error from primary provider, automatically
    switches to fallback provider. Tracks fallback usage for smart routing.
    """

    def __init__(
        self,
        primary: BaseLLMProvider,
        fallback: Optional[BaseLLMProvider] = None
    ):
        """
        Initialize LLMClient with provider chain.

        Args:
            primary: Primary LLM provider (Ollama, OpenAI, Anthropic, etc.)
            fallback: Fallback provider (optional). Used if primary fails.
        """
        self.primary = primary
        self.fallback = fallback
        self.fallback_triggered = False
        self.fallback_count = 0

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout_override: Optional[float] = None,
        document_size_bytes: Optional[int] = None,
        pii_vault: Optional["PIIVault"] = None,
    ) -> str:
        """
        Call LLM with automatic fallback on timeout/connection error.

        When *pii_vault* is supplied the *user_prompt* is pseudonymized before
        dispatch and the response is rehydrated before being returned. The
        caller retains ownership of the vault and must call
        :meth:`~servo_skull.pii_vault.PIIVault.clear` after the run.

        Args:
            system_prompt: System prompt for LLM
            user_prompt: User message
            timeout_override: Override default timeout
            document_size_bytes: Size of source document (for logging context)
            pii_vault: Optional vault used for pseudonymization/rehydration

        Returns:
            LLM response as string (rehydrated if vault was supplied)

        Raises:
            httpx.TimeoutException: If both primary and fallback time out
            httpx.ConnectError: If both primary and fallback fail to connect
            httpx.HTTPError: If both primary and fallback fail with HTTP error
        """
        # --- PII pseudonymization ---
        safe_prompt = user_prompt
        if pii_vault is not None and len(pii_vault) > 0:
            safe_prompt = pii_vault.pseudonymize(user_prompt)
            logger.info(
                "LLMClient: pseudonymized prompt (%d entities replaced).",
                len(pii_vault),
            )

        try:
            logger.debug(
                f"Calling primary provider: {self.primary.__class__.__name__}"
            )
            response = self.primary.chat(
                system_prompt=system_prompt,
                user_prompt=safe_prompt,
                timeout_override=timeout_override
            )
            # --- PII rehydration ---
            if pii_vault is not None:
                response = pii_vault.rehydrate(response)
            return response
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            logger.warning(
                f"Primary provider failed ({type(e).__name__}), "
                f"triggering fallback. Document size: {document_size_bytes} bytes"
            )

            if not self.fallback:
                logger.error("No fallback provider configured, re-raising error")
                raise

            self.fallback_triggered = True
            self.fallback_count += 1

            try:
                logger.debug(
                    f"Calling fallback provider: {self.fallback.__class__.__name__}"
                )
                response = self.fallback.chat(
                    system_prompt=system_prompt,
                    user_prompt=safe_prompt,
                    timeout_override=timeout_override
                )
                # --- PII rehydration ---
                if pii_vault is not None:
                    response = pii_vault.rehydrate(response)
                logger.info(
                    f"Fallback succeeded (fallback count: {self.fallback_count})"
                )
                return response
            except Exception as fallback_error:
                logger.error(
                    f"Fallback provider also failed: "
                    f"{type(fallback_error).__name__}: {fallback_error}"
                )
                raise

    def should_prefer_fallback(self) -> bool:
        """
        Determine if fallback should be preferred for future calls.

        Returns:
            True if fallback should be preferred (>=2 fallbacks in session)
        """
        return self.fallback_count >= 2

    def choose_provider(self, document_size_bytes: Optional[int] = None) -> BaseLLMProvider:
        """
        Choose which provider to use based on routing heuristics.

        Uses document size and fallback frequency to decide:
        - If document_size_bytes > threshold → prefer fallback
        - If should_prefer_fallback() → prefer fallback
        - Else → use primary

        Args:
            document_size_bytes: Size of source document in bytes (optional)

        Returns:
            Selected provider: primary or fallback
        """
        from servo_skull.document_router import should_prefer_fallback_for_document

        # Check document size threshold first
        if document_size_bytes:
            config = {"document_size_warning_mb": 1.0}
            if should_prefer_fallback_for_document(document_size_bytes, config):
                if self.fallback:
                    return self.fallback

        # Check fallback frequency
        if self.should_prefer_fallback():
            if self.fallback:
                return self.fallback

        return self.primary

    def close(self) -> None:
        """Close and cleanup all providers."""
        self.primary.close()
        if self.fallback:
            self.fallback.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
