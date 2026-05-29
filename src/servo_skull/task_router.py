"""Task-aware routing: PII gate + Gemini Flash classifier.

Two-stage decision process for every document ingestion:

1. **PII gate** (local, zero-cloud): scan the ``DocumentExtract`` with
   :class:`~servo_skull.pii_vault.PIIVault` and produce a
   :class:`~servo_skull.models.PIIClassification`. If PII is detected the
   document is forced to the local backend without touching cloud APIs.

2. **Task classifier** (Gemini Flash, cheap): send *only* document metadata
   and the first 512 tokens of content to ``gemini-2.5-flash-latest`` with
   ``response_mime_type=application/json`` to receive a structured
   :class:`~servo_skull.models.TaskRoute`.

The resulting ``TaskRoute`` drives downstream provider selection in
:class:`~servo_skull.llm_client.LLMClient`.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from servo_skull._utils import retry
from servo_skull.models import DocumentExtract, PIIClassification, TaskRoute
from servo_skull.pii_vault import PIIVault

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal routing table
# ---------------------------------------------------------------------------
# Maps (task_type, pii_present) → (backend, model_tier, requires_grounding)
_ROUTING_TABLE: dict[tuple[str, bool], tuple[str, str, bool]] = {
    ("extraction",     True):  ("mlx",    "local",  False),
    ("extraction",     False): ("mlx",    "local",  False),
    ("rule_synthesis", True):  ("mlx",    "local",  False),
    ("rule_synthesis", False): ("gemini", "reason", False),
    ("gap_analysis",   True):  ("mlx",    "local",  False),
    ("gap_analysis",   False): ("gemini", "reason", True),
    ("summary",        True):  ("mlx",    "local",  False),
    ("summary",        False): ("gemini", "fast",   True),
    ("security_check", True):  ("ollama", "local",  False),
    ("security_check", False): ("gemini", "fast",   False),
    ("deep_research",  True):  ("ollama", "local",  False),
    ("deep_research",  False): ("gemini", "reason", True),
}

# Gemini Flash classification prompt
_CLASSIFIER_SYSTEM = (
    "You are a document-task classifier for an ingestion pipeline. "
    "Analyse the document metadata and excerpt, then return ONLY a single "
    "JSON object with these fields:\n"
    "  task_type: one of extraction|rule_synthesis|gap_analysis|summary|"
    "security_check|deep_research\n"
    "  complexity: low|medium|high\n"
    "  requires_grounding: true|false\n"
    "  estimated_tokens: integer\n"
    "No extra text, no markdown fences. Raw JSON only."
)


def _build_classifier_prompt(extract: DocumentExtract, max_chars: int = 2048) -> str:
    """Build the compact prompt sent to Gemini Flash for task classification."""
    excerpt = extract.extracted_text[:max_chars]
    return (
        f"filename: {extract.original_filename}\n"
        f"doc_type: {extract.document_type}\n"
        f"confidence: {extract.confidence}\n"
        f"char_count: {len(extract.extracted_text)}\n\n"
        f"--- excerpt (first {max_chars} chars) ---\n"
        f"{excerpt}"
    )


def _fallback_route(pii_present: bool) -> TaskRoute:
    """Return a safe default route when the classifier cannot be reached."""
    backend = "ollama" if pii_present else "gemini"
    return TaskRoute(
        task_type="gap_analysis",
        complexity="medium",
        requires_grounding=not pii_present,
        pii_present=pii_present,
        backend=backend,
        model_tier="local" if pii_present else "fast",
        estimated_tokens=0,
    )


def _apply_routing_table(
    classifier_result: dict[str, Any],
    pii_present: bool,
) -> TaskRoute:
    """Merge classifier output with the routing table to produce a ``TaskRoute``."""
    task_type = classifier_result.get("task_type", "gap_analysis")
    complexity = classifier_result.get("complexity", "medium")
    estimated_tokens = int(classifier_result.get("estimated_tokens", 0))

    # Look up routing table; fall back to gap_analysis if unknown task_type
    key = (task_type, pii_present)
    if key not in _ROUTING_TABLE:
        key = ("gap_analysis", pii_present)

    backend, model_tier, requires_grounding = _ROUTING_TABLE[key]

    # Classifier can override grounding if it says True; table is the floor
    if classifier_result.get("requires_grounding", False):
        requires_grounding = True

    # Force local for PII regardless of classifier output
    if pii_present:
        requires_grounding = False

    return TaskRoute(
        task_type=task_type,
        complexity=complexity,
        requires_grounding=requires_grounding,
        pii_present=pii_present,
        backend=backend,
        model_tier=model_tier,
        estimated_tokens=estimated_tokens,
    )


class TaskRouter:
    """Routes ``DocumentExtract`` instances to the correct backend and model tier.

    Args:
        config: Pipeline configuration dict (from ``providers.toml``). Must
            contain a ``[router]`` section with at least ``classifier_model``
            and optionally ``pii_ner_model`` and ``classifier_budget_tokens``.
        vault: Optional pre-constructed :class:`PIIVault`. If ``None`` a new
            vault is created for each :meth:`route` call.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        vault: PIIVault | None = None,
    ) -> None:
        self._config = config or {}
        router_cfg = self._config.get("router", {})
        self._classifier_model: str = router_cfg.get(
            "classifier_model", "gemini-2.5-flash-latest"
        )
        self._budget_tokens: int = int(
            router_cfg.get("classifier_budget_tokens", 512)
        )
        self._pii_ner_model: str = router_cfg.get(
            "pii_ner_model", "urchade/gliner_medium-v2.1"
        )
        self._shared_vault = vault  # optional shared vault across calls

    def _get_vault(self) -> PIIVault:
        """Return the configured vault (shared or fresh)."""
        if self._shared_vault is not None:
            return self._shared_vault
        return PIIVault(gliner_model=self._pii_ner_model)

    @retry(max_attempts=2, delay=0.5)
    def _classify_with_gemini(self, extract: DocumentExtract) -> dict[str, Any]:
        """Call Gemini Flash to classify the document task."""
        try:
            from google import genai  # type: ignore[import]
            from google.genai import types  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for the task classifier. "
                "Install it with: uv pip install google-genai"
            ) from exc

        import os
        api_key = (
            self._config.get("providers", {})
            .get("cloud_gemini", {})
            .get("api_key", "")
            or os.environ.get("GEMINI_API_KEY", "")
        )
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set; cannot use Gemini Flash classifier."
            )

        client = genai.Client(api_key=api_key)
        user_prompt = _build_classifier_prompt(
            extract, max_chars=self._budget_tokens * 4  # ~4 chars/token
        )
        config = types.GenerateContentConfig(
            system_instruction=_CLASSIFIER_SYSTEM,
            temperature=0.0,
            response_mime_type="application/json",
        )
        response = client.models.generate_content(
            model=self._classifier_model,
            contents=user_prompt,
            config=config,
        )
        raw = response.text or "{}"
        return json.loads(raw)

    def route(
        self,
        extract: DocumentExtract,
        vault: PIIVault | None = None,
    ) -> tuple[TaskRoute, PIIVault]:
        """Determine the ``TaskRoute`` for a document.

        **Stage 1** — PII gate (always local).
        **Stage 2** — Gemini Flash classification (skipped when PII forces local).

        Args:
            extract: The :class:`~servo_skull.models.DocumentExtract` to route.
            vault: Optional caller-supplied vault. Overrides the instance vault.

        Returns:
            A tuple of ``(TaskRoute, PIIVault)`` where the vault holds the
            entity map populated during this call. The caller is responsible
            for calling :meth:`~servo_skull.pii_vault.PIIVault.clear` after use.
        """
        active_vault = vault or self._get_vault()

        # --- Stage 1: PII Gate ---
        pii_result: PIIClassification = active_vault.scan(extract.extracted_text)
        pii_result = pii_result.model_copy(update={"document_id": extract.document_id})

        if pii_result.contains_pii:
            logger.info(
                "TaskRouter [%s]: PII detected (%d entities) → forced local route.",
                extract.document_id,
                pii_result.entity_count,
            )
            route = TaskRoute(
                task_type="gap_analysis",
                complexity="medium",
                requires_grounding=False,
                pii_present=True,
                backend="mlx",
                model_tier="local",
                estimated_tokens=len(extract.extracted_text) // 4,
            )
            return route, active_vault

        # --- Stage 2: Gemini Flash Classifier ---
        try:
            classifier_output = self._classify_with_gemini(extract)
            logger.info(
                "TaskRouter [%s]: classifier → %s",
                extract.document_id,
                classifier_output,
            )
            route = _apply_routing_table(classifier_output, pii_present=False)
        except Exception as exc:
            logger.warning(
                "TaskRouter [%s]: classifier failed (%s), using fallback route.",
                extract.document_id,
                exc,
            )
            route = _fallback_route(pii_present=False)

        return route, active_vault
