"""Tests for TaskRouter — PII gate and Gemini Flash classification."""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from servo_skull.models import DocumentExtract, TaskRoute


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_extract(text: str = "Sample text.", pii: bool = False) -> DocumentExtract:
    name = "pii_doc.pdf" if pii else "clean_doc.pdf"
    return DocumentExtract(
        original_filename=name,
        document_type="pdf",
        extracted_text=text,
        confidence=0.95,
        extraction_tool="pymupdf4llm",
    )


def _make_router(config: dict | None = None):
    from servo_skull.task_router import TaskRouter
    return TaskRouter(config=config or {})


def _mock_vault(contains_pii: bool, entity_count: int = 0):
    """Return a PIIVault mock with scan() pre-configured."""
    from servo_skull.models import PIIClassification
    vault = MagicMock()
    vault.__len__ = MagicMock(return_value=entity_count)
    vault.scan.return_value = PIIClassification(
        document_id="test-id",
        contains_pii=contains_pii,
        entity_types_found=["person"] if contains_pii else [],
        entity_count=entity_count,
        forced_local=contains_pii,
    )
    vault.pseudonymize.side_effect = lambda t: t  # no-op passthrough
    vault.rehydrate.side_effect = lambda t: t
    return vault


# ---------------------------------------------------------------------------
# PII Gate Tests
# ---------------------------------------------------------------------------

class TestPIIGate:
    def test_pii_detected_forces_local(self):
        """When vault detects PII, route must be local (mlx or ollama)."""
        router = _make_router()
        vault = _mock_vault(contains_pii=True, entity_count=3)
        extract = _make_extract(text="John Smith works at Globex.", pii=True)

        route, _ = router.route(extract, vault=vault)

        assert route.pii_present is True
        assert route.backend in ("mlx", "ollama")
        assert route.requires_grounding is False

    def test_no_pii_proceeds_to_classifier(self):
        """Clean document should attempt the Gemini classifier stage."""
        router = _make_router(config={
            "providers": {"cloud_gemini": {"api_key": "test-key"}}
        })
        vault = _mock_vault(contains_pii=False)
        extract = _make_extract(text="Distributed caching specification.")

        classifier_resp = {
            "task_type": "rule_synthesis",
            "complexity": "high",
            "requires_grounding": False,
            "estimated_tokens": 3000,
        }

        with patch.object(router, "_classify_with_gemini", return_value=classifier_resp):
            route, _ = router.route(extract, vault=vault)

        assert route.pii_present is False
        assert route.task_type == "rule_synthesis"
        assert route.backend == "gemini"
        assert route.model_tier == "reason"

    def test_classifier_failure_uses_fallback(self):
        """If Gemini classifier raises, a safe fallback route is returned."""
        router = _make_router()
        vault = _mock_vault(contains_pii=False)
        extract = _make_extract()

        with patch.object(
            router, "_classify_with_gemini", side_effect=RuntimeError("API down")
        ):
            route, _ = router.route(extract, vault=vault)

        assert isinstance(route, TaskRoute)
        # Fallback should not use MLX (reserved for PII)
        assert route.backend in ("gemini", "ollama")


# ---------------------------------------------------------------------------
# Routing Table Tests
# ---------------------------------------------------------------------------

class TestRoutingTable:
    """Verify that _apply_routing_table produces correct backend/tier/grounding."""

    @pytest.mark.parametrize("task_type,pii,expected_backend,grounding", [
        ("extraction",     True,  "mlx",    False),
        ("extraction",     False, "mlx",    False),
        ("rule_synthesis", True,  "mlx",    False),
        ("rule_synthesis", False, "gemini", False),
        ("gap_analysis",   True,  "mlx",    False),
        ("gap_analysis",   False, "gemini", True),
        ("summary",        True,  "mlx",    False),
        ("summary",        False, "gemini", True),
        ("security_check", True,  "ollama", False),
        ("security_check", False, "gemini", False),
        ("deep_research",  False, "gemini", True),
    ])
    def test_routing_table_entry(self, task_type, pii, expected_backend, grounding):
        from servo_skull.task_router import _apply_routing_table
        result = _apply_routing_table(
            {"task_type": task_type, "complexity": "medium",
             "requires_grounding": False, "estimated_tokens": 1000},
            pii_present=pii,
        )
        assert result.backend == expected_backend, (
            f"task={task_type} pii={pii}: "
            f"expected backend={expected_backend}, got {result.backend}"
        )
        assert result.requires_grounding == grounding, (
            f"task={task_type} pii={pii}: "
            f"expected grounding={grounding}, got {result.requires_grounding}"
        )

    def test_classifier_grounding_override(self):
        """Classifier returning requires_grounding=True must propagate when pii=False."""
        from servo_skull.task_router import _apply_routing_table
        result = _apply_routing_table(
            {"task_type": "rule_synthesis", "complexity": "high",
             "requires_grounding": True, "estimated_tokens": 5000},
            pii_present=False,
        )
        assert result.requires_grounding is True

    def test_pii_always_disables_grounding(self):
        """Even if classifier says grounding=True, PII must force it off."""
        from servo_skull.task_router import _apply_routing_table
        result = _apply_routing_table(
            {"task_type": "gap_analysis", "complexity": "high",
             "requires_grounding": True, "estimated_tokens": 2000},
            pii_present=True,
        )
        assert result.requires_grounding is False
        assert result.backend in ("mlx", "ollama")


# ---------------------------------------------------------------------------
# Gemini Classifier Prompt Tests
# ---------------------------------------------------------------------------

class TestClassifierPrompt:
    def test_prompt_contains_filename(self):
        from servo_skull.task_router import _build_classifier_prompt
        extract = _make_extract("Some content about platform requirements.")
        extract = extract.model_copy(update={"original_filename": "tender_2026.pdf"})
        prompt = _build_classifier_prompt(extract)
        assert "tender_2026.pdf" in prompt

    def test_prompt_truncates_to_budget(self):
        from servo_skull.task_router import _build_classifier_prompt
        long_text = "X" * 10_000
        extract = _make_extract(long_text)
        prompt = _build_classifier_prompt(extract, max_chars=512)
        # The excerpt in the prompt should not exceed 512 chars of 'X'
        x_count = prompt.count("X")
        assert x_count <= 512
