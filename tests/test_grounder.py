"""Test LLM grounder for gap analysis and hallucination detection."""
import json
import pytest
from unittest.mock import MagicMock, patch
import httpx

from servo_skull.grounder import (
    analyze_gaps,
    hallucination_score_func,
    _is_grounded,
)
from servo_skull.llm_client import LLMClient as UnifiedLLMClient
from servo_skull.llm_providers import OllamaProvider, OpenAIProvider
from servo_skull.models import DocumentExtract, GapAnalysis


class TestUnifiedLLMClient:
    """Test unified LLMClient with provider chain."""

    def test_unified_llm_client_initialization(self):
        """Test unified LLMClient initialization with primary and optional fallback."""
        primary = MagicMock(spec=OllamaProvider)
        fallback = MagicMock(spec=OpenAIProvider)

        client = UnifiedLLMClient(primary=primary, fallback=fallback)
        assert client.primary is primary
        assert client.fallback is fallback
        assert client.fallback_triggered is False
        assert client.fallback_count == 0

    def test_unified_llm_client_chat_primary_success(self):
        """Test unified LLMClient uses primary provider on success."""
        primary = MagicMock(spec=OllamaProvider)
        primary.chat.return_value = '{"gaps": {}, "risk_flags": []}'

        client = UnifiedLLMClient(primary=primary)
        result = client.chat("system", "user")

        assert result == '{"gaps": {}, "risk_flags": []}'
        primary.chat.assert_called_once()

    def test_unified_llm_client_fallback_on_timeout(self):
        """Test unified LLMClient falls back on timeout."""
        primary = MagicMock(spec=OllamaProvider)
        primary.chat.side_effect = httpx.TimeoutException("Timeout")
        fallback = MagicMock(spec=OpenAIProvider)
        fallback.chat.return_value = '{"gaps": {}, "risk_flags": []}'

        client = UnifiedLLMClient(primary=primary, fallback=fallback)
        result = client.chat("system", "user")

        assert result == '{"gaps": {}, "risk_flags": []}'
        assert client.fallback_triggered is True
        assert client.fallback_count == 1
        fallback.chat.assert_called_once()

    def test_unified_llm_client_fallback_on_connect_error(self):
        """Test unified LLMClient falls back on connection error."""
        primary = MagicMock(spec=OllamaProvider)
        primary.chat.side_effect = httpx.ConnectError("Connection failed")
        fallback = MagicMock(spec=OpenAIProvider)
        fallback.chat.return_value = '{"gaps": {}, "risk_flags": []}'

        client = UnifiedLLMClient(primary=primary, fallback=fallback)
        result = client.chat("system", "user")

        assert result == '{"gaps": {}, "risk_flags": []}'
        assert client.fallback_triggered is True
        fallback.chat.assert_called_once()


class TestAnalyzeGaps:
    """Test analyze_gaps function."""

    def test_analyze_gaps_with_llm_client_instance(self, sample_document_extract):
        """Test analyze_gaps accepts UnifiedLLMClient instance."""
        codex = "Test codex framework"

        primary = MagicMock(spec=OllamaProvider)
        primary.chat.return_value = json.dumps({
            "gaps": {
                "stakeholder": [
                    {"gap": "Missing requirement", "severity": "high", "context": "Test"}
                ]
            },
            "risk_flags": [],
            "security_flags": [],
        })

        client = UnifiedLLMClient(primary=primary)
        result = analyze_gaps(sample_document_extract, codex, llm_client=client)

        assert isinstance(result, GapAnalysis)
        assert result.document_id == sample_document_extract.document_id
        primary.chat.assert_called_once()

    def test_analyze_gaps_creates_default_client_if_not_provided(self, sample_document_extract):
        """Test analyze_gaps creates default UnifiedLLMClient if none provided."""
        codex = "Test codex"

        with patch("servo_skull.grounder.create_provider") as mock_create:
            mock_provider = MagicMock()
            mock_provider.chat.return_value = json.dumps({
                "gaps": {},
                "risk_flags": [],
                "security_flags": [],
            })
            mock_create.return_value = mock_provider

            result = analyze_gaps(sample_document_extract, codex)

            assert isinstance(result, GapAnalysis)
            assert result.document_id == sample_document_extract.document_id

    def test_analyze_gaps_returns_gap_analysis(self, sample_document_extract):
        """Test analyze_gaps returns GapAnalysis model."""
        codex = "Test codex framework"

        primary = MagicMock(spec=OllamaProvider)
        primary.chat.return_value = json.dumps({
            "gaps": {
                "stakeholder": [
                    {"gap": "Missing requirement", "severity": "high", "context": "Test"}
                ]
            },
            "risk_flags": [],
            "security_flags": [],
        })

        client = UnifiedLLMClient(primary=primary)
        result = analyze_gaps(sample_document_extract, codex, llm_client=client)

        assert isinstance(result, GapAnalysis)
        assert result.document_id == sample_document_extract.document_id

    def test_analyze_gaps_includes_document_context(self, sample_document_extract):
        """Test returned GapAnalysis includes document context."""
        codex = "Test codex"

        primary = MagicMock(spec=OllamaProvider)
        primary.chat.return_value = json.dumps({
            "gaps": {},
            "risk_flags": [],
            "security_flags": [],
        })

        client = UnifiedLLMClient(primary=primary)
        result = analyze_gaps(sample_document_extract, codex, llm_client=client)

        assert result.document_id == sample_document_extract.document_id
        assert result.llm_model is not None

    def test_analyze_gaps_hallucination_score_in_range(self, sample_document_extract):
        """Test hallucination_score is in valid range 0.0-1.0."""
        codex = "Test codex"

        primary = MagicMock(spec=OllamaProvider)
        primary.chat.return_value = json.dumps({
            "gaps": {
                "stakeholder": [
                    {"gap": "Missing requirement", "severity": "high", "context": "Test"}
                ]
            },
            "risk_flags": [],
            "security_flags": [],
        })

        client = UnifiedLLMClient(primary=primary)
        result = analyze_gaps(sample_document_extract, codex, llm_client=client)

        assert 0.0 <= result.hallucination_score <= 1.0

    def test_analyze_gaps_includes_grounding_notes(self, sample_document_extract):
        """Test returned GapAnalysis includes grounding_notes."""
        codex = "Test codex"

        primary = MagicMock(spec=OllamaProvider)
        primary.chat.return_value = json.dumps({
            "gaps": {},
            "risk_flags": [],
            "security_flags": [],
        })

        client = UnifiedLLMClient(primary=primary)
        result = analyze_gaps(sample_document_extract, codex, llm_client=client)

        assert result.grounding_notes is not None
        assert isinstance(result.grounding_notes, str)
        assert len(result.grounding_notes) > 0

    def test_analyze_gaps_handles_http_error(self, sample_document_extract):
        """Test analyze_gaps handles HTTP errors gracefully."""
        codex = "Test codex"

        primary = MagicMock(spec=OllamaProvider)
        primary.chat.side_effect = httpx.HTTPError("HTTP error")

        client = UnifiedLLMClient(primary=primary)
        result = analyze_gaps(sample_document_extract, codex, llm_client=client)

        assert isinstance(result, GapAnalysis)
        assert result.hallucination_score == 0.0

    def test_analyze_gaps_handles_invalid_json_response(self, sample_document_extract):
        """Test analyze_gaps handles invalid JSON from LLM."""
        codex = "Test codex"

        primary = MagicMock(spec=OllamaProvider)
        primary.chat.return_value = "This is not valid JSON"

        client = UnifiedLLMClient(primary=primary)
        result = analyze_gaps(sample_document_extract, codex, llm_client=client)

        assert isinstance(result, GapAnalysis)
        assert "json" in result.grounding_notes.lower()


class TestHallucinationScore:
    """Test hallucination_score_func function."""

    def test_hallucination_score_fully_grounded(self, sample_document_extract):
        """Test hallucination_score returns 0.0 when all claims are grounded."""
        extract_text = sample_document_extract.extracted_text
        llm_claims = {
            "gaps": {
                "stakeholder": [
                    {"gap": "extracted text issue", "severity": "high", "context": "document"}
                ]
            },
            "risk_flags": [],
            "security_flags": [],
        }

        score = hallucination_score_func(extract_text, llm_claims)
        assert 0.0 <= score < 0.5  # Should be mostly grounded

    def test_hallucination_score_fully_speculative(self, sample_document_extract):
        """Test hallucination_score returns high when no claims are grounded."""
        extract_text = sample_document_extract.extracted_text
        llm_claims = {
            "gaps": {
                "stakeholder": [
                    {"gap": "xyzabc quantum xyz", "severity": "high", "context": "foobar baz qux"}
                ]
            },
            "risk_flags": [],
            "security_flags": [],
        }

        score = hallucination_score_func(extract_text, llm_claims)
        assert score > 0.5  # Should be mostly speculative

    def test_hallucination_score_partially_grounded(self, sample_document_extract):
        """Test hallucination_score returns value between 0.0-1.0 for mixed claims."""
        extract_text = sample_document_extract.extracted_text
        llm_claims = {
            "gaps": {
                "stakeholder": [
                    {"gap": "extracted text problem", "severity": "high", "context": "test"},
                    {"gap": "xyzabc quantum", "severity": "high", "context": "foobar qux"},
                ]
            },
            "risk_flags": [],
            "security_flags": [],
        }

        score = hallucination_score_func(extract_text, llm_claims)
        assert 0.0 <= score <= 1.0
        assert 0.3 <= score <= 0.7  # Mixed grounding

    def test_hallucination_score_case_insensitive(self, sample_document_extract):
        """Test hallucination_score matches case-insensitively."""
        extract_text = sample_document_extract.extracted_text.lower()
        llm_claims = {
            "gaps": {
                "stakeholder": [
                    # "PDF" appears in "sample.pdf"
                    {"gap": "PDF DOCUMENT", "severity": "high", "context": "This IS Extracted"}
                ]
            },
            "risk_flags": [],
            "security_flags": [],
        }

        score = hallucination_score_func(extract_text, llm_claims)
        assert score < 0.5  # Should be mostly grounded (PDF match)

    def test_hallucination_score_empty_claims(self, sample_document_extract):
        """Test hallucination_score returns 0.0 for empty claims."""
        extract_text = sample_document_extract.extracted_text
        llm_claims = {}

        score = hallucination_score_func(extract_text, llm_claims)
        assert score == 0.0

    def test_hallucination_score_with_risk_flags(self, sample_document_extract):
        """Test hallucination_score evaluates risk_flags."""
        extract_text = "This document discusses timeline constraints and delivery schedules."
        llm_claims = {
            "gaps": {},
            "risk_flags": [
                {"risk": "timeline constraint", "recommendation": "review schedule"},
                {"risk": "xyzabc quantum flux", "recommendation": "foobar qux"},
            ],
            "security_flags": [],
        }

        score = hallucination_score_func(extract_text, llm_claims)
        assert 0.0 <= score <= 1.0
        assert 0.3 <= score <= 0.7  # Mixed grounding

    def test_hallucination_score_with_security_flags(self, sample_document_extract):
        """Test hallucination_score evaluates security_flags."""
        extract_text = "This document discusses security and encryption requirements."
        llm_claims = {
            "gaps": {},
            "risk_flags": [],
            "security_flags": [
                "encryption requirements",
                "xyzabc quantum security",
            ],
        }

        score = hallucination_score_func(extract_text, llm_claims)
        assert 0.0 <= score <= 1.0
        assert score >= 0.4  # At least some ungrounded


class TestIsGrounded:
    """Test _is_grounded helper function."""

    def test_is_grounded_claim_in_text(self):
        """Test _is_grounded returns True when claim is in text."""
        claim = "pdf document"
        context = ""
        extract = "This is a PDF document for testing."

        result = _is_grounded(claim, context, extract.lower())
        assert result is True

    def test_is_grounded_claim_not_in_text(self):
        """Test _is_grounded returns False when claim is not in text."""
        claim = "xyzabc quantum flux"
        context = ""
        extract = "This is a PDF document for testing."

        result = _is_grounded(claim, context, extract.lower())
        assert result is False

    def test_is_grounded_with_context(self):
        """Test _is_grounded checks context keywords."""
        claim = "xyzabc quantum"
        context = "testing and verification"
        extract = "This is a PDF document for testing."

        result = _is_grounded(claim, context, extract.lower())
        assert result is True  # "testing" from context is in extract

    def test_is_grounded_empty_claim(self):
        """Test _is_grounded returns True for empty claim."""
        claim = ""
        context = ""
        extract = "This is a PDF document."

        result = _is_grounded(claim, context, extract.lower())
        assert result is True  # Empty claim considered grounded

    def test_is_grounded_case_insensitive(self):
        """Test _is_grounded is case-insensitive."""
        claim = "PDF DOCUMENT"
        context = ""
        extract = "This is a pdf document for testing."

        result = _is_grounded(claim, context, extract.lower())
        assert result is True


class TestIntegration:
    """Integration tests for grounder module."""

    def test_end_to_end_gap_analysis(self, sample_document_extract):
        """Test end-to-end gap analysis with realistic data."""
        codex = """
        Your analysis must cover:
        1. Stakeholder requirements (Legal, Sales, Engineering, Finance)
        2. Delivery methodology (PRINCE2, Agile, ITIL)
        3. Affected components (Data, Service, Client, Integration)
        4. Compliance standards (GDPR, SOC2, WCAG, ISO standards)
        """

        with patch("servo_skull.grounder.LLMClient.chat") as mock_chat:
            mock_chat.return_value = json.dumps({
                "gaps": {
                    "stakeholder": [
                        {
                            "gap": "GDPR data residency requirement not specified",
                            "severity": "high",
                            "context": "extracted text",
                            "dimension": "Legal",
                        }
                    ],
                    "compliance": [
                        {
                            "gap": "WCAG 2.2 accessibility audit missing",
                            "severity": "high",
                            "context": "document",
                            "standard": "WCAG",
                        }
                    ],
                },
                "risk_flags": [
                    {
                        "risk": "Tight timeline with regulatory constraints",
                        "domain": "operational",
                        "severity": "high",
                        "recommendation": "Request timeline extension",
                    }
                ],
                "security_flags": [],
            })

            result = analyze_gaps(sample_document_extract, codex)

            assert isinstance(result, GapAnalysis)
            assert result.document_id == sample_document_extract.document_id
            assert len(result.gaps) > 0
            assert len(result.risk_flags) > 0
            assert 0.0 <= result.hallucination_score <= 1.0
            assert len(result.grounding_notes) > 0

    def test_hallucination_scoring_with_realistic_data(self):
        """Test hallucination scoring with realistic extracted text."""
        extract_text = """
        REQUEST FOR PROPOSAL: Enterprise Document Management Platform

        1. Overview
        The Globex Corporation requires a modern document platform compliant with SOC 2
        and GDPR regulations. The system must support OAuth2 authentication
        and integrate with existing CRM infrastructure.

        2. Requirements
        - GDPR data residency (EU only)
        - ISO 27001 compliance
        - Real-time search/indexing integration
        - 24/7 availability SLA
        """

        llm_claims = {
            "gaps": {
                "stakeholder": [
                    # This should be grounded
                    {
                        "gap": "SOC 2 compliance requirement",
                        "severity": "high",
                        "context": "Globex Corporation",
                    },
                    # This should be partially grounded
                    {
                        "gap": "Multi-currency support for international payments",
                        "severity": "medium",
                        "context": "document platform",
                    },
                    # This should be ungrounded
                    {
                        "gap": "Quantum encryption for all transactions",
                        "severity": "high",
                        "context": "xyzabc security",
                    },
                ]
            },
            "risk_flags": [],
            "security_flags": [],
        }

        score = hallucination_score_func(extract_text, llm_claims)
        assert 0.0 <= score <= 1.0
        # With 1 fully grounded, 1 partially grounded, 1 ungrounded: ~0.33
        assert 0.2 <= score <= 0.5
