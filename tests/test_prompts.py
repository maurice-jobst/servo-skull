"""Test LLM prompt template rendering."""
import json
import pytest
from servo_skull._prompts import render_gap_analysis_prompt, render_security_check_prompt
from servo_skull.models import DocumentExtract


class TestRenderGapAnalysisPrompt:
    """Tests for render_gap_analysis_prompt function."""

    def test_render_gap_analysis_prompt_returns_dict(self, sample_document_extract):
        """Verify render_gap_analysis_prompt returns dict with system and user keys."""
        codex = "Sample domain codex rules text for testing."
        result = render_gap_analysis_prompt(sample_document_extract, codex)

        assert isinstance(result, dict)
        assert "system" in result
        assert "user" in result
        assert isinstance(result["system"], str)
        assert isinstance(result["user"], str)

    def test_render_gap_analysis_prompt_includes_codex(self, sample_document_extract):
        """Verify codex framework reference appears in system prompt."""
        codex = "Sample domain codex rules text for testing."
        result = render_gap_analysis_prompt(sample_document_extract, codex)

        system = result["system"].lower()
        # Check for framework/rules references
        assert (
            "codex" in system or "framework" in system or "rules" in system
        ), "System prompt should reference codex/framework/rules"

    def test_render_gap_analysis_prompt_includes_extracted_text(self, sample_document_extract):
        """Verify extracted text from DocumentExtract appears in user prompt."""
        codex = "Sample domain codex rules text for testing."
        result = render_gap_analysis_prompt(sample_document_extract, codex)

        user = result["user"]
        # Filename should appear
        assert sample_document_extract.original_filename in user
        # At least part of the extracted text should appear
        assert sample_document_extract.extracted_text in user

    def test_render_gap_analysis_prompt_json_format_instruction(self, sample_document_extract):
        """Verify JSON output format instruction appears in user prompt."""
        codex = "Sample domain codex rules text for testing."
        result = render_gap_analysis_prompt(sample_document_extract, codex)

        user = result["user"]
        user_lower = user.lower()

        # Check for JSON format guidance
        assert (
            "json" in user_lower
        ), "User prompt should instruct LLM to output JSON"

        # Check for gap structure references
        assert (
            "gaps" in user_lower
        ), "User prompt should reference 'gaps' structure"

    def test_render_gap_analysis_prompt_hallucination_instruction(self, sample_document_extract):
        """Verify hallucination scoring instruction appears in user prompt."""
        codex = "Sample domain codex rules text for testing."
        result = render_gap_analysis_prompt(sample_document_extract, codex)

        user = result["user"]
        user_lower = user.lower()

        # Check for hallucination-related language
        assert (
            "hallucination" in user_lower or "grounded" in user_lower
        ), "User prompt should instruct on hallucination scoring"

        # Check for 0.0-1.0 scale reference
        assert (
            "0.0" in result["system"] or "1.0" in result["system"]
        ), "System prompt should reference hallucination scale"

    def test_render_gap_analysis_prompt_four_dimensions(self, sample_document_extract):
        """Verify the 4D matrix dimensions are mentioned in system prompt."""
        codex = "Sample domain codex rules text for testing."
        result = render_gap_analysis_prompt(sample_document_extract, codex)

        system = result["system"].lower()

        # Check for all four dimensions
        assert "stakeholder" in system, "System prompt should reference Stakeholder dimension"
        assert "methodology" in system, "System prompt should reference Methodology dimension"
        assert "component" in system, "System prompt should reference Components dimension"
        assert "compliance" in system, "System prompt should reference Compliance dimension"

    def test_render_gap_analysis_prompt_codex_included_in_user(self, sample_document_extract):
        """Verify the codex is actually included in the user prompt."""
        codex = "UNIQUE_CODEX_MARKER_12345"
        result = render_gap_analysis_prompt(sample_document_extract, codex)

        user = result["user"]
        assert codex in user, "User prompt should include the full codex text"

    def test_render_gap_analysis_prompt_document_metadata(self, sample_document_extract):
        """Verify document metadata appears in user prompt."""
        codex = "Sample domain codex rules text for testing."
        result = render_gap_analysis_prompt(sample_document_extract, codex)

        user = result["user"]

        # Check that document metadata is included
        assert str(sample_document_extract.confidence) in user
        assert sample_document_extract.document_type in user


class TestRenderSecurityCheckPrompt:
    """Tests for render_security_check_prompt function."""

    def test_render_security_check_prompt_returns_dict(self, sample_document_extract):
        """Verify render_security_check_prompt returns dict with system and user keys."""
        result = render_security_check_prompt(sample_document_extract)

        assert isinstance(result, dict)
        assert "system" in result
        assert "user" in result
        assert isinstance(result["system"], str)
        assert isinstance(result["user"], str)

    def test_render_security_check_prompt_checks_injection(self, sample_document_extract):
        """Verify injection pattern checking is mentioned in system prompt."""
        result = render_security_check_prompt(sample_document_extract)

        system = result["system"].lower()

        # Check for injection-related terms
        assert (
            "injection" in system or "sql" in system or "code" in system
        ), "System prompt should mention injection pattern checking"

    def test_render_security_check_prompt_checks_fraud(self, sample_document_extract):
        """Verify fraud pattern checking is mentioned in system prompt."""
        result = render_security_check_prompt(sample_document_extract)

        system = result["system"].lower()

        # Check for fraud-related terms
        assert (
            "fraud" in system or "payment" in system or "fictitious" in system
        ), "System prompt should mention fraud pattern checking"

    def test_render_security_check_prompt_includes_extracted_text(self, sample_document_extract):
        """Verify extracted text appears in user prompt."""
        result = render_security_check_prompt(sample_document_extract)

        user = result["user"]

        # Filename should appear
        assert sample_document_extract.original_filename in user
        # At least part of the extracted text should appear
        assert sample_document_extract.extracted_text in user

    def test_render_security_check_prompt_json_format(self, sample_document_extract):
        """Verify JSON format instruction appears in security check prompt."""
        result = render_security_check_prompt(sample_document_extract)

        user = result["user"]
        user_lower = user.lower()

        assert "json" in user_lower, "User prompt should instruct JSON output"

    def test_render_security_check_prompt_misinformation_check(self, sample_document_extract):
        """Verify misinformation checking is mentioned."""
        result = render_security_check_prompt(sample_document_extract)

        system = result["system"].lower()

        assert (
            "misinformation" in system or "contradicting" in system or "conflicting" in system
        ), "System prompt should mention misinformation checking"

    def test_render_security_check_prompt_ai_watermark_check(self, sample_document_extract):
        """Verify AI watermark detection is mentioned."""
        result = render_security_check_prompt(sample_document_extract)

        system = result["system"].lower()

        assert (
            "ai" in system or "watermark" in system or "synthetic" in system
        ), "System prompt should mention AI watermark detection"

    def test_render_security_check_prompt_document_metadata(self, sample_document_extract):
        """Verify document metadata appears in user prompt."""
        result = render_security_check_prompt(sample_document_extract)

        user = result["user"]

        # Check that document metadata is included
        assert sample_document_extract.original_filename in user
        assert sample_document_extract.document_type in user

    def test_render_security_check_prompt_output_structure(self, sample_document_extract):
        """Verify security check output structure is documented."""
        result = render_security_check_prompt(sample_document_extract)

        user = result["user"]
        user_lower = user.lower()

        # Check for expected output fields
        assert (
            "security_issues" in user_lower
        ), "User prompt should document security_issues field"
        assert (
            "fraud_indicator" in user_lower
        ), "User prompt should document fraud_indicators field"


class TestPromptParameterHandling:
    """Test parameter handling and edge cases."""

    def test_gap_analysis_handles_large_codex(self, sample_document_extract):
        """Verify gap analysis handles large codex text."""
        # Create a large codex text
        large_codex = "Rule " + "X " * 1000  # Large codex text

        result = render_gap_analysis_prompt(sample_document_extract, large_codex)

        assert large_codex in result["user"]
        assert isinstance(result, dict)
        assert len(result["system"]) > 0
        assert len(result["user"]) > 0

    def test_gap_analysis_handles_special_chars_in_filename(self):
        """Verify prompts handle special characters in filenames."""
        extract = DocumentExtract(
            original_filename="tender-2025_€1.5M_📋.pdf",
            document_type="pdf",
            extracted_text="Test content",
            confidence=0.95,
            extraction_tool="test",
        )

        result = render_gap_analysis_prompt(extract, "Codex")

        assert extract.original_filename in result["user"]

    def test_gap_analysis_handles_empty_extracted_text(self):
        """Verify prompts handle empty extracted text gracefully."""
        extract = DocumentExtract(
            original_filename="empty.pdf",
            document_type="pdf",
            extracted_text="",
            confidence=0.1,
            extraction_tool="test",
        )

        result = render_gap_analysis_prompt(extract, "Codex")

        assert isinstance(result, dict)
        assert "system" in result
        assert "user" in result

    def test_security_check_handles_multiline_text(self, sample_document_extract):
        """Verify security check handles multiline extracted text."""
        extract = DocumentExtract(
            original_filename="test.pdf",
            document_type="pdf",
            extracted_text="Line 1\nLine 2\nLine 3\n\nAnother section",
            confidence=0.95,
            extraction_tool="test",
        )

        result = render_security_check_prompt(extract)

        assert isinstance(result, dict)
        assert len(result["system"]) > 0
        assert len(result["user"]) > 0

    def test_prompt_returns_serializable_dict(self, sample_document_extract):
        """Verify returned dicts are JSON-serializable."""
        import json

        codex = "Test codex"
        result_gap = render_gap_analysis_prompt(sample_document_extract, codex)
        result_security = render_security_check_prompt(sample_document_extract)

        # Should not raise JSON serialization error
        json.dumps(result_gap)
        json.dumps(result_security)

    def test_gap_analysis_preserves_confidence_score(self, sample_document_extract):
        """Verify document confidence score appears in prompt."""
        codex = "Codex"
        result = render_gap_analysis_prompt(sample_document_extract, codex)

        user = result["user"]
        assert str(sample_document_extract.confidence) in user

    def test_security_check_references_document_type(self, sample_document_extract):
        """Verify document type is referenced in security check."""
        result = render_security_check_prompt(sample_document_extract)

        user = result["user"]
        assert sample_document_extract.document_type in user
