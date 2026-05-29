"""Test RAG markdown builder."""
import json
from datetime import datetime

import pytest
from servo_skull.markdown_builder import MarkdownBuilder
from servo_skull.models import RichMarkdown


@pytest.fixture
def markdown_builder():
    """Create a MarkdownBuilder instance."""
    return MarkdownBuilder()


@pytest.fixture
def sample_security_dict():
    """Create a sample security_dict from security_checker."""
    return {
        "security_issues": [
            {"issue": "Potential SQL injection pattern detected"},
            {"issue": "Unverified claims about system performance"},
        ],
        "misinformation_risks": [
            {"risk": "Unsubstantiated performance claims"},
        ],
        "ai_watermarks": [],
        "fraud_indicators": [],
        "recommendations": [
            {"recommendation": "Review security controls before deployment"},
            {"recommendation": "Validate performance claims with independent testing"},
        ],
    }


class TestMarkdownBuilderReturnsRichMarkdown:
    """Test that build_markdown returns RichMarkdown model."""

    def test_build_markdown_returns_rich_markdown(
        self,
        markdown_builder,
        sample_document_extract,
        sample_gap_analysis,
        sample_security_dict,
    ):
        """Calls build_markdown(), verifies returns RichMarkdown model."""
        result = markdown_builder.build_markdown(
            sample_document_extract,
            sample_gap_analysis,
            sample_security_dict,
        )

        assert isinstance(result, RichMarkdown)
        assert result.document_id == sample_document_extract.document_id


class TestMarkdownBuilderFrontmatter:
    """Test frontmatter generation."""

    def test_build_markdown_frontmatter_valid_yaml(
        self,
        markdown_builder,
        sample_document_extract,
        sample_gap_analysis,
        sample_security_dict,
    ):
        """Checks frontmatter is valid YAML dict with required keys."""
        result = markdown_builder.build_markdown(
            sample_document_extract,
            sample_gap_analysis,
            sample_security_dict,
        )

        frontmatter = result.frontmatter
        assert isinstance(frontmatter, dict)

        # Check required keys
        required_keys = [
            "doc_id",
            "filename",
            "document_type",
            "extraction_tool",
            "confidence",
            "analysis_timestamp",
            "hallucination_score",
            "security_issues_count",
        ]
        for key in required_keys:
            assert key in frontmatter, f"Missing required key: {key}"

    def test_build_markdown_frontmatter_correct_values(
        self,
        markdown_builder,
        sample_document_extract,
        sample_gap_analysis,
        sample_security_dict,
    ):
        """Verify frontmatter contains correct values."""
        result = markdown_builder.build_markdown(
            sample_document_extract,
            sample_gap_analysis,
            sample_security_dict,
        )

        fm = result.frontmatter
        assert fm["doc_id"] == sample_document_extract.document_id
        assert fm["filename"] == sample_document_extract.original_filename
        assert fm["document_type"] == sample_document_extract.document_type
        assert fm["extraction_tool"] == sample_document_extract.extraction_tool
        assert fm["confidence"] == sample_document_extract.confidence


class TestMarkdownBuilderContentSections:
    """Test content section generation."""

    def test_build_markdown_includes_all_sections(
        self,
        markdown_builder,
        sample_document_extract,
        sample_gap_analysis,
        sample_security_dict,
    ):
        """Verifies content includes all 6 sections."""
        result = markdown_builder.build_markdown(
            sample_document_extract,
            sample_gap_analysis,
            sample_security_dict,
        )

        content = result.content

        # All 6 sections should be present
        assert "# Document Overview" in content
        assert "# Extracted Content" in content
        assert "# Gap Analysis" in content
        assert "# Risk Assessment" in content
        assert "# Grounding & Hallucination" in content
        assert "# Recommendations" in content

    def test_build_markdown_includes_extracted_text_sample(
        self,
        markdown_builder,
        sample_document_extract,
        sample_gap_analysis,
        sample_security_dict,
    ):
        """Content includes truncated extracted text."""
        result = markdown_builder.build_markdown(
            sample_document_extract,
            sample_gap_analysis,
            sample_security_dict,
        )

        content = result.content
        # Should include a snippet of the extracted text
        assert "extracted text" in content.lower()

    def test_build_markdown_includes_gap_dimensions(
        self,
        markdown_builder,
        sample_document_extract,
        sample_gap_analysis,
        sample_security_dict,
    ):
        """Content mentions all 4 dimensions."""
        result = markdown_builder.build_markdown(
            sample_document_extract,
            sample_gap_analysis,
            sample_security_dict,
        )

        content = result.content.lower()

        # All 4 dimensions should be mentioned
        assert "stakeholder" in content
        assert "methodology" in content
        assert "subsystems" in content
        assert "compliance" in content

    def test_build_markdown_includes_risk_flags(
        self,
        markdown_builder,
        sample_document_extract,
        sample_gap_analysis,
        sample_security_dict,
    ):
        """Content includes risk_flags from GapAnalysis."""
        result = markdown_builder.build_markdown(
            sample_document_extract,
            sample_gap_analysis,
            sample_security_dict,
        )

        content = result.content
        # Risk flags section should be present if there are risk flags
        if sample_gap_analysis.risk_flags:
            assert "Risk Flags" in content

    def test_build_markdown_includes_hallucination_score(
        self,
        markdown_builder,
        sample_document_extract,
        sample_gap_analysis,
        sample_security_dict,
    ):
        """Content includes hallucination_score with explanation."""
        result = markdown_builder.build_markdown(
            sample_document_extract,
            sample_gap_analysis,
            sample_security_dict,
        )

        content = result.content
        # Hallucination score section should be present
        assert "Hallucination Score" in content
        # Score value should be present
        score_str = f"{sample_gap_analysis.hallucination_score:.2f}"
        assert score_str in content


class TestMarkdownBuilderSeeAlso:
    """Test See Also link extraction."""

    def test_build_markdown_see_also_tuples(
        self,
        markdown_builder,
        sample_document_extract,
        sample_gap_analysis,
        sample_security_dict,
    ):
        """see_also is list of (str, str) tuples."""
        result = markdown_builder.build_markdown(
            sample_document_extract,
            sample_gap_analysis,
            sample_security_dict,
        )

        assert isinstance(result.see_also, list)
        for item in result.see_also:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)  # title
            assert isinstance(item[1], str)  # context

    def test_build_markdown_see_also_limit_five(
        self,
        markdown_builder,
        sample_document_extract,
        sample_gap_analysis,
        sample_security_dict,
    ):
        """see_also contains at most 5 links."""
        result = markdown_builder.build_markdown(
            sample_document_extract,
            sample_gap_analysis,
            sample_security_dict,
        )

        assert len(result.see_also) <= 5

    def test_build_markdown_see_also_deduped(
        self,
        markdown_builder,
        sample_document_extract,
        sample_gap_analysis,
        sample_security_dict,
    ):
        """see_also contains no duplicate titles."""
        result = markdown_builder.build_markdown(
            sample_document_extract,
            sample_gap_analysis,
            sample_security_dict,
        )

        titles = [item[0] for item in result.see_also]
        assert len(titles) == len(set(titles)), "Duplicate titles found in see_also"


class TestMarkdownBuilderScriptorumRefs:
    """Test SCRIPTORUM references extraction."""

    def test_build_markdown_scriptorum_refs_present(
        self,
        markdown_builder,
        sample_document_extract,
        sample_gap_analysis,
        sample_security_dict,
    ):
        """scriptorum_refs is list of document IDs."""
        result = markdown_builder.build_markdown(
            sample_document_extract,
            sample_gap_analysis,
            sample_security_dict,
        )

        assert isinstance(result.scriptorum_refs, list)
        # Items should be strings (document references)
        for ref in result.scriptorum_refs:
            assert isinstance(ref, str)

    def test_build_markdown_scriptorum_refs_extracts_filenames(
        self,
        markdown_builder,
    ):
        """scriptorum_refs extracts document filenames from context."""
        from servo_skull.models import DocumentExtract, GapAnalysis

        extract = DocumentExtract(
            document_id="test-123",
            original_filename="test.pdf",
            document_type="pdf",
            extracted_text="Some text",
            confidence=0.95,
            extraction_tool="pymupdf4llm",
        )

        analysis = GapAnalysis(
            document_id="test-123",
            gaps={
                "compliance": [
                    {
                        "gap": "GDPR not addressed",
                        "severity": "high",
                        "context": "See compliance_framework_2025.md for details",
                    }
                ]
            },
            risk_flags=[
                {
                    "risk": "Timeline risk",
                    "severity": "high",
                    "recommendation": "Review tender_specification.pdf before proceeding",
                }
            ],
            security_flags=[],
            hallucination_score=0.1,
            grounding_notes="Grounded",
            llm_model="gemma4-26b-a4b-moe",
        )

        security_dict = {
            "security_issues": [],
            "misinformation_risks": [],
            "ai_watermarks": [],
            "fraud_indicators": [],
            "recommendations": [],
        }

        result = markdown_builder.build_markdown(extract, analysis, security_dict)

        # Should extract filenames from context and recommendations
        assert any("compliance_framework_2025.md" in ref for ref in result.scriptorum_refs)
        assert any("tender_specification.pdf" in ref for ref in result.scriptorum_refs)


class TestMarkdownBuilderBinaricCant:
    """Test Binaric Cant footer generation."""

    def test_build_markdown_binaric_cant_footer_valid_json(
        self,
        markdown_builder,
        sample_document_extract,
        sample_gap_analysis,
        sample_security_dict,
    ):
        """binaric_cant_footer parses as valid JSON."""
        result = markdown_builder.build_markdown(
            sample_document_extract,
            sample_gap_analysis,
            sample_security_dict,
        )

        footer = result.binaric_cant_footer
        assert isinstance(footer, dict)

        # Should be serializable to JSON
        json_str = json.dumps(footer)
        assert isinstance(json_str, str)

    def test_build_markdown_binaric_cant_includes_required_fields(
        self,
        markdown_builder,
        sample_document_extract,
        sample_gap_analysis,
        sample_security_dict,
    ):
        """Footer JSON has required fields."""
        result = markdown_builder.build_markdown(
            sample_document_extract,
            sample_gap_analysis,
            sample_security_dict,
        )

        footer = result.binaric_cant_footer
        required_fields = [
            "doc_id",
            "doc_type",
            "gaps",
            "risks",
            "hallucination_score",
            "security_issues",
            "extraction_confidence",
            "scriptorum_refs",
        ]

        for field in required_fields:
            assert field in footer, f"Missing field in binaric_cant_footer: {field}"

    def test_build_markdown_binaric_cant_footer_structure(
        self,
        markdown_builder,
        sample_document_extract,
        sample_gap_analysis,
        sample_security_dict,
    ):
        """Binaric Cant footer has correct structure."""
        result = markdown_builder.build_markdown(
            sample_document_extract,
            sample_gap_analysis,
            sample_security_dict,
        )

        footer = result.binaric_cant_footer

        # Check nested structures
        assert isinstance(footer["gaps"], dict)
        assert "count" in footer["gaps"]
        assert "dims" in footer["gaps"]
        assert isinstance(footer["gaps"]["count"], int)
        assert isinstance(footer["gaps"]["dims"], list)

        assert isinstance(footer["risks"], dict)
        assert "count" in footer["risks"]
        assert "severity" in footer["risks"]
        assert isinstance(footer["risks"]["count"], int)
        assert isinstance(footer["risks"]["severity"], list)

        assert isinstance(footer["hallucination_score"], float)
        assert isinstance(footer["security_issues"], int)
        assert isinstance(footer["extraction_confidence"], float)
        assert isinstance(footer["scriptorum_refs"], list)


class TestMarkdownBuilderTimestampsAndPaths:
    """Test timestamp and path generation."""

    def test_build_markdown_output_path_set(
        self,
        markdown_builder,
        sample_document_extract,
        sample_gap_analysis,
        sample_security_dict,
    ):
        """output_path is a string."""
        result = markdown_builder.build_markdown(
            sample_document_extract,
            sample_gap_analysis,
            sample_security_dict,
        )

        assert isinstance(result.output_path, str)
        assert len(result.output_path) > 0
        # Should end with .md
        assert result.output_path.endswith(".md")

    def test_build_markdown_generated_timestamp_iso(
        self,
        markdown_builder,
        sample_document_extract,
        sample_gap_analysis,
        sample_security_dict,
    ):
        """generated_timestamp is ISO 8601 format."""
        result = markdown_builder.build_markdown(
            sample_document_extract,
            sample_gap_analysis,
            sample_security_dict,
        )

        # Should be parseable as ISO datetime
        try:
            dt = datetime.fromisoformat(result.generated_timestamp.replace("Z", "+00:00"))
            assert isinstance(dt, datetime)
        except ValueError:
            pytest.fail(f"generated_timestamp is not ISO 8601 format: {result.generated_timestamp}")


class TestMarkdownBuilderEdgeCases:
    """Test edge cases and error handling."""

    def test_build_markdown_handles_empty_gaps(
        self,
        markdown_builder,
        sample_document_extract,
        sample_security_dict,
    ):
        """Handles GapAnalysis with no gaps gracefully."""
        from servo_skull.models import GapAnalysis

        analysis = GapAnalysis(
            document_id=sample_document_extract.document_id,
            gaps={},  # Empty gaps
            risk_flags=[],
            security_flags=[],
            hallucination_score=0.0,
            grounding_notes="No gaps found",
            llm_model="gemma4-26b-a4b-moe",
        )

        result = markdown_builder.build_markdown(
            sample_document_extract,
            analysis,
            sample_security_dict,
        )

        assert isinstance(result, RichMarkdown)
        assert "No gaps identified" in result.content

    def test_build_markdown_handles_empty_security(
        self,
        markdown_builder,
        sample_document_extract,
        sample_gap_analysis,
    ):
        """Handles empty security_dict gracefully."""
        empty_security = {
            "security_issues": [],
            "misinformation_risks": [],
            "ai_watermarks": [],
            "fraud_indicators": [],
            "recommendations": [],
        }

        result = markdown_builder.build_markdown(
            sample_document_extract,
            sample_gap_analysis,
            empty_security,
        )

        assert isinstance(result, RichMarkdown)
        assert result.frontmatter["security_issues_count"] == 0

    def test_build_markdown_preserves_document_context(
        self,
        markdown_builder,
        sample_document_extract,
        sample_gap_analysis,
        sample_security_dict,
    ):
        """Frontmatter includes document context."""
        result = markdown_builder.build_markdown(
            sample_document_extract,
            sample_gap_analysis,
            sample_security_dict,
        )

        fm = result.frontmatter
        assert fm["doc_id"] == sample_document_extract.document_id
        assert fm["filename"] == sample_document_extract.original_filename
        assert fm["document_type"] == sample_document_extract.document_type

    def test_build_markdown_escapes_special_characters(
        self,
        markdown_builder,
    ):
        """Extracted text escapes special markdown characters."""
        from servo_skull.models import DocumentExtract, GapAnalysis

        extract = DocumentExtract(
            document_id="test-123",
            original_filename="test.pdf",
            document_type="pdf",
            extracted_text="Text with `backticks` and \\backslashes\\",
            confidence=0.95,
            extraction_tool="pymupdf4llm",
        )

        analysis = GapAnalysis(
            document_id="test-123",
            gaps={},
            risk_flags=[],
            security_flags=[],
            hallucination_score=0.0,
            grounding_notes="Test",
            llm_model="gemma4-26b-a4b-moe",
        )

        security_dict = {
            "security_issues": [],
            "misinformation_risks": [],
            "ai_watermarks": [],
            "fraud_indicators": [],
            "recommendations": [],
        }

        result = markdown_builder.build_markdown(extract, analysis, security_dict)

        # Should have escaped the backticks and backslashes
        assert "\\`" in result.content or "backticks" in result.content

    def test_build_markdown_truncates_long_text(
        self,
        markdown_builder,
    ):
        """Extracted text longer than 500 chars is truncated."""
        from servo_skull.models import DocumentExtract, GapAnalysis

        long_text = "a" * 1000  # Much longer than 500

        extract = DocumentExtract(
            document_id="test-123",
            original_filename="test.pdf",
            document_type="pdf",
            extracted_text=long_text,
            confidence=0.95,
            extraction_tool="pymupdf4llm",
        )

        analysis = GapAnalysis(
            document_id="test-123",
            gaps={},
            risk_flags=[],
            security_flags=[],
            hallucination_score=0.0,
            grounding_notes="Test",
            llm_model="gemma4-26b-a4b-moe",
        )

        security_dict = {
            "security_issues": [],
            "misinformation_risks": [],
            "ai_watermarks": [],
            "fraud_indicators": [],
            "recommendations": [],
        }

        result = markdown_builder.build_markdown(extract, analysis, security_dict)

        # Should indicate truncation
        assert "truncated" in result.content.lower() or len(result.content) < len(long_text)


class TestMarkdownBuilderIntegration:
    """Integration tests combining multiple features."""

    def test_build_markdown_full_workflow(
        self,
        markdown_builder,
        sample_document_extract,
        sample_gap_analysis,
        sample_security_dict,
    ):
        """Full workflow produces coherent markdown."""
        result = markdown_builder.build_markdown(
            sample_document_extract,
            sample_gap_analysis,
            sample_security_dict,
        )

        # All key components should be present
        assert result.document_id
        assert result.frontmatter
        assert result.content
        assert result.output_path
        assert result.generated_timestamp
        assert result.binaric_cant_footer

        # Content should be coherent
        assert len(result.content) > 500
        assert "# " in result.content  # Should have headings

    def test_build_markdown_gap_counts_in_footer(
        self,
        markdown_builder,
        sample_document_extract,
        sample_gap_analysis,
        sample_security_dict,
    ):
        """Gap counts in footer match content."""
        result = markdown_builder.build_markdown(
            sample_document_extract,
            sample_gap_analysis,
            sample_security_dict,
        )

        # Count gaps in analysis
        total_gaps = sum(
            len(gaps) for gaps in sample_gap_analysis.gaps.values()
            if isinstance(gaps, list)
        )

        # Check footer reflects this count
        footer_gap_count = result.binaric_cant_footer["gaps"]["count"]
        assert footer_gap_count == total_gaps

    def test_build_markdown_risk_counts_in_footer(
        self,
        markdown_builder,
        sample_document_extract,
        sample_gap_analysis,
        sample_security_dict,
    ):
        """Risk counts in footer match content."""
        result = markdown_builder.build_markdown(
            sample_document_extract,
            sample_gap_analysis,
            sample_security_dict,
        )

        # Risk count should match
        footer_risk_count = result.binaric_cant_footer["risks"]["count"]
        expected_count = len(sample_gap_analysis.risk_flags)
        assert footer_risk_count == expected_count
