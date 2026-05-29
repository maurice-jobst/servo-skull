"""Test Pydantic models."""
import pytest
from servo_skull.models import DocumentExtract, GapAnalysis, RichMarkdown


def test_document_extract_valid():
    """Test DocumentExtract with valid data."""
    extract = DocumentExtract(
        original_filename="test.pdf",
        document_type="pdf",
        extracted_text="Lorem ipsum",
        confidence=0.95,
        extraction_tool="pymupdf4llm",
    )
    assert extract.document_id  # UUID generated
    assert extract.original_filename == "test.pdf"
    assert extract.confidence == 0.95


def test_document_extract_confidence_bounds():
    """Test that confidence must be 0.0–1.0."""
    with pytest.raises(ValueError):
        DocumentExtract(
            original_filename="test.pdf",
            document_type="pdf",
            extracted_text="Lorem ipsum",
            confidence=1.5,  # Invalid
            extraction_tool="pymupdf4llm",
        )


def test_gap_analysis_valid():
    """Test GapAnalysis with valid data."""
    analysis = GapAnalysis(
        document_id="doc-123",
        gaps={"stakeholder": [{"gap": "Missing info", "severity": "high"}]},
        hallucination_score=0.15,
        llm_model="gemma4-26b-a4b-moe",
    )
    assert analysis.hallucination_score == 0.15


def test_rich_markdown_valid():
    """Test RichMarkdown with valid data."""
    markdown = RichMarkdown(
        document_id="doc-123",
        frontmatter={"id": "doc-123", "type": "tender"},
        content="# Title\n\nContent here.",
        output_path="compliance/2026-05-23-tender.md",
    )
    assert markdown.output_path == "compliance/2026-05-23-tender.md"


def test_document_extract_serialization():
    """Test JSON serialization/deserialization."""
    extract = DocumentExtract(
        original_filename="test.pdf",
        document_type="pdf",
        extracted_text="Lorem ipsum",
        confidence=0.95,
        extraction_tool="pymupdf4llm",
    )
    json_str = extract.model_dump_json()
    deserialized = DocumentExtract.model_validate_json(json_str)
    assert deserialized.original_filename == "test.pdf"


def test_gap_analysis_hallucination_bounds():
    """Test that hallucination_score must be 0.0–1.0."""
    with pytest.raises(ValueError):
        GapAnalysis(
            document_id="doc-123",
            hallucination_score=1.5,  # Invalid (out of bounds)
            llm_model="gemma4-26b-a4b-moe",
        )


def test_rich_markdown_see_also_tuples():
    """Test that see_also accepts list of (str, str) tuples."""
    markdown = RichMarkdown(
        document_id="doc-123",
        frontmatter={"id": "doc-123"},
        content="# Test",
        see_also=[("Title 1", "path/to/file1.md"), ("Title 2", "path/to/file2.md")],
        output_path="test.md",
    )
    assert len(markdown.see_also) == 2
    assert markdown.see_also[0] == ("Title 1", "path/to/file1.md")


def test_document_extract_missing_required_field():
    """Test that missing required fields are rejected."""
    with pytest.raises(ValueError):
        DocumentExtract(
            # Missing: original_filename (required)
            document_type="pdf",
            extracted_text="Lorem ipsum",
            confidence=0.95,
            extraction_tool="pymupdf4llm",
        )


def test_deep_research_report_requires_fields():
    from servo_skull.models import DeepResearchReport
    with pytest.raises(Exception):
        DeepResearchReport()  # missing required fields


def test_deep_research_report_valid():
    from servo_skull.models import DeepResearchReport
    report = DeepResearchReport(
        query="data retention policy",
        model="deep-research-preview-04-2026",
        report_text="# Summary\n...",
        sources=[{"title": "GDPR Article 5", "url": "https://gdpr-info.eu/art-5-gdpr/"}],
    )
    assert report.query == "data retention policy"
    assert len(report.sources) == 1
    assert report.domain == "general"  # default
