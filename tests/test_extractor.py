"""Test document extraction."""
import pytest
from pathlib import Path
from servo_skull.extractor import Extractor
from servo_skull.models import DocumentExtract


def test_extract_text_file(sample_text_file):
    """Test extraction from plain text file."""
    extractor = Extractor()
    result = extractor.extract(sample_text_file)

    assert isinstance(result, DocumentExtract)
    assert result.document_type == "text"
    assert "sample text content" in result.extracted_text
    assert result.confidence > 0.9
    assert result.extraction_tool == "text_reader"


from unittest.mock import patch, MagicMock

def test_extract_pdf_file(sample_pdf):
    """Test extraction from PDF."""
    extractor = Extractor()
    with patch("pymupdf4llm.to_markdown", return_value="Mocked PDF content"):
        result = extractor.extract(sample_pdf)

    assert isinstance(result, DocumentExtract)
    assert result.document_type == "pdf"
    assert result.extraction_tool == "pymupdf4llm"
    assert isinstance(result.extracted_text, str)
    assert len(result.extracted_text) > 0  # Validate text was extracted
    assert result.extracted_text == "Mocked PDF content"


def test_extract_unsupported_file(tmp_path):
    """Test extraction from unsupported file type."""
    unsupported = tmp_path / "file.xyz"
    unsupported.write_text("content")

    extractor = Extractor()
    with pytest.raises(ValueError, match="Unsupported file type"):
        extractor.extract(unsupported)


def test_extract_nonexistent_file():
    """Test extraction from nonexistent file."""
    extractor = Extractor()
    with pytest.raises(FileNotFoundError):
        extractor.extract(Path("/nonexistent/file.pdf"))


def test_extract_metadata_preserved(sample_text_file):
    """Test that file metadata is preserved."""
    extractor = Extractor()
    result = extractor.extract(sample_text_file)

    assert result.original_filename == "sample.txt"
    assert "extraction_timestamp" in result.model_dump()
    assert result.warnings == []  # No warnings for simple text file


def test_extract_html_file(tmp_path):
    """Test HTML file extraction routing and conversion."""
    html_file = tmp_path / "sample.html"
    html_file.write_text("<h1>Confluence Page Title</h1><p>Some content here</p>", encoding="utf-8")

    extractor = Extractor()
    with patch("markitdown.MarkItDown.convert") as mock_convert:
        mock_result = MagicMock()
        mock_result.text_content = "# Confluence Page Title\n\nSome content here"
        mock_convert.return_value = mock_result

        result = extractor.extract(html_file)

    assert isinstance(result, DocumentExtract)
    assert result.document_type == "office"
    assert result.extraction_tool == "markitdown"
    assert "# Confluence Page Title" in result.extracted_text
    assert result.original_filename == "sample.html"


def test_extract_htm_file(tmp_path):
    """Test HTM file extraction routing and conversion."""
    htm_file = tmp_path / "sample.htm"
    htm_file.write_text("<p>htm content</p>", encoding="utf-8")

    extractor = Extractor()
    with patch("markitdown.MarkItDown.convert") as mock_convert:
        mock_result = MagicMock()
        mock_result.text_content = "htm content"
        mock_convert.return_value = mock_result

        result = extractor.extract(htm_file)

    assert isinstance(result, DocumentExtract)
    assert result.document_type == "office"
    assert result.extraction_tool == "markitdown"
    assert result.extracted_text == "htm content"
    assert result.original_filename == "sample.htm"


def test_extractor_vlm_fallback(tmp_path):
    """Test that Extractor properly flags and degrades confidence on VLM fallback."""
    img_file = tmp_path / "sample.png"
    img_file.write_text("fake image data")

    extractor = Extractor()
    with patch("lexmechanic.Lexmechanic.convert") as mock_convert:
        mock_result = MagicMock()
        mock_result.text_content = "OCR text from fallback"
        mock_result.vlm_fallback = True
        mock_convert.return_value = mock_result

        result = extractor.extract(img_file)

    assert isinstance(result, DocumentExtract)
    assert result.document_type == "image"
    assert result.extraction_tool == "tesseract_ocr(vlm_fallback)"
    assert result.confidence == 0.60
    assert "Ollama VLM failed; fell back to local Tesseract OCR." in result.warnings
    assert result.extracted_text == "OCR text from fallback"


