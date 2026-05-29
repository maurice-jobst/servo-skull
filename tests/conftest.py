"""Pytest configuration and fixtures for servo-skull."""
import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from servo_skull.models import DocumentExtract, GapAnalysis


@pytest.fixture
def tmp_spool(tmp_path):
    """Create a temporary spool directory structure."""
    spool_dir = tmp_path / "spool"
    (spool_dir / "staging").mkdir(parents=True)
    (spool_dir / "pending").mkdir(parents=True)
    (spool_dir / "processing").mkdir(parents=True)
    (spool_dir / "done").mkdir(parents=True)
    (spool_dir / "failed").mkdir(parents=True)
    return spool_dir


@pytest.fixture
def tmp_inbox(tmp_path):
    """Create temporary inbox directory."""
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    return inbox_dir


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a dummy PDF file for testing."""
    pdf_path = tmp_path / "sample.pdf"
    # Write minimal PDF header (not a real PDF, but enough to test path handling)
    pdf_path.write_bytes(b"%PDF-1.4\n%dummy content")
    return pdf_path


@pytest.fixture
def sample_docx(tmp_path):
    """Create a dummy DOCX file for testing."""
    docx_path = tmp_path / "sample.docx"
    # DOCX is a ZIP file with minimal structure
    import zipfile
    with zipfile.ZipFile(docx_path, 'w') as zf:
        zf.writestr("word/document.xml", "<w:document></w:document>")
    return docx_path


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a sample text file."""
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("This is sample text content.")
    return txt_path


@pytest.fixture
def mock_llm_client():
    """Mock LLMClient for testing."""
    mock = MagicMock()
    mock.ping.return_value = "LLMClient[analyzer→local_gemma] ok (config-only)"
    mock.chat.return_value = json.dumps({
        "gaps": {
            "stakeholder": [
                {"gap": "Test gap", "severity": "high", "context": "Test context"}
            ]
        },
        "risk_flags": [],
        "grounding_notes": "All grounded.",
    })
    return mock


@pytest.fixture
def sample_document_extract():
    """Create a sample DocumentExtract for testing."""
    return DocumentExtract(
        document_id="test-doc-123",
        original_filename="test.pdf",
        document_type="pdf",
        extracted_text="This is extracted text from a PDF document.",
        metadata={"page_count": 5, "author": "Test Author"},
        structure={"headings": ["Title", "Section 1"]},
        confidence=0.95,
        extraction_tool="pymupdf4llm",
    )


@pytest.fixture
def sample_gap_analysis():
    """Create a sample GapAnalysis for testing."""
    return GapAnalysis(
        document_id="test-doc-123",
        gaps={
            "stakeholder": [
                {"gap": "Missing legal review", "severity": "high", "context": "Legal clearance required"}
            ],
            "compliance": [
                {"gap": "GDPR not addressed", "severity": "high", "context": "Data residency unclear"}
            ],
        },
        risk_flags=[
            {"flag": "timeline_aggressive", "severity": "high", "detail": "12-month delivery"}
        ],
        security_flags=[],
        hallucination_score=0.12,
        grounding_notes="Analysis grounded against extracted text.",
        llm_model="gemma4-26b-a4b-moe",
    )


@pytest.fixture
def spool_db(tmp_path):
    """Create a temporary SQLite spool database."""
    db_path = tmp_path / "jobs.sqlite"

    _INIT_PRAGMAS = """
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous  = NORMAL;
    PRAGMA busy_timeout = 5000;
    """

    _INIT_SCHEMA = """
    CREATE TABLE IF NOT EXISTS jobs (
      id            TEXT PRIMARY KEY,
      pipeline      TEXT NOT NULL,
      status        TEXT NOT NULL CHECK (status IN ('pending','processing','done','failed')),
      payload       TEXT,
      attempts      INTEGER NOT NULL DEFAULT 0,
      worker_id     TEXT,
      leased_until  TEXT,
      created_at    TEXT NOT NULL DEFAULT (datetime('now')),
      updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
      error         TEXT
    );
    """

    with sqlite3.connect(db_path) as conn:
        conn.executescript(_INIT_PRAGMAS)
        conn.executescript(_INIT_SCHEMA)
        conn.commit()

    return db_path
