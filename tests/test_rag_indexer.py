"""Unit tests for SQLite FTS5 RAG Indexer."""
import json
import sqlite3
from pathlib import Path

import pytest

from servo_skull.models import DocumentExtract
from servo_skull.rag_indexer import (
    chunk_text_by_paragraphs,
    get_db_path,
    index_extract_file,
    init_db,
    query_index,
)


def test_chunk_text_by_paragraphs():
    """Test chunk_text_by_paragraphs splits correctly and tracks headings."""
    sample_text = """# Main Title

This is paragraph one under Main Title.

## Sub Section

This is paragraph two under Sub Section.

Another paragraph under Sub Section.
"""
    chunks = list(chunk_text_by_paragraphs(sample_text))
    
    assert len(chunks) == 5  # Heading1, P1, Heading2, P2, P3
    
    # Check paragraph contents
    assert chunks[0][0] == "# Main Title"
    assert chunks[0][1] == "Main Title"
    
    assert chunks[1][0] == "This is paragraph one under Main Title."
    assert chunks[1][1] == "Main Title"
    
    assert chunks[2][0] == "## Sub Section"
    assert chunks[2][1] == "Sub Section"
    
    assert chunks[3][0] == "This is paragraph two under Sub Section."
    assert chunks[3][1] == "Sub Section"
    
    assert chunks[4][0] == "Another paragraph under Sub Section."
    assert chunks[4][1] == "Sub Section"


def test_init_db(tmp_path):
    """Test init_db creates table schema correctly."""
    db_path = tmp_path / "test_rag.sqlite"
    init_db(db_path)
    
    assert db_path.exists()
    
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='document_chunks';")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "document_chunks"
    finally:
        conn.close()


def test_index_extract_file_and_query(tmp_path):
    """Test indexing extracts and querying FTS5 database."""
    db_path = tmp_path / "test_rag.sqlite"
    init_db(db_path)
    
    extract = DocumentExtract(
        document_id="test-doc-999",
        original_filename="sample_guide.md",
        document_type="text",
        extracted_text="""# Reader Specifications

All edge devices must support offline cache invalidation.

## Memory Requirements

At least 512MB RAM is required on the board.
""",
        confidence=0.99,
        extraction_tool="test_parser"
    )
    
    # Write mock extract to temp file
    extract_file = tmp_path / "test-doc-999.extract.json"
    extract_file.write_text(extract.model_dump_json(), encoding="utf-8")
    
    # Index file
    chunks_added = index_extract_file(db_path, extract_file)
    assert chunks_added == 4  # Heading1, P1, Heading2, P2
    
    # Query index for terms
    results = query_index(db_path, "cache")
    assert len(results) >= 1
    assert results[0]["document_id"] == "test-doc-999"
    assert "offline cache invalidation" in results[0]["content"]
    assert results[0]["metadata"]["last_heading"] == "Reader Specifications"
    
    # Query memory RAM term
    results_mem = query_index(db_path, "RAM")
    assert len(results_mem) >= 1
    assert "512MB" in results_mem[0]["content"]
    assert results_mem[0]["metadata"]["last_heading"] == "Memory Requirements"


def test_incremental_updates(tmp_path):
    """Test that indexing deletes old entries for same doc/filename."""
    db_path = tmp_path / "test_rag.sqlite"
    init_db(db_path)
    
    extract1 = DocumentExtract(
        document_id="test-doc-unique",
        original_filename="doc.md",
        document_type="text",
        extracted_text="Initial paragraph text.",
        confidence=0.99,
        extraction_tool="test"
    )
    extract_file = tmp_path / "doc.extract.json"
    extract_file.write_text(extract1.model_dump_json(), encoding="utf-8")
    
    index_extract_file(db_path, extract_file)
    
    # Verify indexed
    results1 = query_index(db_path, "Initial")
    assert len(results1) == 1
    
    # Re-index with same doc_id but updated content
    extract2 = DocumentExtract(
        document_id="test-doc-unique",
        original_filename="doc.md",
        document_type="text",
        extracted_text="Updated replacement text here.",
        confidence=0.99,
        extraction_tool="test"
    )
    extract_file.write_text(extract2.model_dump_json(), encoding="utf-8")
    
    index_extract_file(db_path, extract_file)
    
    # Verify old content is deleted
    results_old = query_index(db_path, "Initial")
    assert len(results_old) == 0
    
    # Verify new content is found
    results_new = query_index(db_path, "Updated")
    assert len(results_new) == 1
    assert results_new[0]["content"] == "Updated replacement text here."
