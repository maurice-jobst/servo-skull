"""SQLite FTS5 RAG Ingest and Indexing logic for Servo-Skull."""
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Generator, Optional

import click

from servo_skull._utils import setup_logging
from servo_skull.models import DocumentExtract

logger = setup_logging(__name__)

DB_NAME = "rag_index.sqlite"


def get_db_path(spool_dir: Optional[str] = None) -> Path:
    """Resolve database file path in the spool directory."""
    if spool_dir:
        base_path = Path(spool_dir)
    else:
        # Default fallback
        base_path = Path("./workspace/spool")
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path / DB_NAME


def init_db(db_path: Path) -> None:
    """Initialize the SQLite database with the FTS5 virtual table."""
    conn = sqlite3.connect(db_path)
    try:
        # Enable WAL mode for concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        
        # Create FTS5 virtual table
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks USING fts5(
                document_id,
                filename,
                chunk_index,
                content,
                metadata,
                tokenize='unicode61'
            );
        """)
        conn.commit()
    finally:
        conn.close()


def chunk_text_by_paragraphs(text: str) -> Generator[tuple[str, str], None, None]:
    """
    Split text into chunks by double-newlines, keeping headings aligned.
    Yields (chunk_text, last_heading).
    """
    paragraphs = text.split("\n\n")
    current_heading = "Root"
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # Detect markdown heading
        if paragraph.startswith("#"):
            # Clean heading text
            current_heading = paragraph.lstrip("#").strip().split("\n")[0]
            
        yield paragraph, current_heading


def index_extract_file(db_path: Path, extract_file: Path) -> int:
    """Index a single extract file into the FTS5 database, removing existing entries first."""
    data = json.loads(extract_file.read_text(encoding="utf-8"))
    
    if "extract" in data:
        extract = DocumentExtract(**data["extract"])
    else:
        extract = DocumentExtract(**data)
        
    if not extract.extracted_text:
        logger.warning(f"No text extracted for document {extract.document_id}; skipping indexing.")
        return 0
        
    conn = sqlite3.connect(db_path)
    try:
        # 1. Incremental update: Delete old chunks for this document
        conn.execute(
            "DELETE FROM document_chunks WHERE document_id = ? OR filename = ?;",
            (extract.document_id, extract.original_filename)
        )
        
        # 2. Slice text and insert chunks
        chunks_inserted = 0
        for chunk_idx, (chunk_text, last_heading) in enumerate(chunk_text_by_paragraphs(extract.extracted_text)):
            metadata_dict = {
                "last_heading": last_heading,
                "confidence": extract.confidence,
                "extraction_tool": extract.extraction_tool
            }
            conn.execute(
                """
                INSERT INTO document_chunks(document_id, filename, chunk_index, content, metadata)
                VALUES (?, ?, ?, ?, ?);
                """,
                (
                    extract.document_id,
                    extract.original_filename,
                    chunk_idx,
                    chunk_text,
                    json.dumps(metadata_dict)
                )
            )
            chunks_inserted += 1
            
        conn.commit()
        logger.info(f"Indexed {chunks_inserted} chunks from {extract.original_filename} (ID: {extract.document_id})")
        return chunks_inserted
    finally:
        conn.close()


def query_index(db_path: Path, query_str: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Query the FTS5 database and return ranked results using BM25."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # FTS5 bm25() score sorts ASC (lower/more negative is better)
        cursor = conn.execute(
            """
            SELECT document_id, filename, chunk_index, content, metadata, bm25(document_chunks) as rank
            FROM document_chunks
            WHERE document_chunks MATCH ?
            ORDER BY rank ASC
            LIMIT ?;
            """,
            (query_str, top_k)
        )
        results = []
        for row in cursor.fetchall():
            results.append({
                "document_id": row["document_id"],
                "filename": row["filename"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]),
                "rank": round(float(row["rank"]), 4)
            })
        return results
    finally:
        conn.close()


@click.command()
@click.argument("extract_path", type=click.Path(exists=True))
@click.option(
    "--spool-dir",
    type=click.Path(),
    default="./workspace/spool/",
    help="Spool directory path containing SQLite database.",
)
def index_cmd(extract_path: str, spool_dir: str) -> None:
    """Index an extract file or directory of extract files into the RAG SQLite FTS5 database."""
    db_path = get_db_path(spool_dir)
    init_db(db_path)
    
    extract_path_obj = Path(extract_path)
    
    if extract_path_obj.is_dir():
        extract_files = list(extract_path_obj.glob("*.extract.json"))
        if not extract_files:
            click.echo(f"No *.extract.json files found in directory {extract_path_obj}")
            return
            
        click.echo(f"✓ Found {len(extract_files)} extract files to index in {extract_path_obj}")
        total_chunks = 0
        for f in extract_files:
            try:
                chunks = index_extract_file(db_path, f)
                total_chunks += chunks
                click.echo(f"  + Indexed: {f.name} ({chunks} chunks)")
            except Exception as e:
                click.echo(f"  x Failed: {f.name} ({type(e).__name__}: {e})")
        click.echo(f"✓ Completed indexing. Total chunks added: {total_chunks}")
    else:
        try:
            chunks = index_extract_file(db_path, extract_path_obj)
            click.echo(f"✓ Indexed single file: {extract_path_obj.name} ({chunks} chunks)")
        except Exception as e:
            click.echo(f"✗ Indexing failed for {extract_path_obj.name}: {e}")


@click.command()
@click.argument("query_str", type=str)
@click.option(
    "--spool-dir",
    type=click.Path(),
    default="./workspace/spool/",
    help="Spool directory path containing SQLite database.",
)
@click.option(
    "--limit",
    type=int,
    default=5,
    help="Maximum number of search results to return (default: 5).",
)
def search_cmd(query_str: str, spool_dir: str, limit: int) -> None:
    """Search the SQLite FTS5 RAG database and output a formatted markdown table."""
    db_path = get_db_path(spool_dir)
    if not db_path.exists():
        click.echo(f"✗ RAG Database not found at {db_path}. Please run index command first.")
        return
        
    try:
        results = query_index(db_path, query_str, top_k=limit)
        if not results:
            click.echo(f"No results found matching query: '{query_str}'")
            return
            
        click.echo(f"### Search Results for Query: `{query_str}`\n")
        click.echo("| Relevance | Source File | Section/Heading | Snippet |")
        click.echo("|---|---|---|---|")
        
        for r in results:
            filename = r["filename"]
            heading = r["metadata"].get("last_heading", "n/a")
            rank = r["rank"]
            
            # Clean snippet for markdown representation (replace newlines with spaces)
            snippet = r["content"].replace("\n", " ").replace("\r", "")
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            # Escape pipes to keep table formatting clean
            snippet = snippet.replace("|", "\\|")
            
            click.echo(f"| {rank} | {filename} | {heading} | {snippet} |")
    except Exception as e:
        click.echo(f"✗ Search failed: {e}")
