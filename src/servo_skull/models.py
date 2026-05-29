"""Pydantic models for Servo-Skull Phase 1 pipeline."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class PIIClassification(BaseModel):
    """Output of a PIIVault NER scan on a document."""

    document_id: str
    contains_pii: bool = False
    entity_types_found: list[str] = Field(
        default_factory=list,
        description="GLiNER label types detected, e.g. ['person', 'organization']",
    )
    entity_count: int = 0
    forced_local: bool = Field(
        default=False,
        description="True when PII is present and content must not leave local inference.",
    )


class TaskRoute(BaseModel):
    """Routing decision emitted by the task classifier for a single document."""

    task_type: Literal[
        "extraction",
        "rule_synthesis",
        "gap_analysis",
        "summary",
        "security_check",
        "deep_research",
    ]
    complexity: Literal["low", "medium", "high"] = "medium"
    requires_grounding: bool = False
    pii_present: bool = False
    backend: Literal["mlx", "ollama", "gemini"] = "ollama"
    model_tier: Literal["fast", "reason", "local"] = "local"
    estimated_tokens: int = 0

    class Config:
        json_schema_extra = {
            "example": {
                "task_type": "rule_synthesis",
                "complexity": "high",
                "requires_grounding": True,
                "pii_present": False,
                "backend": "gemini",
                "model_tier": "reason",
                "estimated_tokens": 4200,
            }
        }


class DocumentExtract(BaseModel):
    """Phase 1 output: ground truth from deterministic extraction."""
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    original_filename: str
    document_type: Literal["pdf", "office", "image", "audio", "text"]
    extracted_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    structure: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    warnings: list[str] = Field(default_factory=list)
    extraction_tool: str

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "d7a3e2c1-4f9b-47e8-b2d3-5e1f8a9c6d2b",
                "original_filename": "tender.pdf",
                "document_type": "pdf",
                "extracted_text": "REQUEST FOR PROPOSAL...",
                "metadata": {"page_count": 47, "author": "Globex"},
                "confidence": 0.94,
                "extraction_tool": "pymupdf4llm",
            }
        }


class GapAnalysis(BaseModel):
    """Phase 2 output: 4-D matrix gap analysis + security checks."""
    document_id: str
    gaps: dict[str, list[dict[str, Any]]] = Field(
        default_factory=dict,
        description="gaps[dimension] = [{gap, severity, context}, ...]"
    )
    risk_flags: list[dict[str, Any]] = Field(default_factory=list)
    security_flags: list[str] = Field(default_factory=list)
    hallucination_score: float = Field(ge=0.0, le=1.0, description="0.0=hallucinated, 1.0=grounded")
    grounding_notes: str = ""
    analysis_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    llm_model: str

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "d7a3e2c1-4f9b-47e8-b2d3-5e1f8a9c6d2b",
                "gaps": {
                    "stakeholder": [
                        {"gap": "GDPR data residency not specified", "severity": "high"}
                    ]
                },
                "risk_flags": [{"risk": "Non-compliant with SOC 2", "domain": "operational"}],
                "security_flags": [],
                "hallucination_score": 0.15,
                "llm_model": "gemma4-26b-a4b-moe",
            }
        }


class RichMarkdown(BaseModel):
    """Phase 3 output: RAG-optimized Markdown."""
    document_id: str
    frontmatter: dict[str, Any]
    content: str
    see_also: list[tuple[str, str]] = Field(default_factory=list)
    scriptorum_refs: list[str] = Field(default_factory=list)
    binaric_cant_footer: dict[str, Any] = Field(default_factory=dict)
    output_path: str
    generated_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "d7a3e2c1-4f9b-47e8-b2d3-5e1f8a9c6d2b",
                "frontmatter": {
                    "id": "doc-d7a3e2c1",
                    "type": "specification",
                    "domain": "product-management",
                    "summary": "Acme Platform Requirements Spec",
                },
                "content": "# Acme Platform...",
                "see_also": [("GDPR Compliance", "compliance/gdpr.md"), ("Technical Architecture", "technical/architecture.md")],
                "scriptorum_refs": ["01_HOT/specs/acme-platform", "03_CODEX/standards/iso27001"],
                "binaric_cant_footer": {"doc_id": "d7a3e2c1", "gaps_count": 3, "hallucination_score": 0.15},
                "output_path": "specs/2026-05-23-acme-platform.md",
            }
        }


class Requirement(BaseModel):
    """Structured rule/requirement extracted from document."""
    rule_id: str = Field(description="Unique rule identifier, e.g., Rule_5_1 or Req_2_17")
    title: str = Field(description="Short descriptive title of the requirement")
    description: str = Field(description="Detailed text of the requirement")
    original_clause: str = Field(description="Exact snippet or clause from the source text")
    source_reference: str = Field(description="Page number or section reference in the document")
    status: str = Field(default="draft", description="Status of the requirement")


@dataclass
class BenchStats:
    """Timing, throughput and cost statistics from a single LLM call.

    TTFT (time-to-first-token) is the prefill duration — how long the model
    takes to digest the input before generating any output. Prefill speed is
    especially important for large-context ingestion tasks.
    """
    text: str
    ttft_s: float             # seconds from request to first output token (prefill duration)
    total_s: float            # total wall-clock time
    prompt_tokens: int        # input tokens processed
    output_tokens: int        # tokens generated
    tokens_per_s: float       # output generation throughput (post-first-token)
    prefill_tokens_per_s: float  # input processing rate (prompt ingestion speed)
    cost_usd: float | None = None  # None for local inference; set for cloud providers


class DeepResearchReport(BaseModel):
    """Output of a Gemini Deep Research query."""
    query: str
    model: str
    domain: str = "general"
    report_text: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    research_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    duration_seconds: float = 0.0

    class Config:
        json_schema_extra = {
            "example": {
                "query": "data retention requirements for SaaS platforms",
                "model": "deep-research-preview-04-2026",
                "domain": "compliance-review",
                "report_text": "# Data Retention Overview\n...",
                "sources": [
                    {"title": "GDPR Article 5", "url": "https://gdpr-info.eu/art-5-gdpr/"}
                ],
                "duration_seconds": 42.5,
            }
        }

