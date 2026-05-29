<div align="center">

# ✙ Servo-Skull

**A local-first document scoping engine.**
Turn unstructured specifications into a grounded, four-dimensional gap analysis —
without your documents ever leaving the machine.

`extract → ground → markdown → codex`

</div>

---

## What it does

You feed Servo-Skull a messy document — an RFP, a requirements spec, a statement
of work — and it returns a structured, citation-grounded analysis of what the
document is **missing, ambiguous about, or self-contradictory on**, scored
against a domain codex you control.

It is built for the pre-project scoping moment: the point where a vague brief has
to become a defensible scope, and the cost of an unasked question is highest.

### The pipeline

| Stage | Verb | What happens |
|-------|------|--------------|
| **1. Extract** | `extract` / `assimilate` | Deterministic-first text + structure extraction from PDF, DOCX, XLSX, PPTX, images (OCR), audio (transcription), and plain text → schema-validated JSON. |
| **2. Ground** | `grounder` / `cogitate` | 4-dimensional gap analysis against your domain codex, with a hallucination score that compares every LLM claim back against the source text. |
| **3. Markdown** | `markdown` / `inscribe` | A RAG-optimized markdown "twin" of the analysis — readable by humans and retrievable by machines. |

*(Each stage has two names: a plain technical verb and a thematic one. They are
the same command — use whichever you prefer.)*

### The four dimensions

Every document is scored across four intersecting axes, defined by your codex:

1. **Stakeholder Ecosystem** — Legal/GRC, Sales, Engineering, Finance, Operations, Partners
2. **Delivery Methodology Stack** — Governance / Execution / Operations layers
3. **Affected Components** — Data / Service / Client / Integration layers
4. **Compliance Stack** — GDPR, SOC 2 / ISO 27001, WCAG 2.2, and any domain-specific standard

The codex is just a text file. Swap it and the same engine scopes a different
domain — that is the entire point of the design.

## Why local-first

Scoping documents are exactly the documents you cannot paste into a public API:
contracts, customer data, unreleased plans. Servo-Skull is built around that
constraint.

- **Local by default.** The primary LLM provider is a local, OpenAI-compatible
  endpoint (Ollama / MLX). Cloud is a *fallback* that only fires on local
  failure, and only if you've supplied an API key.
- **PII vault gate.** [`pii_vault.py`](src/servo_skull/pii_vault.py) runs a
  named-entity scan over every document. Any document carrying personal data is
  pinned to local inference and pseudonymized before dispatch, then rehydrated
  on the response — so PII never crosses the network boundary even on fallback.
- **Hallucination scoring.** Grounding isn't trusted blindly: every gap is rated
  0.0–1.0 for how well it's anchored in the source text.

## Architecture

```
                ┌──────────────┐
   document ──► │  extractor   │ ──► DocumentExtract (JSON)
                └──────────────┘
                       │
                ┌──────▼───────┐     ┌──────────────┐
                │   grounder   │ ◄── │ domain codex │
                └──────────────┘     └──────────────┘
                       │  (LLMClient → local | fallback,
                       │   PII vault, security checker)
                ┌──────▼───────┐
                │   GapAnalysis │ ──► markdown_builder ──► rich markdown
                └──────────────┘
```

| Module | Responsibility |
|--------|----------------|
| `extractor.py` | Multi-format deterministic extraction (PDF/Office/image/audio/text) |
| `grounder.py` | 4-D gap analysis orchestration + hallucination scoring |
| `markdown_builder.py` | RAG-optimized markdown synthesis |
| `models.py` | Pydantic schemas: `DocumentExtract`, `GapAnalysis`, `RichMarkdown` |
| `llm_client.py` / `llm_providers.py` | Provider chain with automatic fallback (Ollama, OpenAI, Anthropic, Gemini, MLX) |
| `pii_vault.py` | NER-based PII detection, pseudonymization, rehydration |
| `security_checker.py` | Injection / misinformation / AI-watermark / fraud screening |
| `rules_extractor.py` | Extract structured, atomic requirements from a document |
| `rag_indexer.py` | Local index + search over processed artifacts |
| `spool_coordinator.py` / `task_router.py` | Spool-watch + worker queue for batch/daemon processing |
| `deep_research.py` | Optional research-augmentation pass |

## Install

Requires Python ≥ 3.11. [`uv`](https://github.com/astral-sh/uv) recommended.

```bash
git clone https://github.com/maurice-jobst/servo-skull.git
cd servo-skull
uv sync                      # or: pip install -e .
```

Optional Apple-Silicon MLX backend:

```bash
uv sync --extra mlx          # or: pip install -e ".[mlx]"
```

You'll also need a local LLM endpoint. The default config points at
[Ollama](https://ollama.com):

```bash
ollama pull gemma4:26b       # or edit config/providers.toml
```

## Quickstart

A complete, fully synthetic walkthrough lives in [`demo/`](demo/):

```bash
# 1. extract
servo-skull-extract demo/input/sample-spec.md --output-dir artifacts

# 2. ground against the domain codex
servo-skull-grounder artifacts/sample-spec.extract.json \
    demo/codex/domain-codex.md --output-dir artifacts

# 3. render the markdown twin
servo-skull-markdown artifacts/sample-spec.extract.json \
    artifacts/sample-spec.analysis.json --output-dir artifacts
```

See [`demo/README.md`](demo/README.md) for what each stage produces and the gaps
the sample is designed to surface.

## Configuration

LLM routing lives in [`config/providers.toml`](config/providers.toml) — set your
primary (local) and fallback (cloud) providers there. Environment variables are
expanded with `${VAR}` syntax, so secrets stay out of the file:

```toml
[routing]
primary  = "local_gemma"
fallback = "cloud_openai"
```

## Tests

```bash
uv run pytest                # 19 test modules covering the full pipeline
```

## Status & license

`v0.1.0` — early but exercised end-to-end. Released under the [MIT License](LICENSE).

---

<div align="center">
<sub>The skull does the reading so you don't have to. ✙</sub>
</div>
