<div align="center">

# ✙ Servo-Skull

**A local-first document scoping engine.**
Turn unstructured specs, briefings, and entropic media into structured, grounded, citation-backed gap analyses — without your data ever leaving your machine.

`extract → ground → markdown`

[![Python](https://img.shields.io/badge/Python-3.11%2B-C05640?style=flat-square&logo=python&logoColor=B59963&labelColor=16161A)](#install)
[![Tests](https://img.shields.io/badge/tests-358-B59963?style=flat-square&labelColor=C05640)](#install)
[![License](https://img.shields.io/badge/license-MIT-B59963?style=flat-square&labelColor=C05640)](LICENSE)
[![Local-first](https://img.shields.io/badge/AI-Local--first%20·%20Air--gapped-6A1B9A?style=flat-square)](#)

</div>

---

## What it does

Servo-Skull ingests messy briefs — RFPs, spec sheets, meeting recordings, chat transcripts, scanned PDFs — and outputs a structured, **citation-grounded** analysis that highlights **gaps, compliance risks, and self-contradictions**, scored against a domain codex you supply. Every claim traces back to the source text, so no model invents what a document said. It runs entirely against local models (Ollama / MLX), with an optional cloud fallback.

```
                 ┌────────────────────────────────────────────────┐
   document ───► │   Extractor  (layout-preserving ingestion)    │ ──► DocumentExtract (JSON)
                 └────────────────────────────────────────────────┘
                                         │
                 ┌───────────────────────▼────────────────────────┐      ┌─────────────────┐
                 │       Grounder  (hallucination scoring)        │ ◄─── │  domain codex   │
                 └────────────────────────────────────────────────┘      └─────────────────┘
                                         │  (local LLM — Ollama / MLX)
                 ┌───────────────────────▼────────────────────────┐
                 │            GapAnalysis  (Pydantic)             │
                 └────────────────────────────────────────────────┘
                                         │
                 ┌───────────────────────▼────────────────────────┐
                 │         RAG-optimized Markdown output          │
                 └────────────────────────────────────────────────┘
```

---

## What it ingests

Deterministic, layout-aware parsing across structured files and entropic media:

- **PDFs** — high-fidelity, layout-aware extraction (`pymupdf4llm`).
- **Images** — local OCR (`tesseract`), or a vision-language model for charts, receipts, and hand-written diagrams.
- **Audio** — local speech-to-text (`whisper`) for meeting recordings (`.mp3/.wav/.m4a/.flac/.mp4/.mov`).
- **Chat transcripts & JSON** — detects role/sender threads and formats clean markdown.
- **Office documents** — `.docx / .xlsx / .pptx / .csv`.

---

## Install

The ingestion layer lives in its own package, [`lexmechanic`](https://github.com/maurice-jobst/lexmechanic) — `uv sync` pulls it automatically.

```bash
git clone https://github.com/maurice-jobst/servo-skull.git
cd servo-skull
uv sync
uv run pytest        # 358 tests across parsing + LLM chains
```

LLM routing is local-first by default — point it at an Ollama endpoint and supply your domain codex. A cloud fallback fires only if the local model is unavailable *and* a key is present.

---

## The name

The Mechanicus naming is deliberate, not decoration: a servo-skull is a hovering construct that reads, scans, and scribes so its operator's attention stays free. That's exactly the job. The skull does the reading so your mind stays pure. ✙

---

<div align="center">
<sub>Part of a local-first, sovereign-AI toolkit by <a href="https://github.com/maurice-jobst">Maurice Jobst</a> · MIT licensed</sub>
</div>
