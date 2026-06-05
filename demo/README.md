# Servo-Skull Demo — Walkthrough

This demo runs the full **extract → ground → markdown** pipeline against a
fully synthetic requirements document. No real customer, vendor, or contract
data is involved — `sample-spec.md` and `domain-codex.md` are invented for
illustration.

```
demo/
├── input/
│   └── sample-spec.md     # synthetic RFP with deliberate scoping gaps
├── codex/
│   └── domain-codex.md    # the grounding framework gaps are scored against
└── README.md              # this file
```

## What the demo shows

`sample-spec.md` is a plausible "Customer Feedback Portal" RFP that is silent on
several things a delivery team would need: it never states a data-retention
policy, never names an accessibility target for its public web form, never names
the identity-provider protocol, and never defines SLA penalty terms. The codex
encodes exactly those expectations, so the gap analysis surfaces them.

## Run it

From the repository root, with the package installed (`uv sync` or
`pip install -e .`):

```bash
# 1. EXTRACT — deterministic-first text + structure extraction → JSON
servo-skull-extract demo/input/sample-spec.md --output-dir artifacts

# 2. GROUND — 4-dimensional gap analysis against the domain codex
servo-skull-grounder artifacts/sample-spec.extract.json \
    demo/codex/domain-codex.md \
    --output-dir artifacts

# 3. MARKDOWN — render a RAG-optimized markdown twin of the analysis
servo-skull-markdown artifacts/sample-spec.extract.json \
    artifacts/sample-spec.analysis.json \
    --output-dir artifacts
```

Each stage writes to `artifacts/`:

| Stage | Output | Contents |
|-------|--------|----------|
| extract | `sample-spec.extract.json` | Extracted text, document type, confidence, extraction tool |
| ground | `sample-spec.analysis.json` | 4-D gaps, risk flags, security flags, hallucination score |
| markdown | `sample-spec.md` (rich) | Human + RAG-readable synthesis of the above |

## The four dimensions

Gap analysis scores the document across four intersecting axes (see the codex):

1. **Stakeholder Ecosystem** — Legal, Sales, Engineering, Finance, Operations, Partners
2. **Delivery Methodology Stack** — Governance / Execution / Operations layers
3. **Affected Components** — Data / Service / Client / Integration layers
4. **Compliance Stack** — GDPR, SOC 2 / ISO 27001, WCAG 2.2, domain-specific

## Local-first note

Grounding requires an LLM. Servo-Skull is built to point at a **local** endpoint
(e.g. an OpenAI-compatible server on `127.0.0.1`) by default, so document content
never leaves the machine. The **PII vault** gate (`pii_vault.py`) scans for
personal data and forces any PII-bearing document onto local inference. See the
top-level [README](../README.md) for configuration.
