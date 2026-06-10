# AGENTS.md — servo-skull

Local-first document scoping engine: specs → citation-grounded gap analysis, hallucination-scored on local LLMs · Federation role: engine · Sync-zone: private Gitea canonical + GitHub mirror.
Self-contained for ≤50k-token context; load named files on demand. Human-facing detail: [README.md](README.md).
Federation source of truth: the federation hub's `AGENTS.md` (`~/Projects/federation-tools/AGENTS.md`) — cite it, don't copy it.
> CLAUDE.md is a symlink to this file.

**What this workspace is.** Deterministic extract → LLM-grounded gap analysis pipeline: ingests RFPs/specs/transcripts/PDFs/images/audio, extracts via lexmechanic, scores gaps against a domain codex on a local LLM (Ollama/MLX), emits RAG-optimized markdown. Runs as systemd workers + spool watchers on air-gapped infra.

## Commands
```bash
uv sync && uv run pytest                                  # setup + suite (358 tests)
servo-skull-assimilate <file> --output-dir ./artifacts    # full pipeline (extract → ground → markdown)
servo-skull-worker --spool-dir ./workspace/spool --pipeline extraction
```

## §1 Repo map
- `src/servo_skull/_cli.py` — all entry points; telemetry → `workspace/scriptorum/telemetry.jsonl`
- `src/servo_skull/grounder.py` — 4-D gap matrix (Stakeholder × Methodology × Component × Compliance), hallucination score 0–1
- `src/servo_skull/pii_vault.py` — GLiNER NER; pseudonymize → LLM → rehydrate; session-scoped, in-memory only
- `config/providers.toml` — routing: primary local Ollama; cloud fallback only on timeout + env key present

## §0 Non-negotiable invariants
1. Local-first: Ollama/MLX is primary; cloud fallback fires only on primary timeout AND an env API key — never by default.
2. PII gate is mandatory: every extract passes `PIIVault.scan()` before LLM dispatch; PII-bearing docs are forced local regardless of routing config.
3. Extraction is deterministic (lexmechanic); the LLM grounds and scores, it never extracts.

## §3 Decision rules
- *Provider down?* → the fallback chain in `llm_client.py` is the intended degrade; don't bypass it.
- *Hollow gap analysis (all gaps scored low)?* → the codex is unstructured; fix its `## Section` structure, not the prompt.

## §5 Failure modes & known state
- GLiNER load failure is non-fatal — the vault logs a warning and skips scanning, so PII could reach a cloud fallback in that state. Treat as page-worthy, not ignorable.
- SQLite spool contention: 3 retries / 1s delay; scale with per-worker spool dirs, not bigger locks.

## §6 Hard "never" list
- Never serialize the PII vault. Never skip the PII gate. Never commit extracts/analyses.
- Never lean on cloud fallback at scale — fix local inference instead.
