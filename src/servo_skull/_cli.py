"""CLI entry points for Servo-Skull Phase 1 extraction pipeline."""
import json
import logging
import os
import signal
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import click

from servo_skull._utils import setup_logging
from servo_skull.extractor import Extractor
from servo_skull.grounder import analyze_gaps
from servo_skull.markdown_builder import MarkdownBuilder
from servo_skull.models import DocumentExtract, GapAnalysis, RichMarkdown
from servo_skull.security_checker import SecurityChecker
from servo_skull.spool_coordinator import SpoolCoordinator
from servo_skull.sync import sync_file

logger = setup_logging(__name__)

# ──────────────────────────────────────────────────────────────────
# MECHANICUS-SEAL TELEMETRY — Benchmark baseline: M5 Max ~39s
# Records per-command execution times for long-term local LLM study
# ──────────────────────────────────────────────────────────────────
_M5_MAX_BASELINE_SECONDS = 39.0
_SCRIPTORUM_DIR = Path(os.environ.get("SERVO_SKULL_WORKSPACE", "./workspace")) / "scriptorum"


def _emit_telemetry(
    command: str,
    duration_seconds: float,
    document_id: str = "",
    extra: Optional[dict] = None,
) -> None:
    """Write Mechanicus-Seal telemetry to terminal and scriptorum/telemetry.jsonl."""
    perf_ratio = (duration_seconds / _M5_MAX_BASELINE_SECONDS) * 100
    baseline_cmp = f"{duration_seconds:.2f}s ({perf_ratio:.1f}% of M5 Max ~{_M5_MAX_BASELINE_SECONDS}s)"

    # Terminal output — Mechanicus-Seal style
    click.echo("")
    click.echo("✙ OMNISSIAH SCRIPTORUM — Servo-Skull MECHANICUS-SEAL TELEMETRY ✙")
    click.echo(f"  CMD     : {command}")
    click.echo(f"  DOC-ID  : {document_id or 'n/a'}")
    click.echo(f"  CYCLE   : {baseline_cmp}")
    if extra:
        for k, v in extra.items():
            click.echo(f"  {k:<8}: {v}")
    click.echo("  [01000011 00110100 00110001 00100000 01010100 01000011 01000001 01001001]")
    click.echo("")

    # Write to scriptorum/telemetry.jsonl for long-term benchmarking
    try:
        _SCRIPTORUM_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "document_id": document_id,
            "duration_seconds": round(duration_seconds, 3),
            "m5_max_baseline_seconds": _M5_MAX_BASELINE_SECONDS,
            "perf_ratio_pct": round(perf_ratio, 1),
            **(extra or {}),
        }
        telemetry_file = _SCRIPTORUM_DIR / "telemetry.jsonl"
        with telemetry_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as tel_err:
        logger.warning(f"Telemetry write failed (non-fatal): {tel_err}")


# Global state for graceful shutdown
_shutdown_event = threading.Event()


def _handle_sigterm(signum: int, frame: Any) -> None:
    """Handle SIGTERM signal for graceful shutdown."""
    _shutdown_event.set()
    logger.info("Shutdown requested via SIGTERM")


def run_extract(file_path: str, output_dir: str, verbose: bool = False, extract_rules_flag: bool = False) -> None:
    """Core extraction logic supporting piping, directory recursion, and rule extraction."""
    try:
        file_path_obj = Path(file_path)
        extractor = Extractor()

        is_stdout_pipe = output_dir == "-" or (not sys.stdout.isatty() and "PYTEST_CURRENT_TEST" not in os.environ)

        if file_path_obj.is_dir():
            supported_suffixes = set(extractor.SUPPORTED_TYPES.keys())
            files_to_process = []
            for p in file_path_obj.rglob("*"):
                if p.is_file() and p.suffix.lower() in supported_suffixes:
                    files_to_process.append(p)

            if not files_to_process:
                click.echo(f"No supported files found in directory {file_path_obj}")
                return

            if is_stdout_pipe:
                combined_results = []
                for f in files_to_process:
                    try:
                        extract_result = extractor.extract(f)
                        if extract_rules_flag:
                            from servo_skull.rules_extractor import extract_rules
                            rules = extract_rules(extract_result, verbose=verbose)
                            combined_results.append({
                                "extract": extract_result.model_dump(),
                                "rules": [r.model_dump() for r in rules]
                            })
                        else:
                            combined_results.append(extract_result.model_dump())
                    except Exception as file_err:
                        logger.error(f"Failed to extract file {f}: {file_err}")
                click.echo(json.dumps(combined_results, indent=2))
            else:
                output_dir_obj = Path(output_dir)
                output_dir_obj.mkdir(parents=True, exist_ok=True)
                click.echo(f"✓ Found {len(files_to_process)} supported files in directory: {file_path_obj}")

                for f in files_to_process:
                    try:
                        extract_result = extractor.extract(f)
                        output_file = output_dir_obj / f"{extract_result.document_id}.extract.json"
                        output_file.write_text(extract_result.model_dump_json(indent=2))
                        if verbose:
                            logger.info(f"Extraction complete: {f.name} → {extract_result.document_id}")
                        click.echo(f"  + Extracted: {f.name} → {output_file.name}")

                        if extract_rules_flag:
                            from servo_skull.rules_extractor import extract_rules, write_rule_markdown
                            rules = extract_rules(extract_result, verbose=verbose)
                            requirements_dir = output_dir_obj / "requirements"
                            for rule in rules:
                                write_rule_markdown(rule, requirements_dir)
                            click.echo(f"    - Extracted {len(rules)} rules to {requirements_dir}")
                    except Exception as file_err:
                        logger.error(f"Failed to extract file {f}: {file_err}")
                        click.echo(f"  x Failed: {f.name} ({type(file_err).__name__})")
            return

        # Single file processing (default flow)
        extract_result = extractor.extract(file_path_obj)

        if is_stdout_pipe:
            if extract_rules_flag:
                from servo_skull.rules_extractor import extract_rules
                rules = extract_rules(extract_result, verbose=verbose)
                combined = {
                    "extract": extract_result.model_dump(),
                    "rules": [r.model_dump() for r in rules]
                }
                click.echo(json.dumps(combined, indent=2))
            else:
                click.echo(extract_result.model_dump_json(indent=2))
        else:
            output_dir_obj = Path(output_dir)
            output_dir_obj.mkdir(parents=True, exist_ok=True)
            output_file = output_dir_obj / f"{extract_result.document_id}.extract.json"
            output_file.write_text(extract_result.model_dump_json(indent=2))

            if verbose:
                logger.info(f"Extraction complete: {file_path_obj.name} → {extract_result.document_id}")
            click.echo(f"✓ Extracted: {output_file}")


            if extract_rules_flag:
                from servo_skull.rules_extractor import extract_rules, write_rule_markdown
                rules = extract_rules(extract_result, verbose=verbose)
                requirements_dir = output_dir_obj / "requirements"
                for rule in rules:
                    write_rule_markdown(rule, requirements_dir)
                click.echo(f"✓ Extracted {len(rules)} rules to {requirements_dir}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid file type: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)


def run_grounder(extract_json: str, codex_file: str, output_dir: str, verbose: bool = False) -> None:
    """Core grounding logic supporting piping and directory-wide batch processing."""
    _start = time.time()
    try:
        codex_file_obj = Path(codex_file)
        codex_text = codex_file_obj.read_text()

        # Check if extract_json is a directory
        is_dir = False
        if extract_json != "-":
            extract_json_obj = Path(extract_json)
            if extract_json_obj.is_dir():
                is_dir = True

        if is_dir:
            extract_files = list(extract_json_obj.glob("*.extract.json"))
            if not extract_files:
                click.echo(f"No *.extract.json files found in directory {extract_json_obj}")
                return

            click.echo(f"✓ Found {len(extract_files)} extract files to analyze in {extract_json_obj}")
            output_dir_obj = Path(output_dir)
            output_dir_obj.mkdir(parents=True, exist_ok=True)

            is_stdout_pipe = output_dir == "-" or (not sys.stdout.isatty() and "PYTEST_CURRENT_TEST" not in os.environ)
            combined_results = []

            for f in extract_files:
                try:
                    extract_data = json.loads(f.read_text())
                    if "extract" in extract_data:
                        extract = DocumentExtract(**extract_data["extract"])
                    else:
                        extract = DocumentExtract(**extract_data)

                    if verbose:
                        logger.info(f"Starting gap analysis for {extract.document_id} ({f.name})")
                    analysis = analyze_gaps(extract, codex_text)
                    total_gaps = sum(len(v) for v in analysis.gaps.values())

                    if is_stdout_pipe:
                        combined_results.append({
                            "extract": extract.model_dump(),
                            "analysis": analysis.model_dump()
                        })
                    else:
                        output_file = output_dir_obj / f"{extract.document_id}.analysis.json"
                        output_file.write_text(analysis.model_dump_json(indent=2))
                        click.echo(f"  + Analyzed: {f.name} → {output_file.name} (gaps: {total_gaps}, score: {analysis.hallucination_score:.2f})")
                except Exception as file_err:
                    logger.error(f"Failed to analyze {f}: {file_err}")
                    click.echo(f"  x Failed: {f.name} ({type(file_err).__name__})")

            if is_stdout_pipe:
                click.echo(json.dumps(combined_results, indent=2))
            return

        # Single file/pipe processing (default flow)
        if extract_json == "-":
            raw_data = sys.stdin.read().strip()
            # Clean up potential binaric seals if any exist
            if "[ENGRAM_SEAL]::" in raw_data:
                raw_data = raw_data.split("[ENGRAM_SEAL]::")[1]
            extract_data = json.loads(raw_data)
        else:
            extract_json_obj = Path(extract_json)
            extract_data = json.loads(extract_json_obj.read_text())

        if "extract" in extract_data:
            extract = DocumentExtract(**extract_data["extract"])
        else:
            extract = DocumentExtract(**extract_data)

        # Perform gap analysis
        if verbose:
            logger.info(f"Starting gap analysis for {extract.document_id}")
        analysis = analyze_gaps(extract, codex_text)

        duration = time.time() - _start
        total_gaps = sum(len(v) for v in analysis.gaps.values())

        is_stdout_pipe = output_dir == "-" or (not sys.stdout.isatty() and "PYTEST_CURRENT_TEST" not in os.environ)
        if is_stdout_pipe:
            combined = {
                "extract": extract.model_dump(),
                "analysis": analysis.model_dump()
            }
            click.echo(json.dumps(combined, indent=2))
        else:
            output_dir_obj = Path(output_dir)
            output_dir_obj.mkdir(parents=True, exist_ok=True)
            output_file = output_dir_obj / f"{extract.document_id}.analysis.json"
            output_file.write_text(analysis.model_dump_json(indent=2))

            if verbose:
                logger.info(
                    f"Gap analysis complete: hallucination_score={analysis.hallucination_score:.2f}"
                )
            click.echo(f"✓ Analysis saved: {output_file}")
            click.echo(f"  Hallucination score: {analysis.hallucination_score:.2f}")
            click.echo(f"  Gaps found: {total_gaps}")

        _emit_telemetry(
            command="grounder",
            duration_seconds=duration,
            document_id=extract.document_id,
            extra={
                "GAPS    ": total_gaps,
                "HALLUCIN": f"{analysis.hallucination_score:.2f}",
            },
        )

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        sys.exit(1)
    except TimeoutError as e:
        logger.error(f"Gap analysis timeout: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Gap analysis failed: {e}")
        sys.exit(1)



def run_markdown(extract_json: str, analysis_json: str, output_dir: str, sync: bool = False, verbose: bool = False) -> None:
    """Core markdown generation logic supporting piping and sync."""
    try:
        # Load models
        if extract_json == "-" and (not analysis_json or analysis_json == "-"):
            raw_data = sys.stdin.read().strip()
            if "[ENGRAM_SEAL]::" in raw_data:
                raw_data = raw_data.split("[ENGRAM_SEAL]::")[1]
            combined_data = json.loads(raw_data)

            if "extract" in combined_data:
                extract = DocumentExtract(**combined_data["extract"])
                analysis = GapAnalysis(**combined_data["analysis"])
            else:
                analysis = GapAnalysis(**combined_data)
                extract = DocumentExtract(
                    document_id=analysis.document_id,
                    original_filename=f"{analysis.document_id}.txt",
                    document_type="text",
                    extracted_text="",
                    confidence=1.0,
                    extraction_tool="mock"
                )
        else:
            extract_json_obj = Path(extract_json)
            analysis_json_obj = Path(analysis_json)

            extract_data = json.loads(extract_json_obj.read_text())
            extract = DocumentExtract(**extract_data)

            analysis_data = json.loads(analysis_json_obj.read_text())
            analysis = GapAnalysis(**analysis_data)

        # Perform security check
        if verbose:
            logger.info(f"Running security check for {extract.document_id}")
        security_checker = SecurityChecker()
        security_dict = security_checker.check_security(extract)

        # Build markdown
        if verbose:
            logger.info(f"Building markdown for {extract.document_id}")
        builder = MarkdownBuilder()
        rich_markdown = builder.build_markdown(extract, analysis, security_dict)

        is_stdout_pipe = output_dir == "-" or (not sys.stdout.isatty() and "PYTEST_CURRENT_TEST" not in os.environ)
        if is_stdout_pipe:
            click.echo(rich_markdown.content)
        else:
            output_dir_obj = Path(output_dir)
            output_dir_obj.mkdir(parents=True, exist_ok=True)

            # Write markdown file
            md_file = output_dir_obj / f"{extract.document_id}.md"
            md_file.write_text(rich_markdown.content)

            # Save RichMarkdown model to JSON
            markdown_json_file = output_dir_obj / f"{extract.document_id}.markdown.json"
            markdown_json_file.write_text(rich_markdown.model_dump_json(indent=2))

            if verbose:
                logger.info(f"Markdown generation complete: {md_file}")
            click.echo(f"✓ Markdown saved: {md_file}")
            click.echo(f"✓ Markdown metadata: {markdown_json_file}")

            if sync:
                if verbose:
                    logger.info(f"Triggering sync for {md_file}")
                sync_file(md_file)
                click.echo(f"✓ Sync complete for {md_file}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Markdown generation failed: {e}")
        sys.exit(1)


@click.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    type=click.Path(),
    default="artifacts",
    help="Output directory for extracted JSON (default: artifacts/)",
)
@click.option(
    "--extract-rules",
    is_flag=True,
    help="Extract structured requirements/rules to requirements/ directory.",
)
def extract(file_path: str, output_dir: str, extract_rules: bool) -> None:
    """Extract text and structure from a document.

    FILE_PATH: Path to document to extract (PDF, DOCX, XLSX, PPTX, image, audio, text)
    """
    run_extract(file_path, output_dir, verbose=True, extract_rules_flag=extract_rules)


@click.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Routes execution telemetry to stderr.")
@click.option(
    "--output-dir",
    type=click.Path(),
    default="artifacts",
    help="Output directory for extracted JSON",
)
@click.option(
    "--extract-rules",
    is_flag=True,
    help="Extract structured requirements/rules to requirements/ directory.",
)
def assimilate(file_path: str, verbose: bool, output_dir: str, extract_rules: bool) -> None:
    """Purifies unstructured local files into schema-validated JSON."""
    run_extract(file_path, output_dir, verbose=verbose, extract_rules_flag=extract_rules)


@click.command()
@click.argument("extract_json", type=click.Path())
@click.argument("codex_file", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    type=click.Path(),
    default="artifacts",
    help="Output directory for analysis JSON (default: artifacts/)",
)
def grounder(extract_json: str, codex_file: str, output_dir: str) -> None:
    """Perform 4-dimensional gap analysis on extracted document.

    EXTRACT_JSON: Path to .extract.json file from extract command
    CODEX_FILE: Path to domain codex rules text file
    """
    run_grounder(extract_json, codex_file, output_dir, verbose=True)


@click.command()
@click.argument("json_input", type=str, default="-")
@click.argument("codex_path", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="artifacts",
    help="Output directory for analysis JSON",
)
def cogitate(json_input: str, codex_path: str, verbose: bool, output_dir: str) -> None:
    """Executes multi-dimensional compliance matrix alignment checks."""
    run_grounder(json_input, codex_path, output_dir, verbose=verbose)


@click.command()
@click.argument("extract_json", type=click.Path())
@click.argument("analysis_json", type=click.Path())
@click.option(
    "--output-dir",
    type=click.Path(),
    default="artifacts",
    help="Output directory for markdown output (default: artifacts/)",
)
def markdown(extract_json: str, analysis_json: str, output_dir: str) -> None:
    """Generate RAG-optimized markdown from extracted data and gap analysis.

    EXTRACT_JSON: Path to .extract.json file
    ANALYSIS_JSON: Path to .analysis.json file
    """
    run_markdown(extract_json, analysis_json, output_dir, sync=False, verbose=True)


@click.command()
@click.argument("json_analysis", type=str, default="-")
@click.option("--sync", is_flag=True, help="Triggers egress target workspace file sync.")
@click.option(
    "--output-dir",
    type=click.Path(),
    default="artifacts",
    help="Output directory for markdown output",
)
def inscribe(json_analysis: str, sync: bool, output_dir: str) -> None:
    """Formats final document files, creates security signatures, and writes to disk."""
    run_markdown(json_analysis, "", output_dir, sync=sync, verbose=True)


@click.command()
@click.option(
    "--spool-dir",
    type=click.Path(),
    default="./workspace/spool/",
    help="Spool directory path (default: ./workspace/spool/)",
)
@click.option(
    "--poll-interval",
    type=int,
    default=5,
    help="Seconds between polls (default: 5)",
)
def spool_watch(spool_dir: str, poll_interval: int) -> None:
    """Monitor spool directory for new files and enqueue them for processing.

    Continuously monitors spool_dir/staging/ for new files and adds them to the
    job queue via SpoolCoordinator.
    """
    spool_dir_obj = Path(spool_dir)
    staging_dir = spool_dir_obj / "staging"
    db_path = spool_dir_obj / "jobs.sqlite"

    if not db_path.exists():
        logger.error(f"Spool database not found: {db_path}")
        sys.exit(1)

    try:
        coordinator = SpoolCoordinator(db_path)
        logger.info(f"Started spool watcher: monitoring {staging_dir}")
        logger.info(f"Poll interval: {poll_interval}s")

        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGTERM, _handle_sigterm)
        signal.signal(signal.SIGINT, _handle_sigterm)

        processed_files = set()

        while not _shutdown_event.is_set():
            try:
                # List files in staging directory
                if not staging_dir.exists():
                    staging_dir.mkdir(parents=True, exist_ok=True)

                files = list(staging_dir.glob("*"))
                for file_path in files:
                    if file_path.is_file() and file_path.name not in processed_files:
                        try:
                            # Enqueue file
                            job_id = coordinator.enqueue_file(file_path, pipeline="extraction")
                            logger.info(f"Enqueued: {file_path.name} as job {job_id}")
                            click.echo(f"✓ Enqueued: {file_path.name} → {job_id}")

                            # Track processed file
                            processed_files.add(file_path.name)

                            # Optionally move to done directory (or delete after enqueuing)
                            # For now, leave in staging to avoid duplicates via processed_files tracking
                        except Exception as e:
                            logger.error(f"Failed to enqueue {file_path.name}: {e}")

                # Sleep before next poll
                time.sleep(poll_interval)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Spool watch error: {e}")
                time.sleep(poll_interval)

        logger.info("Spool watcher stopped")

    except Exception as e:
        logger.error(f"Spool watcher failed: {e}")
        sys.exit(1)


@click.command()
@click.option(
    "--spool-dir",
    type=click.Path(),
    default="./workspace/spool/",
    help="Spool directory path (default: ./workspace/spool/)",
)
@click.option(
    "--pipeline",
    type=str,
    default="extraction",
    help="Pipeline name (default: extraction)",
)
@click.option(
    "--worker-id",
    type=str,
    default=None,
    help="Unique worker identifier (default: auto-generate)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="artifacts",
    help="Output directory for extracted JSON (default: artifacts/)",
)
@click.option(
    "--max-tasks",
    type=int,
    default=0,
    help="Max tasks to process (0 = infinite, default)",
)
def worker(spool_dir: str, pipeline: str, worker_id: Optional[str], output_dir: str, max_tasks: int) -> None:
    """Process extraction tasks from spool queue.

    Claims pending tasks from the job queue, extracts documents, and marks
    tasks as complete or failed. Includes automatic lease expiration recovery.
    """
    # Generate worker_id if not provided
    if not worker_id:
        hostname = socket.gethostname()
        worker_id = f"worker-{hostname}-{os.getpid()}"

    db_path = Path(spool_dir) / "jobs.sqlite"
    if not db_path.exists():
        logger.error(f"Spool database not found: {db_path}")
        sys.exit(1)

    try:
        coordinator = SpoolCoordinator(db_path)
        extractor = Extractor()

        logger.info(f"Worker started: id={worker_id}, pipeline={pipeline}, max_tasks={max_tasks}")
        click.echo(f"✓ Worker {worker_id} started (pipeline: {pipeline})")

        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGTERM, _handle_sigterm)
        signal.signal(signal.SIGINT, _handle_sigterm)

        tasks_completed = 0
        last_lease_reclaim = time.time()
        lease_reclaim_failures = 0

        while not _shutdown_event.is_set():
            # Periodically reclaim expired leases (every 30 seconds)
            current_time = time.time()
            if current_time - last_lease_reclaim > 30:
                try:
                    reclaimed = coordinator.reclaim_expired_leases(pipeline)
                    lease_reclaim_failures = 0
                    if reclaimed > 0:
                        logger.info(f"Reclaimed {reclaimed} expired leases")
                    last_lease_reclaim = current_time
                except Exception as e:
                    lease_reclaim_failures += 1
                    logger.warning(f"Lease reclamation failed ({lease_reclaim_failures}x): {e}")
                    if lease_reclaim_failures >= 5:
                        logger.error("Too many lease reclamation failures; exiting")
                        sys.exit(1)

            # Try to claim a task
            try:
                task = coordinator.claim_task(pipeline=pipeline, worker_id=worker_id)

                if not task:
                    # No pending tasks, sleep and retry
                    time.sleep(5)
                    continue

                job_id = task["job_id"]
                payload = json.loads(task["payload"]) if isinstance(task["payload"], str) else task["payload"]
                file_path = payload.get("file_path")

                if not file_path:
                    coordinator.fail_task(job_id, "Task payload missing 'file_path' key")
                    logger.error(f"Task {job_id} has invalid payload: missing file_path")
                    continue

                logger.info(f"Claimed task {job_id}: {file_path}")

                try:
                    # Extract file
                    start_time = time.time()
                    extract = extractor.extract(Path(file_path))
                    elapsed = time.time() - start_time

                    # Save extract
                    artifacts_dir = Path(output_dir)
                    artifacts_dir.mkdir(parents=True, exist_ok=True)
                    extract_file = artifacts_dir / f"{extract.document_id}.extract.json"
                    extract_file.write_text(extract.model_dump_json(indent=2))

                    # Mark complete
                    coordinator.complete_task(job_id)
                    tasks_completed += 1

                    logger.info(f"Completed task {job_id} in {elapsed:.2f}s → {extract.document_id}")
                    click.echo(f"✓ Task {job_id[:8]} done: {extract.original_filename} ({elapsed:.2f}s)")

                    _emit_telemetry(
                        command="worker",
                        duration_seconds=elapsed,
                        document_id=extract.document_id,
                        extra={
                            "JOB-ID  ": job_id[:8],
                            "FILE    ": extract.original_filename,
                            "PIPELINE": pipeline,
                        },
                    )

                    # Check if we've reached max_tasks
                    if max_tasks > 0 and tasks_completed >= max_tasks:
                        logger.info(f"Reached max_tasks limit ({max_tasks}), exiting")
                        break

                except FileNotFoundError as e:
                    coordinator.fail_task(job_id, f"File not found: {e}")
                    logger.error(f"Task {job_id} failed: file not found: {e}")
                    click.echo(f"✗ Task {job_id[:8]} failed: file not found")

                except Exception as e:
                    coordinator.fail_task(job_id, f"Extraction error: {e}")
                    logger.error(f"Task {job_id} failed: {e}")
                    click.echo(f"✗ Task {job_id[:8]} failed: {type(e).__name__}")

            except Exception as e:
                logger.error(f"Worker error during task claim: {e}")
                time.sleep(5)

        logger.info(f"Worker stopped. Tasks completed: {tasks_completed}")
        click.echo(f"Worker stopped. Completed {tasks_completed} tasks.")

    except Exception as e:
        logger.error(f"Worker initialization failed: {e}")
        sys.exit(1)


@click.command()
@click.option(
    "--pipeline",
    required=True,
    type=click.Choice(["watch", "run", "autopilot"]),
    help="Defines the operational layout mode for the background process loops.",
)
@click.option(
    "--spool-dir",
    type=click.Path(),
    default="./workspace/spool/",
    help="Spool directory path (default: ./workspace/spool/)",
)
@click.option(
    "--poll-interval",
    type=int,
    default=5,
    help="Seconds between polls (default: 5)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="artifacts",
    help="Output directory for extracted JSON",
)
def awaken(pipeline: str, spool_dir: str, poll_interval: int, output_dir: str) -> None:
    """Activates background daemon processing layers to automate your scoping tasks."""
    click.echo(f"[*] RITE AWAKEN: Launching background loop in '{pipeline}' mode.", err=True)

    spool_dir_obj = Path(spool_dir)
    staging_dir = spool_dir_obj / "staging"
    db_path = spool_dir_obj / "jobs.sqlite"

    db_path.parent.mkdir(parents=True, exist_ok=True)
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS active_jobs (
            job_id TEXT PRIMARY KEY NOT NULL, pipeline_mode TEXT NOT NULL,
            payload_data TEXT NOT NULL, job_status TEXT NOT NULL DEFAULT 'QUEUED',
            lease_holder TEXT, lease_expires_at REAL, created_at REAL, updated_at REAL
        )
    """)
    conn.close()

    def run_watcher():
        try:
            coordinator = SpoolCoordinator(db_path)
            processed_files = set()
            while not _shutdown_event.is_set():
                if not staging_dir.exists():
                    staging_dir.mkdir(parents=True, exist_ok=True)
                files = list(staging_dir.glob("*"))
                for file_path in files:
                    if file_path.is_file() and file_path.name not in processed_files:
                        try:
                            job_id = coordinator.enqueue_file(file_path, pipeline="extraction")
                            click.echo(f"✓ Enqueued: {file_path.name} → {job_id}", err=True)
                            processed_files.add(file_path.name)
                        except Exception as e:
                            logger.error(f"Failed to enqueue {file_path.name}: {e}")
                time.sleep(poll_interval)
        except Exception as e:
            logger.error(f"Watcher thread failed: {e}")

    def run_worker():
        try:
            coordinator = SpoolCoordinator(db_path)
            extractor = Extractor()
            worker_id = f"worker-{socket.gethostname()}-{os.getpid()}"
            last_lease_reclaim = time.time()

            while not _shutdown_event.is_set():
                current_time = time.time()
                if current_time - last_lease_reclaim > 30:
                    try:
                        coordinator.reclaim_expired_leases("extraction")
                        last_lease_reclaim = current_time
                    except Exception as e:
                        logger.warning(f"Lease reclamation failed: {e}")

                try:
                    task = coordinator.claim_task(pipeline="extraction", worker_id=worker_id)
                    if not task:
                        time.sleep(poll_interval)
                        continue

                    job_id = task["job_id"]
                    payload = json.loads(task["payload"]) if isinstance(task["payload"], str) else task["payload"]
                    file_path = payload.get("file_path")
                    if not file_path:
                        coordinator.fail_task(job_id, "Missing file_path")
                        continue

                    start_time = time.time()
                    extract_res = extractor.extract(Path(file_path))
                    elapsed = time.time() - start_time

                    # Save extract
                    artifacts_dir = Path(output_dir)
                    artifacts_dir.mkdir(parents=True, exist_ok=True)
                    extract_file = artifacts_dir / f"{extract_res.document_id}.extract.json"
                    extract_file.write_text(extract_res.model_dump_json(indent=2))

                    coordinator.complete_task(job_id)
                    click.echo(f"✓ Completed task {job_id[:8]} in {elapsed:.2f}s", err=True)
                except Exception as e:
                    if 'job_id' in locals():
                        coordinator.fail_task(job_id, str(e))
                    logger.error(f"Worker task error: {e}")
                    time.sleep(poll_interval)
        except Exception as e:
            logger.error(f"Worker thread failed: {e}")

    # Register signals
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    if pipeline == "watch":
        run_watcher()
    elif pipeline == "run":
        run_worker()
    elif pipeline == "autopilot":
        watcher_t = threading.Thread(target=run_watcher, daemon=True)
        worker_t = threading.Thread(target=run_worker, daemon=True)
        watcher_t.start()
        worker_t.start()
        try:
            while not _shutdown_event.is_set():
                time.sleep(1.0)
        except KeyboardInterrupt:
            _shutdown_event.set()


@click.command()
@click.option("--query", required=True, help="Natural language research question")
@click.option(
    "--domain",
    default="general",
    help="Domain tag for output metadata (default: general)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="artifacts",
    help="Output directory for research JSON and markdown (default: artifacts/)",
)
@click.option(
    "--max",
    "use_max",
    is_flag=True,
    help="Use deep-research-max model (slower, more thorough)",
)
def research(query: str, domain: str, output_dir: str, use_max: bool) -> None:
    """Submit a deep research query to Gemini and save the report.

    Requires GEMINI_API_KEY environment variable.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        click.echo("Error: GEMINI_API_KEY environment variable is not set", err=True)
        raise SystemExit(1)

    from servo_skull.deep_research import DeepResearchClient
    from pathlib import Path

    client = DeepResearchClient(api_key=api_key, use_max=use_max)
    click.echo(f"✙ Deep Research starting (model: {client.model})")
    click.echo(f"  Query: {query[:100]}...")

    _start = time.time()
    report = client.research(query=query, domain=domain)
    duration = time.time() - _start

    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    slug = "".join(c if c.isalnum() else "-" for c in query[:40]).strip("-").lower()
    stem = f"dr-{slug}"

    json_file = output_dir_obj / f"{stem}.research.json"
    md_file = output_dir_obj / f"{stem}.research.md"

    json_file.write_text(report.model_dump_json(indent=2))
    md_file.write_text(
        f"---\nquery: \"{report.query}\"\nmodel: {report.model}\n"
        f"domain: {report.domain}\nduration_s: {report.duration_seconds}\n"
        f"sources_count: {len(report.sources)}\n---\n\n"
        + report.report_text
    )

    click.echo(f"✓ Report saved: {md_file}")
    click.echo(f"✓ JSON saved:   {json_file}")
    _emit_telemetry(
        command="research",
        duration_seconds=duration,
        extra={"DOMAIN  ": domain, "SOURCES ": len(report.sources)},
    )

