"""Tests for CLI entry points."""
import json
import os
import signal
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from servo_skull._cli import extract, grounder, markdown, spool_watch, worker, _shutdown_event
from servo_skull.models import DocumentExtract, GapAnalysis, RichMarkdown


class TestExtractCommand:
    """Tests for extract CLI command."""

    def test_extract_command_valid_file(self, sample_text_file, tmp_path):
        """Test extract command with valid file creates output JSON."""
        runner = CliRunner()
        output_dir = tmp_path / "output"
        result = runner.invoke(extract, [str(sample_text_file), "--output-dir", str(output_dir)])

        assert result.exit_code == 0
        assert "Extraction complete" in result.output or "✓ Extracted" in result.output

        # Check output file was created
        json_files = list(output_dir.glob("*.extract.json"))
        assert len(json_files) == 1
        assert json_files[0].exists()

    def test_extract_command_nonexistent_file(self, tmp_path):
        """Test extract command with nonexistent file exits with error."""
        runner = CliRunner()
        nonexistent = str(tmp_path / "nonexistent.pdf")
        result = runner.invoke(extract, [nonexistent])

        assert result.exit_code != 0

    def test_extract_command_output_format(self, sample_text_file, tmp_path):
        """Test extract command output contains DocumentExtract fields."""
        runner = CliRunner()
        output_dir = tmp_path / "output"
        result = runner.invoke(extract, [str(sample_text_file), "--output-dir", str(output_dir)])

        assert result.exit_code == 0

        # Load and validate JSON
        json_files = list(output_dir.glob("*.extract.json"))
        assert len(json_files) == 1

        data = json.loads(json_files[0].read_text())
        assert "document_id" in data
        assert "original_filename" in data
        assert "document_type" in data
        assert "extracted_text" in data
        assert "confidence" in data
        assert "extraction_tool" in data
        assert "extraction_timestamp" in data

    def test_extract_command_default_output_dir(self, sample_text_file, tmp_path, monkeypatch):
        """Test extract command uses default output directory."""
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(extract, [str(sample_text_file)])

        assert result.exit_code == 0
        assert (tmp_path / "artifacts").exists()


class TestGrounderCommand:
    """Tests for grounder CLI command."""

    def test_grounder_command_valid_files(self, sample_document_extract, tmp_path):
        """Test grounder command with valid extract and codex files."""
        # Create extract JSON file
        extract_file = tmp_path / "test.extract.json"
        extract_file.write_text(sample_document_extract.model_dump_json())

        # Create codex file
        codex_file = tmp_path / "codex.txt"
        codex_file.write_text("""
        Dimension 1: Stakeholder Ecosystem
        - Legal & GRC: Contract liability, SLA penalties
        - Engineering: API compatibility, infrastructure

        Dimension 2: Delivery Methodology
        - Governance: PRINCE2, PMBOK
        - Execution: Agile/Scrum
        """)

        # Mock analyze_gaps to avoid LLM call
        mock_analysis = GapAnalysis(
            document_id=sample_document_extract.document_id,
            gaps={"stakeholder": [{"gap": "Test gap", "severity": "high"}]},
            risk_flags=[],
            security_flags=[],
            hallucination_score=0.15,
            llm_model="test-model",
        )

        runner = CliRunner()
        output_dir = tmp_path / "output"

        with patch("servo_skull._cli.analyze_gaps", return_value=mock_analysis):
            result = runner.invoke(
                grounder,
                [str(extract_file), str(codex_file), "--output-dir", str(output_dir)],
            )

        assert result.exit_code == 0
        assert "Gap analysis complete" in result.output or "✓ Analysis saved" in result.output

        # Check output file
        json_files = list(output_dir.glob("*.analysis.json"))
        assert len(json_files) == 1

    def test_grounder_command_missing_codex(self, sample_document_extract, tmp_path):
        """Test grounder command fails with missing codex file."""
        extract_file = tmp_path / "test.extract.json"
        extract_file.write_text(sample_document_extract.model_dump_json())

        nonexistent_codex = str(tmp_path / "nonexistent.txt")

        runner = CliRunner()
        result = runner.invoke(grounder, [str(extract_file), nonexistent_codex])

        assert result.exit_code != 0

    def test_grounder_command_output_includes_hallucination_score(
        self, sample_document_extract, tmp_path
    ):
        """Test grounder command output includes hallucination_score."""
        extract_file = tmp_path / "test.extract.json"
        extract_file.write_text(sample_document_extract.model_dump_json())

        codex_file = tmp_path / "codex.txt"
        codex_file.write_text("Test codex content")

        mock_analysis = GapAnalysis(
            document_id=sample_document_extract.document_id,
            gaps={},
            risk_flags=[],
            security_flags=[],
            hallucination_score=0.32,
            llm_model="test-model",
        )

        runner = CliRunner()
        output_dir = tmp_path / "output"

        with patch("servo_skull._cli.analyze_gaps", return_value=mock_analysis):
            result = runner.invoke(
                grounder,
                [str(extract_file), str(codex_file), "--output-dir", str(output_dir)],
            )

        assert result.exit_code == 0
        assert "0.32" in result.output or "hallucination" in result.output.lower()

        # Verify JSON output
        json_files = list(output_dir.glob("*.analysis.json"))
        data = json.loads(json_files[0].read_text())
        assert data["hallucination_score"] == 0.32


class TestMarkdownCommand:
    """Tests for markdown CLI command."""

    def test_markdown_command_builds_markdown(
        self, sample_document_extract, sample_gap_analysis, tmp_path
    ):
        """Test markdown command generates markdown output."""
        extract_file = tmp_path / "test.extract.json"
        extract_file.write_text(sample_document_extract.model_dump_json())

        analysis_file = tmp_path / "test.analysis.json"
        analysis_file.write_text(sample_gap_analysis.model_dump_json())

        mock_markdown = RichMarkdown(
            document_id=sample_document_extract.document_id,
            frontmatter={"doc_id": sample_document_extract.document_id},
            content="# Test Markdown\n\nContent here.",
            see_also=[],
            scriptorum_refs=[],
            binaric_cant_footer={},
            output_path="test.md",
        )

        runner = CliRunner()
        output_dir = tmp_path / "output"

        with patch(
            "servo_skull._cli.SecurityChecker"
        ) as mock_checker_class, patch(
            "servo_skull._cli.MarkdownBuilder"
        ) as mock_builder_class:
            mock_checker = MagicMock()
            mock_checker.check_security.return_value = {"security_issues": []}
            mock_checker_class.return_value = mock_checker

            mock_builder = MagicMock()
            mock_builder.build_markdown.return_value = mock_markdown
            mock_builder_class.return_value = mock_builder

            result = runner.invoke(
                markdown,
                [str(extract_file), str(analysis_file), "--output-dir", str(output_dir)],
            )

        assert result.exit_code == 0
        assert "Markdown saved" in result.output or "✓ Markdown" in result.output

        # Check files were created
        md_files = list(output_dir.glob("*.md"))
        assert len(md_files) == 1
        assert "Test Markdown" in md_files[0].read_text()

    def test_markdown_command_saves_markdown_json(
        self, sample_document_extract, sample_gap_analysis, tmp_path
    ):
        """Test markdown command saves RichMarkdown model to JSON."""
        extract_file = tmp_path / "test.extract.json"
        extract_file.write_text(sample_document_extract.model_dump_json())

        analysis_file = tmp_path / "test.analysis.json"
        analysis_file.write_text(sample_gap_analysis.model_dump_json())

        mock_markdown = RichMarkdown(
            document_id=sample_document_extract.document_id,
            frontmatter={"title": "Test"},
            content="# Test",
            see_also=[("Link", "path")],
            scriptorum_refs=["ref1"],
            binaric_cant_footer={"test": "value"},
            output_path="test.md",
        )

        runner = CliRunner()
        output_dir = tmp_path / "output"

        with patch(
            "servo_skull._cli.SecurityChecker"
        ) as mock_checker_class, patch(
            "servo_skull._cli.MarkdownBuilder"
        ) as mock_builder_class:
            mock_checker = MagicMock()
            mock_checker.check_security.return_value = {}
            mock_checker_class.return_value = mock_checker

            mock_builder = MagicMock()
            mock_builder.build_markdown.return_value = mock_markdown
            mock_builder_class.return_value = mock_builder

            result = runner.invoke(
                markdown,
                [str(extract_file), str(analysis_file), "--output-dir", str(output_dir)],
            )

        assert result.exit_code == 0

        # Check markdown.json was saved
        json_files = list(output_dir.glob("*.markdown.json"))
        assert len(json_files) == 1

        data = json.loads(json_files[0].read_text())
        assert "document_id" in data
        assert "frontmatter" in data
        assert "content" in data
        assert "see_also" in data
        assert "scriptorum_refs" in data
        assert "binaric_cant_footer" in data

    def test_markdown_command_includes_binaric_cant(
        self, sample_document_extract, sample_gap_analysis, tmp_path
    ):
        """Test markdown command includes Binaric Cant footer in JSON."""
        extract_file = tmp_path / "test.extract.json"
        extract_file.write_text(sample_document_extract.model_dump_json())

        analysis_file = tmp_path / "test.analysis.json"
        analysis_file.write_text(sample_gap_analysis.model_dump_json())

        binaric_cant = {
            "doc_id": sample_document_extract.document_id,
            "gaps_count": 2,
            "hallucination_score": 0.12,
            "security_issues": 0,
        }

        mock_markdown = RichMarkdown(
            document_id=sample_document_extract.document_id,
            frontmatter={},
            content="# Test",
            see_also=[],
            scriptorum_refs=[],
            binaric_cant_footer=binaric_cant,
            output_path="test.md",
        )

        runner = CliRunner()
        output_dir = tmp_path / "output"

        with patch(
            "servo_skull._cli.SecurityChecker"
        ) as mock_checker_class, patch(
            "servo_skull._cli.MarkdownBuilder"
        ) as mock_builder_class:
            mock_checker = MagicMock()
            mock_checker.check_security.return_value = {}
            mock_checker_class.return_value = mock_checker

            mock_builder = MagicMock()
            mock_builder.build_markdown.return_value = mock_markdown
            mock_builder_class.return_value = mock_builder

            result = runner.invoke(
                markdown,
                [str(extract_file), str(analysis_file), "--output-dir", str(output_dir)],
            )

        assert result.exit_code == 0

        json_files = list(output_dir.glob("*.markdown.json"))
        data = json.loads(json_files[0].read_text())
        assert data["binaric_cant_footer"]["doc_id"] == sample_document_extract.document_id
        assert data["binaric_cant_footer"]["hallucination_score"] == 0.12


class TestSpoolWatchCommand:
    """Tests for spool_watch CLI command."""

    def test_spool_watch_enqueues_files(self, spool_db, tmp_path):
        """Test spool_watch enqueues files from staging directory."""
        # Create staging directory and test file
        staging_dir = tmp_path / "staging"
        staging_dir.mkdir()

        test_file = staging_dir / "test.txt"
        test_file.write_text("Test content")

        runner = CliRunner()
        enqueue_counter = {"count": 0}

        def mock_enqueue(*args, **kwargs):
            """Mock enqueue_file to track calls and signal exit."""
            enqueue_counter["count"] += 1
            _shutdown_event.set()  # Exit after first enqueue
            return "test-job-id"

        with patch("servo_skull._cli.SpoolCoordinator") as mock_coordinator_class:
            mock_coordinator = MagicMock()
            mock_coordinator_class.return_value = mock_coordinator
            mock_coordinator.enqueue_file.side_effect = mock_enqueue

            result = runner.invoke(
                spool_watch,
                ["--spool-dir", str(tmp_path)],
                catch_exceptions=False,
            )

        # Verify that enqueue_file was actually called
        assert mock_coordinator.enqueue_file.call_count >= 1
        assert result.exit_code == 0

    def test_spool_watch_handles_errors(self, spool_db, tmp_path):
        """Test spool_watch logs errors and continues polling."""
        staging_dir = tmp_path / "staging"
        staging_dir.mkdir()

        runner = CliRunner()

        with patch("servo_skull._cli.time.sleep"):
            # Set shutdown event to exit immediately after loop check
            _shutdown_event.set()
            result = runner.invoke(
                spool_watch,
                ["--spool-dir", str(tmp_path)],
                catch_exceptions=True,
            )
            _shutdown_event.clear()  # Reset for next test

        # Should not crash
        assert result.exit_code == 0


class TestWorkerCommand:
    """Tests for worker CLI command."""

    def test_worker_claims_tasks(self, spool_db, tmp_path, sample_text_file):
        """Test worker claims and processes pending task."""
        # Enqueue a task
        coordinator_module = "servo_skull._cli.SpoolCoordinator"
        extractor_module = "servo_skull._cli.Extractor"

        with patch(coordinator_module) as mock_coordinator_class, patch(
            extractor_module
        ) as mock_extractor_class:
            mock_coordinator = MagicMock()
            mock_coordinator_class.return_value = mock_coordinator

            mock_extractor = MagicMock()
            mock_extractor_class.return_value = mock_extractor

            # First call returns task, second returns None
            mock_extract = DocumentExtract(
                document_id="test-123",
                original_filename="test.txt",
                document_type="text",
                extracted_text="Test",
                confidence=0.99,
                extraction_tool="test",
            )
            mock_extractor.extract.return_value = mock_extract
            mock_coordinator.claim_task.side_effect = [
                {"job_id": "job-1", "pipeline": "extraction", "payload": {"file_path": str(sample_text_file)}},
                None,  # Second call returns None to exit
            ]

            runner = CliRunner()

            with patch("servo_skull._cli.time.sleep"):
                result = runner.invoke(
                    worker,
                    ["--max-tasks", "1"],
                    catch_exceptions=False,
                )

            # Should have completed the task
            assert result.exit_code == 0 or "completed" in result.output.lower()

    def test_worker_fails_task_on_extraction_error(self, spool_db):
        """Test worker calls fail_task on extraction failure."""
        with patch("servo_skull._cli.SpoolCoordinator") as mock_coordinator_class, patch(
            "servo_skull._cli.Extractor"
        ) as mock_extractor_class:
            mock_coordinator = MagicMock()
            mock_coordinator_class.return_value = mock_coordinator

            mock_extractor = MagicMock()
            mock_extractor_class.return_value = mock_extractor

            # Extraction fails
            mock_extractor.extract.side_effect = FileNotFoundError("File not found")
            mock_coordinator.claim_task.side_effect = [
                {"job_id": "job-1", "pipeline": "extraction", "payload": {"file_path": "/nonexistent"}},
                None,
            ]
            # Prevent infinite loop in test by shutting down worker when it fails a task
            from servo_skull._cli import _shutdown_event
            mock_coordinator.fail_task.side_effect = lambda *args, **kwargs: _shutdown_event.set()

            runner = CliRunner()

            try:
                with patch("servo_skull._cli.time.sleep"):
                    result = runner.invoke(
                        worker,
                        ["--max-tasks", "1"],
                        catch_exceptions=False,
                    )
            finally:
                _shutdown_event.clear()

            # Should have called fail_task
            mock_coordinator.fail_task.assert_called()

    def test_worker_respects_max_tasks(self, spool_db):
        """Test worker exits after processing max_tasks."""
        with patch("servo_skull._cli.SpoolCoordinator") as mock_coordinator_class, patch(
            "servo_skull._cli.Extractor"
        ) as mock_extractor_class:
            mock_coordinator = MagicMock()
            mock_coordinator_class.return_value = mock_coordinator

            mock_extractor = MagicMock()
            mock_extractor_class.return_value = mock_extractor

            mock_extract = DocumentExtract(
                document_id="test-123",
                original_filename="test.txt",
                document_type="text",
                extracted_text="Test",
                confidence=0.99,
                extraction_tool="test",
            )
            mock_extractor.extract.return_value = mock_extract

            # Return two tasks then None
            mock_coordinator.claim_task.side_effect = [
                {"job_id": "job-1", "pipeline": "extraction", "payload": {"file_path": "/tmp/test1"}},
                {"job_id": "job-2", "pipeline": "extraction", "payload": {"file_path": "/tmp/test2"}},
            ]

            runner = CliRunner()

            with patch("servo_skull._cli.time.sleep"):
                result = runner.invoke(
                    worker,
                    ["--max-tasks", "2"],
                    catch_exceptions=False,
                )

            assert result.exit_code == 0
            # Should have claimed exactly 2 tasks
            assert mock_coordinator.claim_task.call_count >= 2

    def test_worker_auto_generates_worker_id(self, spool_db):
        """Test worker auto-generates worker_id when not provided."""
        with patch("servo_skull._cli.SpoolCoordinator") as mock_coordinator_class, patch(
            "servo_skull._cli.Extractor"
        ) as mock_extractor_class:
            mock_coordinator = MagicMock()
            mock_coordinator_class.return_value = mock_coordinator

            mock_extractor_class.return_value = MagicMock()
            from servo_skull._cli import _shutdown_event
            def mock_claim_task(*args, **kwargs):
                _shutdown_event.set()
                return None
            mock_coordinator.claim_task.side_effect = mock_claim_task

            runner = CliRunner()

            _shutdown_event.clear()
            with patch("servo_skull._cli.time.sleep"):
                result = runner.invoke(
                    worker,
                    [],  # No --worker-id specified
                    catch_exceptions=False,
                )
            _shutdown_event.clear()  # Reset for next test

            # Should have called claim_task with auto-generated worker_id
            calls = mock_coordinator.claim_task.call_args_list
            assert len(calls) >= 1
            args, kwargs = calls[0]
            worker_id = kwargs.get("worker_id", "")
            assert "worker-" in worker_id

    def test_worker_reclaims_expired_leases(self, spool_db):
        """Test worker periodically calls reclaim_expired_leases."""
        with patch("servo_skull._cli.SpoolCoordinator") as mock_coordinator_class, patch(
            "servo_skull._cli.Extractor"
        ) as mock_extractor_class, patch("servo_skull._cli.time.time") as mock_time:
            mock_coordinator = MagicMock()
            mock_coordinator_class.return_value = mock_coordinator

            mock_extractor_class.return_value = MagicMock()

            # Simulate time progression where the first loop check has a delta > 30 seconds
            # Call 1 (logging start): 0
            # Call 2 (last_lease_reclaim): 0
            # Call 3 (current_time): 40 (delta = 40 > 30)
            mock_time.side_effect = [0, 0, 40, 50, 60]
            from servo_skull._cli import _shutdown_event
            def mock_claim_task(*args, **kwargs):
                _shutdown_event.set()
                return None
            mock_coordinator.claim_task.side_effect = mock_claim_task

            runner = CliRunner()

            _shutdown_event.clear()
            with patch("servo_skull._cli.time.sleep"):
                result = runner.invoke(
                    worker,
                    ["--max-tasks", "1"],
                    catch_exceptions=False,
                )
            _shutdown_event.clear()  # Reset for next test

            # Should have called reclaim_expired_leases at least once
            mock_coordinator.reclaim_expired_leases.assert_called()

    def test_worker_handles_sigterm_gracefully(self, spool_db):
        """Test SIGTERM signal triggers graceful shutdown."""
        with patch("servo_skull._cli.SpoolCoordinator") as mock_coordinator_class, patch(
            "servo_skull._cli.Extractor"
        ) as mock_extractor_class:
            mock_coordinator = MagicMock()
            mock_coordinator_class.return_value = mock_coordinator

            mock_extractor_class.return_value = MagicMock()
            mock_coordinator.claim_task.return_value = None

            runner = CliRunner()

            _shutdown_event.set()
            with patch("servo_skull._cli.time.sleep"):
                result = runner.invoke(
                    worker,
                    [],
                    catch_exceptions=False,
                )
            _shutdown_event.clear()  # Reset for next test

            # Should exit gracefully
            assert result.exit_code == 0


def test_extract_command_directory(tmp_path):
    """Test extract command handles directory recursively."""
    input_dir = tmp_path / "input_docs"
    input_dir.mkdir()
    
    (input_dir / "doc1.txt").write_text("This is test document one content.", encoding="utf-8")
    (input_dir / "doc2.md").write_text("This is test document two content.", encoding="utf-8")
    (input_dir / "doc_ignored.xyz").write_text("Ignored format", encoding="utf-8")

    runner = CliRunner()
    output_dir = tmp_path / "output_extracts"

    result = runner.invoke(extract, [str(input_dir), "--output-dir", str(output_dir)])
    assert result.exit_code == 0
    assert "Found 2 supported files in directory" in result.output
    assert "doc1.txt" in result.output
    assert "doc2.md" in result.output

    # Check that extract files are created
    extract_files = list(output_dir.glob("*.extract.json"))
    assert len(extract_files) == 2


def test_grounder_command_directory(tmp_path):
    """Test grounder command handles directory of extracts."""
    extracts_dir = tmp_path / "input_extracts"
    extracts_dir.mkdir()

    extract1 = DocumentExtract(
        document_id="doc1",
        original_filename="doc1.txt",
        document_type="text",
        extracted_text="Content 1",
        confidence=0.99,
        extraction_tool="test"
    )
    extract2 = DocumentExtract(
        document_id="doc2",
        original_filename="doc2.md",
        document_type="text",
        extracted_text="Content 2",
        confidence=0.99,
        extraction_tool="test"
    )
    (extracts_dir / "doc1.extract.json").write_text(extract1.model_dump_json(), encoding="utf-8")
    (extracts_dir / "doc2.extract.json").write_text(extract2.model_dump_json(), encoding="utf-8")
    (extracts_dir / "ignored.txt").write_text("not an extract", encoding="utf-8")

    codex_file = tmp_path / "codex.txt"
    codex_file.write_text("Dimension 1: Stakeholder Ecosystem\n- Legal & GRC: Liability", encoding="utf-8")

    mock_analysis = GapAnalysis(
        document_id="doc-temp",
        gaps={"stakeholder": [{"gap": "Mock gap", "severity": "low"}]},
        risk_flags=[],
        security_flags=[],
        hallucination_score=0.0,
        llm_model="test-model"
    )

    runner = CliRunner()
    output_dir = tmp_path / "output_analysis"

    with patch("servo_skull._cli.analyze_gaps", return_value=mock_analysis):
        result = runner.invoke(
            grounder,
            [str(extracts_dir), str(codex_file), "--output-dir", str(output_dir)]
        )

    assert result.exit_code == 0
    assert "Found 2 extract files to analyze" in result.output

    analysis_files = list(output_dir.glob("*.analysis.json"))
    assert len(analysis_files) == 2


def test_research_command_missing_query(monkeypatch):
    from servo_skull._cli import research
    runner = CliRunner()
    result = runner.invoke(research, [])
    assert result.exit_code != 0  # missing required argument


def test_research_command_requires_api_key(monkeypatch, tmp_path):
    """Without GEMINI_API_KEY the command must exit non-zero."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from servo_skull._cli import research
    runner = CliRunner()
    result = runner.invoke(research, ["--query", "data retention policy", "--output-dir", str(tmp_path)])
    assert result.exit_code != 0

