"""Comprehensive tests for SpoolCoordinator (persistent job queue)."""
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from servo_skull.spool_coordinator import SpoolCoordinator


class TestEnqueueFile:
    """Tests for SpoolCoordinator.enqueue_file()."""

    def test_enqueue_file_creates_job(self, spool_db, sample_pdf):
        """Calling enqueue_file() returns a valid job_id UUID."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)

        # Verify it's a valid UUID
        try:
            UUID(job_id)
            is_valid_uuid = True
        except ValueError:
            is_valid_uuid = False

        assert is_valid_uuid, f"Job ID {job_id} is not a valid UUID"

    def test_enqueue_file_sets_pending_status(self, spool_db, sample_pdf):
        """Job status is 'pending' immediately after enqueue."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)

        # Query the database
        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            )
            status = cursor.fetchone()[0]

        assert status == "pending"

    def test_enqueue_file_includes_checksum(self, spool_db, sample_pdf):
        """Payload includes checksum field."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)

        # Query the database
        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT payload FROM jobs WHERE id = ?", (job_id,)
            )
            payload_json = cursor.fetchone()[0]

        payload = json.loads(payload_json)
        assert "checksum" in payload
        assert isinstance(payload["checksum"], str)
        assert len(payload["checksum"]) == 64  # SHA256 is 64 hex chars

    def test_enqueue_file_includes_file_path(self, spool_db, sample_pdf):
        """Payload includes file_path field."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT payload FROM jobs WHERE id = ?", (job_id,)
            )
            payload_json = cursor.fetchone()[0]

        payload = json.loads(payload_json)
        assert "file_path" in payload
        assert payload["file_path"] == str(sample_pdf)

    def test_enqueue_file_includes_filename(self, spool_db, sample_pdf):
        """Payload includes original_filename field."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT payload FROM jobs WHERE id = ?", (job_id,)
            )
            payload_json = cursor.fetchone()[0]

        payload = json.loads(payload_json)
        assert "original_filename" in payload
        assert payload["original_filename"] == sample_pdf.name

    def test_enqueue_file_includes_file_size(self, spool_db, sample_pdf):
        """Payload includes file_size field."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT payload FROM jobs WHERE id = ?", (job_id,)
            )
            payload_json = cursor.fetchone()[0]

        payload = json.loads(payload_json)
        assert "file_size" in payload
        assert payload["file_size"] == sample_pdf.stat().st_size

    def test_enqueue_file_nonexistent_raises_valueerror(self, spool_db):
        """ValueError raised if file_path doesn't exist."""
        coordinator = SpoolCoordinator(spool_db)

        with pytest.raises(ValueError, match="File does not exist"):
            coordinator.enqueue_file(Path("/nonexistent/file.pdf"))

    def test_enqueue_file_custom_pipeline(self, spool_db, sample_pdf):
        """Custom pipeline name is stored correctly."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf, pipeline="custom_pipeline")

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT pipeline FROM jobs WHERE id = ?", (job_id,)
            )
            pipeline = cursor.fetchone()[0]

        assert pipeline == "custom_pipeline"

    def test_enqueue_file_multiple_jobs(self, spool_db, tmp_path):
        """Multiple files can be enqueued separately."""
        coordinator = SpoolCoordinator(spool_db)

        # Create two test files
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content 1")
        file2.write_text("content 2")

        job_id_1 = coordinator.enqueue_file(file1)
        job_id_2 = coordinator.enqueue_file(file2)

        assert job_id_1 != job_id_2

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = 'pending'"
            )
            count = cursor.fetchone()[0]

        assert count == 2


class TestMoveToPending:
    """Tests for SpoolCoordinator.move_to_pending()."""

    def test_move_to_pending_updates_status(self, spool_db, sample_pdf):
        """Status changes from 'pending' (was already pending after enqueue)."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)

        result = coordinator.move_to_pending(job_id)
        assert result is True

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            )
            status = cursor.fetchone()[0]

        assert status == "pending"

    def test_move_to_pending_nonexistent_returns_false(self, spool_db):
        """Returns False if job_id not found."""
        coordinator = SpoolCoordinator(spool_db)
        result = coordinator.move_to_pending("nonexistent-job-id")
        assert result is False


class TestClaimTask:
    """Tests for SpoolCoordinator.claim_task()."""

    def test_claim_task_returns_dict(self, spool_db, sample_pdf):
        """claim_task() returns dict with job_id, pipeline, payload."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)

        result = coordinator.claim_task()

        assert isinstance(result, dict)
        assert "job_id" in result
        assert "pipeline" in result
        assert "payload" in result

    def test_claim_task_sets_processing_status(self, spool_db, sample_pdf):
        """Job status becomes 'processing' after claim."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)

        coordinator.claim_task()

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            )
            status = cursor.fetchone()[0]

        assert status == "processing"

    def test_claim_task_sets_worker_id(self, spool_db, sample_pdf):
        """Job worker_id set to provided worker_id."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)

        result = coordinator.claim_task(worker_id="test-worker-123")

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT worker_id FROM jobs WHERE id = ?", (job_id,)
            )
            worker_id = cursor.fetchone()[0]

        assert worker_id == "test-worker-123"

    def test_claim_task_sets_lease_expiry(self, spool_db, sample_pdf):
        """Job leased_until set to NOW + lease_seconds."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)

        before_claim = datetime.now()
        result = coordinator.claim_task(lease_seconds=300)
        after_claim = datetime.now()

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT leased_until FROM jobs WHERE id = ?", (job_id,)
            )
            leased_until_str = cursor.fetchone()[0]

        leased_until = datetime.fromisoformat(leased_until_str)

        # Lease should be approximately NOW + 300 seconds
        # Allow 5 second tolerance for execution time
        expected_min = before_claim + timedelta(seconds=295)
        expected_max = after_claim + timedelta(seconds=305)

        assert expected_min <= leased_until <= expected_max

    def test_claim_task_no_pending_returns_none(self, spool_db):
        """Returns None if no pending jobs."""
        coordinator = SpoolCoordinator(spool_db)
        result = coordinator.claim_task()
        assert result is None

    def test_claim_task_claims_oldest_first(self, spool_db, tmp_path):
        """Claims oldest created_at job first (FIFO)."""
        coordinator = SpoolCoordinator(spool_db)

        # Create two jobs with slight delay
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content 1")
        file2.write_text("content 2")

        job_id_1 = coordinator.enqueue_file(file1)

        # Small delay to ensure created_at differs
        import time
        time.sleep(0.1)

        job_id_2 = coordinator.enqueue_file(file2)

        # Claim should get the first job
        result = coordinator.claim_task()
        assert result["job_id"] == job_id_1

    def test_claim_task_returns_payload(self, spool_db, sample_pdf):
        """Returned dict includes complete payload with checksums."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)

        result = coordinator.claim_task()

        payload = result["payload"]
        assert "file_path" in payload
        assert "original_filename" in payload
        assert "file_size" in payload
        assert "checksum" in payload

    def test_claim_task_custom_pipeline(self, spool_db, sample_pdf):
        """Only claims from specified pipeline."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf, pipeline="custom")

        # Try to claim from wrong pipeline
        result = coordinator.claim_task(pipeline="extraction")
        assert result is None

        # Claim from correct pipeline
        result = coordinator.claim_task(pipeline="custom")
        assert result is not None
        assert result["job_id"] == job_id


class TestCompleteTask:
    """Tests for SpoolCoordinator.complete_task()."""

    def test_complete_task_updates_status(self, spool_db, sample_pdf):
        """Status changes from 'processing' → 'done'."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)
        coordinator.claim_task()

        result = coordinator.complete_task(job_id)
        assert result is True

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            )
            status = cursor.fetchone()[0]

        assert status == "done"

    def test_complete_task_clears_worker_id(self, spool_db, sample_pdf):
        """worker_id cleared after completion."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)
        coordinator.claim_task(worker_id="test-worker")

        coordinator.complete_task(job_id)

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT worker_id FROM jobs WHERE id = ?", (job_id,)
            )
            worker_id = cursor.fetchone()[0]

        assert worker_id is None

    def test_complete_task_clears_lease(self, spool_db, sample_pdf):
        """leased_until cleared after completion."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)
        coordinator.claim_task()

        coordinator.complete_task(job_id)

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT leased_until FROM jobs WHERE id = ?", (job_id,)
            )
            leased_until = cursor.fetchone()[0]

        assert leased_until is None

    def test_complete_task_nonexistent_returns_false(self, spool_db):
        """Returns False if job_id not found."""
        coordinator = SpoolCoordinator(spool_db)
        result = coordinator.complete_task("nonexistent-job-id")
        assert result is False


class TestFailTask:
    """Tests for SpoolCoordinator.fail_task()."""

    def test_fail_task_increments_attempts(self, spool_db, sample_pdf):
        """attempts counter incremented."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)
        coordinator.claim_task()

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT attempts FROM jobs WHERE id = ?", (job_id,)
            )
            attempts_before = cursor.fetchone()[0]

        coordinator.fail_task(job_id, "Test error")

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT attempts FROM jobs WHERE id = ?", (job_id,)
            )
            attempts_after = cursor.fetchone()[0]

        assert attempts_after == attempts_before + 1

    def test_fail_task_retries_if_attempts_lte_3(self, spool_db, sample_pdf):
        """Status → 'pending' if attempts <= 3."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)
        coordinator.claim_task()

        # First failure (attempts = 1)
        coordinator.fail_task(job_id, "Error 1")

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            )
            status = cursor.fetchone()[0]

        assert status == "pending"

    def test_fail_task_fails_if_attempts_gt_3(self, spool_db, sample_pdf):
        """Status → 'failed' and error set if attempts > 3."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)

        # Manually set attempts to 3, then claim and fail
        with sqlite3.connect(spool_db) as conn:
            conn.execute(
                "UPDATE jobs SET attempts = 3 WHERE id = ?", (job_id,)
            )
            conn.commit()

        coordinator.claim_task()
        coordinator.fail_task(job_id, "Final error")

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT status, attempts FROM jobs WHERE id = ?", (job_id,)
            )
            status, attempts = cursor.fetchone()

        assert status == "failed"
        assert attempts == 4

    def test_fail_task_includes_error_message(self, spool_db, sample_pdf):
        """error field populated with message."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)

        # Set attempts to 3 to trigger failure
        with sqlite3.connect(spool_db) as conn:
            conn.execute(
                "UPDATE jobs SET attempts = 3 WHERE id = ?", (job_id,)
            )
            conn.commit()

        coordinator.claim_task()
        error_msg = "Extraction timeout after 60 seconds"
        coordinator.fail_task(job_id, error_msg)

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT error FROM jobs WHERE id = ?", (job_id,)
            )
            stored_error = cursor.fetchone()[0]

        assert stored_error == error_msg

    def test_fail_task_clears_worker_id_on_retry(self, spool_db, sample_pdf):
        """worker_id cleared when task retried."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)
        coordinator.claim_task(worker_id="worker-1")

        coordinator.fail_task(job_id, "Temporary error")

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT worker_id FROM jobs WHERE id = ?", (job_id,)
            )
            worker_id = cursor.fetchone()[0]

        assert worker_id is None

    def test_fail_task_clears_lease_on_retry(self, spool_db, sample_pdf):
        """leased_until cleared when task retried."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)
        coordinator.claim_task()

        coordinator.fail_task(job_id, "Temporary error")

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT leased_until FROM jobs WHERE id = ?", (job_id,)
            )
            leased_until = cursor.fetchone()[0]

        assert leased_until is None


class TestReclaimExpiredLeases:
    """Tests for SpoolCoordinator.reclaim_expired_leases()."""

    def test_reclaim_expired_leases_finds_expired(self, spool_db, sample_pdf):
        """Finds jobs with leased_until < NOW."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)
        coordinator.claim_task(lease_seconds=1)

        # Wait for lease to expire
        import time
        time.sleep(1.5)

        # Should find the expired job
        count = coordinator.reclaim_expired_leases(grace_seconds=0)
        assert count >= 1

    def test_reclaim_expired_leases_returns_count(self, spool_db, tmp_path):
        """Returns count of reclaimed tasks."""
        coordinator = SpoolCoordinator(spool_db)

        # Create multiple jobs
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content 1")
        file2.write_text("content 2")

        coordinator.enqueue_file(file1)
        coordinator.enqueue_file(file2)

        # Claim both with short leases
        coordinator.claim_task(worker_id="worker-1", lease_seconds=1)
        coordinator.claim_task(worker_id="worker-2", lease_seconds=1)

        # Wait for leases to expire
        import time
        time.sleep(1.5)

        # Reclaim should find both
        count = coordinator.reclaim_expired_leases(grace_seconds=0)
        assert count == 2

    def test_reclaim_expired_leases_moves_to_pending(self, spool_db, sample_pdf):
        """Reclaimed jobs status → 'pending'."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)
        coordinator.claim_task(lease_seconds=1)

        # Wait for lease to expire
        import time
        time.sleep(1.5)

        coordinator.reclaim_expired_leases(grace_seconds=0)

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            )
            status = cursor.fetchone()[0]

        assert status == "pending"

    def test_reclaim_expired_leases_clears_worker_id(self, spool_db, sample_pdf):
        """Reclaimed jobs have worker_id cleared."""
        coordinator = SpoolCoordinator(spool_db)
        job_id = coordinator.enqueue_file(sample_pdf)
        coordinator.claim_task(worker_id="stale-worker", lease_seconds=1)

        import time
        time.sleep(1.5)

        coordinator.reclaim_expired_leases(grace_seconds=0)

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT worker_id FROM jobs WHERE id = ?", (job_id,)
            )
            worker_id = cursor.fetchone()[0]

        assert worker_id is None

    def test_reclaim_expired_leases_skips_valid_leases(self, spool_db, tmp_path):
        """Does not reclaim jobs with valid leases."""
        coordinator = SpoolCoordinator(spool_db)

        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content 1")
        file2.write_text("content 2")

        job_id_1 = coordinator.enqueue_file(file1)
        job_id_2 = coordinator.enqueue_file(file2)

        # Claim with short lease
        coordinator.claim_task(worker_id="worker-1", lease_seconds=1)

        # Claim with long lease
        coordinator.claim_task(worker_id="worker-2", lease_seconds=600)

        # Wait for first to expire
        import time
        time.sleep(1.5)

        # Reclaim should only get the first
        count = coordinator.reclaim_expired_leases(grace_seconds=0)
        assert count == 1

        # Second job should still be processing
        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id_2,)
            )
            status = cursor.fetchone()[0]

        assert status == "processing"


class TestEndToEndWorkflow:
    """Integration tests for complete workflow."""

    def test_spool_workflow_end_to_end(self, spool_db, sample_pdf):
        """Full workflow: enqueue → move_to_pending → claim → complete."""
        coordinator = SpoolCoordinator(spool_db)

        # 1. Enqueue file
        job_id = coordinator.enqueue_file(sample_pdf)
        assert job_id is not None

        # 2. Verify pending status
        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            )
            status = cursor.fetchone()[0]
        assert status == "pending"

        # 3. Claim task
        claimed = coordinator.claim_task(worker_id="test-worker")
        assert claimed is not None
        assert claimed["job_id"] == job_id
        assert claimed["payload"]["original_filename"] == sample_pdf.name

        # 4. Verify processing status
        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            )
            status = cursor.fetchone()[0]
        assert status == "processing"

        # 5. Complete task
        result = coordinator.complete_task(job_id)
        assert result is True

        # 6. Verify done status
        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT status FROM jobs WHERE id = ?", (job_id,)
            )
            status = cursor.fetchone()[0]
        assert status == "done"

    def test_spool_workflow_with_failure_and_retry(self, spool_db, sample_pdf):
        """Workflow with failure, retry, and eventual completion."""
        coordinator = SpoolCoordinator(spool_db)

        # 1. Enqueue and claim
        job_id = coordinator.enqueue_file(sample_pdf)
        claimed = coordinator.claim_task(worker_id="worker-1")

        # 2. Fail the task (attempt 1)
        coordinator.fail_task(job_id, "Timeout error")

        # Should be back to pending with attempts=1
        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT status, attempts FROM jobs WHERE id = ?", (job_id,)
            )
            status, attempts = cursor.fetchone()
        assert status == "pending"
        assert attempts == 1

        # 3. Claim again
        claimed = coordinator.claim_task(worker_id="worker-2")
        assert claimed["job_id"] == job_id

        # 4. Complete successfully
        coordinator.complete_task(job_id)

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT status, attempts FROM jobs WHERE id = ?", (job_id,)
            )
            status, attempts = cursor.fetchone()
        assert status == "done"
        assert attempts == 1  # Not incremented since we completed

    def test_spool_workflow_max_retries_exhausted(self, spool_db, sample_pdf):
        """Task fails permanently after 3 failed attempts."""
        coordinator = SpoolCoordinator(spool_db)

        job_id = coordinator.enqueue_file(sample_pdf)

        # Fail 3 times
        for i in range(3):
            coordinator.claim_task(worker_id=f"worker-{i+1}")
            coordinator.fail_task(job_id, f"Error {i+1}")

        # After 3rd failure, should still be retryable
        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT status, attempts FROM jobs WHERE id = ?", (job_id,)
            )
            status, attempts = cursor.fetchone()
        assert status == "pending"
        assert attempts == 3

        # 4th failure should mark as failed
        coordinator.claim_task(worker_id="worker-4")
        coordinator.fail_task(job_id, "Final failure")

        with sqlite3.connect(spool_db) as conn:
            cursor = conn.execute(
                "SELECT status, attempts, error FROM jobs WHERE id = ?",
                (job_id,),
            )
            status, attempts, error = cursor.fetchone()
        assert status == "failed"
        assert attempts == 4
        assert error == "Final failure"
