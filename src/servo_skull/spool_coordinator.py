"""Spool Coordinator: persistent job queue management for extraction pipelines."""
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from ._utils import calculate_checksum, retry, setup_logging

logger = setup_logging(__name__)


class SpoolCoordinator:
    """Manages atomic POSIX spooling architecture for extraction tasks."""

    def __init__(self, db_path: Path | str):
        """Initialize SpoolCoordinator with database path.

        Args:
            db_path: Path to SQLite database (jobs.sqlite)
        """
        self.db_path = Path(db_path)

    @retry(max_attempts=3, delay=1.0)
    def enqueue_file(self, file_path: Path | str, pipeline: str = "extraction") -> str:
        """Enqueue a file for extraction processing.

        Args:
            file_path: Path to file to extract
            pipeline: Pipeline name (default: "extraction")

        Returns:
            str: Job ID (UUID)

        Raises:
            ValueError: If file_path doesn't exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValueError(f"File does not exist: {file_path}")

        # Read file and calculate checksum
        file_content = file_path.read_bytes()
        checksum = calculate_checksum(file_content)

        # Build payload
        payload = {
            "file_path": str(file_path),
            "original_filename": file_path.name,
            "file_size": len(file_content),
            "checksum": checksum,
        }

        # Generate job ID
        job_id = str(uuid4())

        # Insert into database
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO jobs
                (id, pipeline, status, payload, attempts, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))
                """,
                (
                    job_id,
                    pipeline,
                    "pending",
                    json.dumps(payload),
                    0,
                ),
            )
            conn.commit()
            logger.info(f"Enqueued job {job_id} for pipeline={pipeline}, file={file_path.name}")
            return job_id
        finally:
            conn.close()

    @retry(max_attempts=3, delay=1.0)
    def move_to_pending(self, job_id: str) -> bool:
        """Move a job from any status to pending.

        Args:
            job_id: UUID from enqueue_file()

        Returns:
            bool: True if updated, False if not found
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE jobs SET status = ?, updated_at = datetime('now') WHERE id = ?",
                ("pending", job_id),
            )
            conn.commit()
            updated = cursor.rowcount > 0
            if updated:
                logger.info(f"Moved job {job_id} to pending")
            return updated
        finally:
            conn.close()

    @retry(max_attempts=3, delay=1.0)
    def claim_task(
        self,
        pipeline: str = "extraction",
        worker_id: str = "default",
        lease_seconds: int = 600,
    ) -> dict[str, Any] | None:
        """Claim a pending task for processing.

        Args:
            pipeline: Pipeline name
            worker_id: Identifier for this worker process
            lease_seconds: Lock duration (default 600 = 10 minutes)

        Returns:
            dict with job_id, pipeline, payload or None if no pending tasks
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.isolation_level = "IMMEDIATE"  # Serializable for atomicity
            cursor = conn.cursor()

            # Find oldest pending job
            cursor.execute(
                """
                SELECT id, pipeline, payload FROM jobs
                WHERE pipeline = ? AND status = 'pending'
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (pipeline,),
            )
            row = cursor.fetchone()

            if not row:
                return None

            job_id, pipe, payload_json = row

            # Calculate lease expiry
            leased_until = (datetime.now() + timedelta(seconds=lease_seconds)).isoformat()

            # Update to processing
            cursor.execute(
                """
                UPDATE jobs
                SET status = ?, worker_id = ?, leased_until = ?, updated_at = datetime('now')
                WHERE id = ?
                """,
                ("processing", worker_id, leased_until, job_id),
            )
            conn.commit()

            # Parse payload
            payload = json.loads(payload_json)

            logger.info(
                f"Claimed job {job_id} for worker={worker_id}, "
                f"lease_until={leased_until}"
            )

            return {
                "job_id": job_id,
                "pipeline": pipe,
                "payload": payload,
            }
        finally:
            conn.close()

    @retry(max_attempts=3, delay=1.0)
    def complete_task(self, job_id: str) -> bool:
        """Mark a processing task as done.

        Args:
            job_id: UUID from claim_task()

        Returns:
            bool: True if updated, False if not found
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE jobs
                SET status = ?, worker_id = NULL, leased_until = NULL, updated_at = datetime('now')
                WHERE id = ?
                """,
                ("done", job_id),
            )
            conn.commit()
            updated = cursor.rowcount > 0
            if updated:
                logger.info(f"Completed job {job_id}")
            return updated
        finally:
            conn.close()

    @retry(max_attempts=3, delay=1.0)
    def fail_task(self, job_id: str, error_message: str = "") -> bool:
        """Mark a processing task as failed or retry.

        If attempts <= 3, returns to 'pending' for retry.
        If attempts > 3, marks as 'failed' with error message.

        Args:
            job_id: UUID from claim_task()
            error_message: Human-readable error description

        Returns:
            bool: True if updated, False if not found
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()

            # Get current attempts
            cursor.execute("SELECT attempts FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if not row:
                return False

            current_attempts = row[0]
            new_attempts = current_attempts + 1

            if new_attempts > 3:
                # Mark as failed
                cursor.execute(
                    """
                    UPDATE jobs
                    SET status = ?, attempts = ?, error = ?, worker_id = NULL,
                        leased_until = NULL, updated_at = datetime('now')
                    WHERE id = ?
                    """,
                    ("failed", new_attempts, error_message, job_id),
                )
                logger.error(
                    f"Job {job_id} failed after {new_attempts} attempts: {error_message}"
                )
            else:
                # Retry: return to pending
                cursor.execute(
                    """
                    UPDATE jobs
                    SET status = ?, attempts = ?, worker_id = NULL, leased_until = NULL,
                        updated_at = datetime('now')
                    WHERE id = ?
                    """,
                    ("pending", new_attempts, job_id),
                )
                logger.warning(
                    f"Job {job_id} failed (attempt {new_attempts}/3), "
                    f"returning to pending: {error_message}"
                )

            conn.commit()
            return True
        finally:
            conn.close()

    @retry(max_attempts=3, delay=1.0)
    def reclaim_expired_leases(
        self, pipeline: str = "extraction", grace_seconds: int = 60
    ) -> int:
        """Reclaim processing tasks with expired leases.

        Args:
            pipeline: Pipeline name
            grace_seconds: Additional buffer before reclaim (default 60)

        Returns:
            int: Count of reclaimed tasks
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.cursor()

            # Find expired leases
            now = datetime.now()
            grace_cutoff = (now - timedelta(seconds=grace_seconds)).isoformat()

            cursor.execute(
                """
                SELECT id FROM jobs
                WHERE pipeline = ? AND status = 'processing'
                  AND leased_until < ?
                """,
                (pipeline, grace_cutoff),
            )
            expired_jobs = cursor.fetchall()

            reclaimed_count = 0
            for (job_id,) in expired_jobs:
                cursor.execute(
                    """
                    UPDATE jobs
                    SET status = ?, worker_id = NULL, leased_until = NULL,
                        updated_at = datetime('now')
                    WHERE id = ?
                    """,
                    ("pending", job_id),
                )
                reclaimed_count += 1
                logger.info(f"Reclaimed expired lease for job {job_id}")

            conn.commit()
            logger.info(f"Reclaimed {reclaimed_count} expired leases for pipeline={pipeline}")
            return reclaimed_count
        finally:
            conn.close()
