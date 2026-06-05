"""Egress synchronization utility for Servo-Skull."""
import os
import shutil
import subprocess
import tomllib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_sync_config() -> dict:
    """Load sync.toml configuration."""
    config_path = Path(__file__).parent.parent.parent.parent.parent / "config" / "sync.toml"
    if not config_path.exists():
        logger.warning(f"Sync config not found at {config_path}")
        return {}
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def sync_file(source_file: Path) -> None:
    """
    Sync a generated file locally and remotely.

    Args:
        source_file: Path to the local file to sync.
    """
    source_file = Path(source_file)
    if not source_file.exists():
        logger.error(f"Sync source file does not exist: {source_file}")
        return

    config = load_sync_config()
    if not config:
        logger.warning("No sync configuration loaded, skipping sync.")
        return

    # Local Sync
    local_cfg = config.get("local", {})
    if local_cfg.get("enabled", False):
        local_path = Path(local_cfg.get("path", "./workspace/artifacts"))
        local_path.mkdir(parents=True, exist_ok=True)
        dest_file = local_path / source_file.name
        if source_file.resolve() != dest_file.resolve():
            shutil.copy2(source_file, dest_file)
            logger.info(f"Synced file locally to {dest_file}")

    # Remote Sync
    remote_cfg = config.get("remote", {})
    if remote_cfg.get("enabled", False):
        host = remote_cfg.get("host", "localhost")
        user = remote_cfg.get("user", "user")
        remote_path = remote_cfg.get("path", "/srv/inbox")

        logger.info(f"Syncing file to remote host: {user}@{host}:{remote_path}")
        try:
            cmd = ["scp", str(source_file), f"{user}@{host}:{remote_path}/"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Remote sync successful: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.strip()
            logger.error(f"Remote sync failed: {err_msg}")
            raise RuntimeError(f"SCP sync failed: {err_msg}") from e
