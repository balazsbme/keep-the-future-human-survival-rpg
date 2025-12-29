"""Backup scheduler for SQLite evaluation database."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from rpg.session_monitor import SessionActivityMonitor, SessionActivitySnapshot

logger = logging.getLogger(__name__)


class BackupTriggerCondition(Protocol):
    """Interface for backup trigger conditions."""

    def should_trigger(self, snapshot: SessionActivitySnapshot) -> bool:
        """Return ``True`` when a backup should be triggered."""


@dataclass(frozen=True)
class BackupSchedulerConfig:
    enabled: bool = True
    poll_interval_seconds: float = 30.0
    session_inactive_seconds: float = 600.0
    trigger_type: str = "closed_sessions_threshold"
    closed_sessions_threshold: int = 5


class ClosedSessionsThresholdCondition:
    """Trigger when closed sessions since last backup reach a threshold."""

    def __init__(self, threshold: int) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self.threshold = threshold

    def should_trigger(self, snapshot: SessionActivitySnapshot) -> bool:
        return snapshot.closed_since_last_backup >= self.threshold


def load_backup_scheduler_config(path: Path) -> BackupSchedulerConfig:
    """Load backup scheduler configuration from a YAML file."""

    if not path.exists():
        logger.info("Backup scheduler config missing at %s; using defaults", path)
        return BackupSchedulerConfig()
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        logger.info("Backup scheduler config at %s malformed; using defaults", path)
        return BackupSchedulerConfig()
    trigger = payload.get("trigger", {}) if isinstance(payload.get("trigger", {}), dict) else {}
    return BackupSchedulerConfig(
        enabled=bool(payload.get("enabled", True)),
        poll_interval_seconds=float(payload.get("poll_interval_seconds", 30.0)),
        session_inactive_seconds=float(payload.get("session_inactive_seconds", 600.0)),
        trigger_type=str(trigger.get("type", "closed_sessions_threshold")),
        closed_sessions_threshold=int(trigger.get("closed_sessions_threshold", 5)),
    )


def build_trigger(config: BackupSchedulerConfig) -> BackupTriggerCondition:
    """Instantiate a trigger condition from configuration."""

    if config.trigger_type == "closed_sessions_threshold":
        return ClosedSessionsThresholdCondition(config.closed_sessions_threshold)
    raise ValueError(f"Unknown trigger type {config.trigger_type!r}")


def perform_sqlite_backup(db_path: Path, backup_path: Path) -> None:
    """Run a VACUUM INTO backup for the SQLite database."""

    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found at {db_path}")
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if backup_path.exists():
        backup_path.unlink()
    connection = sqlite3.connect(db_path, timeout=30)
    try:
        connection.execute("VACUUM INTO ?", (str(backup_path),))
    finally:
        connection.close()


class BackupScheduler:
    """Poll session activity and trigger SQLite backups."""

    def __init__(
        self,
        *,
        db_path: Path,
        backup_path: Path,
        trigger: BackupTriggerCondition,
        session_monitor: SessionActivityMonitor,
        session_inactive_seconds: float,
        poll_interval_seconds: float,
    ) -> None:
        self.db_path = db_path
        self.backup_path = backup_path
        self.trigger = trigger
        self.session_monitor = session_monitor
        self.session_inactive_seconds = session_inactive_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="backup-scheduler", daemon=True)
        self._thread.start()
        logger.info("Backup scheduler started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is None:
            return
        self._thread.join(timeout=5)
        self._thread = None

    def run_once(self) -> bool:
        """Check triggers once and perform a backup if needed."""

        self.session_monitor.close_inactive_sessions(self.session_inactive_seconds)
        snapshot = self.session_monitor.snapshot()
        if not self.trigger.should_trigger(snapshot):
            return False
        logger.info(
            "Backup triggered after %d closed session(s); %d active session(s)",
            snapshot.closed_since_last_backup,
            snapshot.active_session_count,
        )
        perform_sqlite_backup(self.db_path, self.backup_path)
        self.session_monitor.reset_for_backup()
        return True

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception:
                logger.exception("Backup scheduler encountered an error")
            self._stop_event.wait(self.poll_interval_seconds)


__all__ = [
    "BackupScheduler",
    "BackupSchedulerConfig",
    "BackupTriggerCondition",
    "ClosedSessionsThresholdCondition",
    "build_trigger",
    "load_backup_scheduler_config",
    "perform_sqlite_backup",
]
