"""Backup scheduler for SQLite evaluation database."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from rpg.session_monitor import SessionActivityMonitor, SessionActivitySnapshot
from .sqlite3_connector import DatabaseLockedError, sqlite_write_lock

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
    cleanup_after_backup: bool = True


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
        cleanup_after_backup=bool(payload.get("cleanup_after_backup", True)),
    )


def build_trigger(config: BackupSchedulerConfig) -> BackupTriggerCondition:
    """Instantiate a trigger condition from configuration."""

    if config.trigger_type == "closed_sessions_threshold":
        return ClosedSessionsThresholdCondition(config.closed_sessions_threshold)
    raise ValueError(f"Unknown trigger type {config.trigger_type!r}")


def _resolve_backup_target_parts(db_path: Path, backup_path: Path) -> tuple[Path, str, str]:
    if backup_path.exists() and backup_path.is_dir():
        backup_dir = backup_path
        stem = db_path.stem
        suffix = ".db"
        logger.info(
            "Backup path %s is a directory; writing backup into it",
            backup_path,
        )
    elif not backup_path.exists() and backup_path.suffix == "":
        backup_dir = backup_path
        stem = db_path.stem
        suffix = ".db"
        logger.info(
            "Backup path %s has no suffix; treating as directory target",
            backup_path,
        )
    else:
        backup_dir = backup_path.parent
        stem = backup_path.stem
        suffix = backup_path.suffix or ".db"
    return backup_dir, stem, suffix


def _resolve_backup_path(db_path: Path, backup_path: Path) -> Path:
    """Return the file path to use for the backup output."""

    backup_dir, stem, suffix = _resolve_backup_target_parts(db_path, backup_path)

    timestamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    for attempt in range(100):
        suffix_suffix = f"-{attempt}" if attempt else ""
        candidate = backup_dir / f"{stem}-{timestamp}{suffix_suffix}{suffix}"
        if candidate.exists():
            if candidate.is_dir():
                raise IsADirectoryError(
                    f"Backup path {candidate} is a directory; expected a file path"
                )
            continue
        return candidate
    raise FileExistsError(
        f"Could not find an available backup file name in {backup_dir}"
    )


def _pending_marker_path(db_path: Path, backup_dir: Path) -> Path:
    return backup_dir / f"{db_path.stem}.backup-pending.json"


def _load_pending_marker(path: Path) -> tuple[Path, Path] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    temp = payload.get("temp")
    final = payload.get("final")
    if not isinstance(temp, str) or not isinstance(final, str):
        return None
    return Path(temp), Path(final)


def _write_pending_marker(path: Path, temp_path: Path, final_path: Path) -> None:
    payload = {"temp": str(temp_path), "final": str(final_path)}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _execute_with_retry(
    connection: sqlite3.Connection,
    statement: str,
    parameters: tuple | list | None = None,
    *,
    max_attempts: int = 5,
    base_sleep_seconds: float = 0.1,
) -> None:
    for attempt in range(1, max_attempts + 1):
        try:
            connection.execute(statement, parameters or ())
            return
        except sqlite3.OperationalError as exc:
            if "database is locked" not in str(exc).lower() or attempt == max_attempts:
                raise
            delay = base_sleep_seconds * (2 ** (attempt - 1))
            logger.warning(
                "SQLite database locked while executing %s; retrying in %.2fs",
                statement,
                delay,
            )
            time.sleep(delay)


def _cleanup_sqlite_database(connection: sqlite3.Connection) -> None:
    foreign_keys_enabled = connection.execute("PRAGMA foreign_keys").fetchone()[0]
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA foreign_keys = OFF")
    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        for (table_name,) in rows:
            _execute_with_retry(
                connection,
                f"DELETE FROM {_quote_identifier(table_name)}",
            )
        sequence_present = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'"
        ).fetchone()
        if sequence_present:
            _execute_with_retry(connection, "DELETE FROM sqlite_sequence")
    finally:
        if foreign_keys_enabled:
            connection.execute("PRAGMA foreign_keys = ON")


def perform_sqlite_backup(
    db_path: Path,
    backup_path: Path,
    *,
    cleanup_after_backup: bool = True,
) -> None:
    """Run a VACUUM INTO backup for the SQLite database."""

    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found at {db_path}")
    backup_dir, _, _ = _resolve_backup_target_parts(db_path, backup_path)
    backup_dir.mkdir(parents=True, exist_ok=True)
    pending_marker = _pending_marker_path(db_path, backup_dir)
    pending_info = _load_pending_marker(pending_marker) if pending_marker.exists() else None

    def _cleanup_with_retry(connection: sqlite3.Connection) -> None:
        if not cleanup_after_backup:
            return
        for cleanup_attempt in range(1, 6):
            try:
                _cleanup_sqlite_database(connection)
                connection.commit()
                return
            except sqlite3.OperationalError as exc:
                if "database is locked" not in str(exc).lower():
                    raise
                if cleanup_attempt >= 5:
                    raise
                delay = 0.2 * cleanup_attempt
                logger.warning(
                    "SQLite cleanup locked; retrying in %.1fs",
                    delay,
                )
                time.sleep(delay)

    def _finalize_pending(temp_path: Path, final_path: Path) -> None:
        for lock_attempt in range(1, 11):
            try:
                with sqlite_write_lock(db_path):
                    connection = sqlite3.connect(db_path, timeout=30)
                    try:
                        _cleanup_with_retry(connection)
                    finally:
                        connection.close()
                temp_path.replace(final_path)
                pending_marker.unlink(missing_ok=True)
                return
            except DatabaseLockedError:
                if lock_attempt >= 10:
                    raise
                delay = 0.5 * lock_attempt
                logger.warning(
                    "SQLite write lock busy while finalizing backup; retrying in %.1fs",
                    delay,
                )
                time.sleep(delay)

    if pending_info:
        temp_path, final_path = pending_info
        if temp_path.exists():
            _finalize_pending(temp_path, final_path)
            return
        pending_marker.unlink(missing_ok=True)

    backup_file = _resolve_backup_path(db_path, backup_path)
    temp_file = backup_file.with_suffix(backup_file.suffix + ".partial")
    for attempt in range(1, 11):
        try:
            with sqlite_write_lock(db_path):
                connection = sqlite3.connect(db_path, timeout=30)
                try:
                    connection.execute("VACUUM INTO ?", (str(temp_file),))
                    _write_pending_marker(pending_marker, temp_file, backup_file)
                    _cleanup_with_retry(connection)
                finally:
                    connection.close()
            temp_file.replace(backup_file)
            pending_marker.unlink(missing_ok=True)
            return
        except DatabaseLockedError:
            if attempt >= 10:
                raise
            delay = 0.5 * attempt
            logger.warning(
                "SQLite write lock busy while backing up; retrying in %.1fs",
                delay,
            )
            time.sleep(delay)


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
        cleanup_after_backup: bool,
    ) -> None:
        self.db_path = db_path
        self.backup_path = backup_path
        self.trigger = trigger
        self.session_monitor = session_monitor
        self.session_inactive_seconds = session_inactive_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.cleanup_after_backup = cleanup_after_backup
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
        if snapshot.closed_since_last_backup < 1:
            return False
        if snapshot.active_session_count > 0:
            logger.info(
                "Backup skipped; %d active session(s) remain",
                snapshot.active_session_count,
            )
            return False
        if not self.trigger.should_trigger(snapshot):
            return False
        logger.info(
            "Backup triggered after %d closed session(s); %d active session(s)",
            snapshot.closed_since_last_backup,
            snapshot.active_session_count,
        )
        perform_sqlite_backup(
            self.db_path,
            self.backup_path,
            cleanup_after_backup=self.cleanup_after_backup,
        )
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
