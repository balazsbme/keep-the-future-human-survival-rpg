# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sqlite3
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluations.backup_scheduler import (
    BackupScheduler,
    ClosedSessionsThresholdCondition,
    _execute_with_retry,
    perform_sqlite_backup,
)
from evaluations.sqlite3_connector import (
    DatabaseLockedError,
    SQLiteConnector,
    sqlite_write_lock,
)
from rpg.session_monitor import SessionActivityMonitor


def test_perform_sqlite_backup_accepts_directory(tmp_path):
    db_path = tmp_path / "game.sqlite"
    backup_dir = tmp_path / "backups"
    sqlite3.connect(db_path).close()

    perform_sqlite_backup(db_path, backup_dir)

    backup_files = list(backup_dir.glob("game-*.db"))
    assert len(backup_files) == 1


def test_perform_sqlite_backup_cleans_up_db(tmp_path):
    db_path = tmp_path / "game.sqlite"
    backup_dir = tmp_path / "backups"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)")
    connection.execute("INSERT INTO test_table (name) VALUES ('Ada')")
    connection.commit()
    connection.close()

    perform_sqlite_backup(db_path, backup_dir, cleanup_after_backup=True)

    connection = sqlite3.connect(db_path)
    tables = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    rows = connection.execute("SELECT * FROM test_table").fetchall()
    connection.close()

    assert tables == [("test_table",)]
    assert rows == []


def test_perform_sqlite_backup_retries_cleanup_on_lock(tmp_path):
    db_path = tmp_path / "game.sqlite"
    backup_dir = tmp_path / "backups"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)")
    connection.execute("INSERT INTO test_table (name) VALUES ('Ada')")
    connection.commit()
    connection.close()

    import evaluations.backup_scheduler as backup_scheduler

    original_cleanup = backup_scheduler._cleanup_sqlite_database
    attempts = {"count": 0}

    def flaky_cleanup(connection):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise sqlite3.OperationalError("database is locked")
        return original_cleanup(connection)

    with patch(
        "evaluations.backup_scheduler._cleanup_sqlite_database",
        side_effect=flaky_cleanup,
    ):
        perform_sqlite_backup(db_path, backup_dir, cleanup_after_backup=True)

    connection = sqlite3.connect(db_path)
    rows = connection.execute("SELECT * FROM test_table").fetchall()
    connection.close()

    assert rows == []


def test_backup_uses_single_backup_file_after_cleanup_retry(tmp_path):
    db_path = tmp_path / "game.sqlite"
    backup_dir = tmp_path / "backups"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)")
    connection.execute("INSERT INTO test_table (name) VALUES ('Ada')")
    connection.commit()
    connection.close()

    import evaluations.backup_scheduler as backup_scheduler

    original_cleanup = backup_scheduler._cleanup_sqlite_database

    def always_locked(_connection):
        raise sqlite3.OperationalError("database is locked")

    with patch(
        "evaluations.backup_scheduler._cleanup_sqlite_database",
        side_effect=always_locked,
    ):
        with pytest.raises(sqlite3.OperationalError):
            perform_sqlite_backup(db_path, backup_dir, cleanup_after_backup=True)

    assert list(backup_dir.glob("game-*.db")) == []
    pending_files = list(backup_dir.glob("game.backup-pending.json"))
    assert len(pending_files) == 1
    temp_files = list(backup_dir.glob("game-*.db.partial"))
    assert len(temp_files) == 1

    with patch(
        "evaluations.backup_scheduler._cleanup_sqlite_database",
        side_effect=original_cleanup,
    ):
        perform_sqlite_backup(db_path, backup_dir, cleanup_after_backup=True)

    backup_files = list(backup_dir.glob("game-*.db"))
    assert len(backup_files) == 1
    assert list(backup_dir.glob("game.backup-pending.json")) == []


def test_execute_with_retry_retries_on_locked_database():
    connection = MagicMock()
    connection.execute.side_effect = [
        sqlite3.OperationalError("database is locked"),
        None,
    ]

    _execute_with_retry(connection, "DELETE FROM test_table", max_attempts=2, base_sleep_seconds=0)

    assert connection.execute.call_count == 2


def test_sqlite_write_lock_blocks_connector(tmp_path):
    db_path = tmp_path / "game.sqlite"
    sqlite3.connect(db_path).close()

    with sqlite_write_lock(db_path):
        connector = SQLiteConnector(db_path=db_path)
        with pytest.raises(DatabaseLockedError):
            connector.initialise()


def test_perform_sqlite_backup_respects_write_lock(tmp_path):
    db_path = tmp_path / "game.sqlite"
    backup_dir = tmp_path / "backups"
    sqlite3.connect(db_path).close()

    @contextmanager
    def _locked(*_args, **_kwargs):
        raise DatabaseLockedError("locked")
        yield  # pragma: no cover - defensive

    with patch("evaluations.backup_scheduler.sqlite_write_lock", _locked):
        with pytest.raises(DatabaseLockedError):
            perform_sqlite_backup(db_path, backup_dir)


def test_perform_sqlite_backup_succeeds_with_idle_connector(tmp_path):
    db_path = tmp_path / "game.sqlite"
    backup_dir = tmp_path / "backups"
    connector = SQLiteConnector(db_path=db_path)
    connector.initialise()

    perform_sqlite_backup(db_path, backup_dir)

    backup_files = list(backup_dir.glob("game-*.db"))
    assert len(backup_files) == 1
    connector.close()


def test_backup_scheduler_skips_when_active_sessions(tmp_path):
    db_path = tmp_path / "game.sqlite"
    backup_dir = tmp_path / "backups"
    sqlite3.connect(db_path).close()
    monitor = SessionActivityMonitor()
    monitor.register_session("session-one")
    monitor.register_session("session-two")
    monitor.mark_closed("session-one")
    scheduler = BackupScheduler(
        db_path=db_path,
        backup_path=backup_dir,
        trigger=ClosedSessionsThresholdCondition(1),
        session_monitor=monitor,
        session_inactive_seconds=9999,
        poll_interval_seconds=0.1,
        cleanup_after_backup=True,
    )

    with patch("evaluations.backup_scheduler.perform_sqlite_backup") as backup_mock:
        assert scheduler.run_once() is False
        backup_mock.assert_not_called()


def test_backup_scheduler_recovers_after_failed_backup(tmp_path):
    db_path = tmp_path / "game.sqlite"
    backup_dir = tmp_path / "backups"
    sqlite3.connect(db_path).close()
    monitor = SessionActivityMonitor()
    monitor.register_session("session-one")
    monitor.mark_closed("session-one")
    scheduler = BackupScheduler(
        db_path=db_path,
        backup_path=backup_dir,
        trigger=ClosedSessionsThresholdCondition(1),
        session_monitor=monitor,
        session_inactive_seconds=9999,
        poll_interval_seconds=0.1,
        cleanup_after_backup=True,
    )

    with patch(
        "evaluations.backup_scheduler.perform_sqlite_backup",
        side_effect=[RuntimeError("backup failed"), None],
    ) as backup_mock:
        with pytest.raises(RuntimeError):
            scheduler.run_once()
        assert scheduler.run_once() is True
        assert backup_mock.call_count == 2


def test_backup_scheduler_allows_new_backups_after_success(tmp_path):
    db_path = tmp_path / "game.sqlite"
    backup_dir = tmp_path / "backups"
    sqlite3.connect(db_path).close()
    monitor = SessionActivityMonitor()
    monitor.register_session("session-one")
    monitor.mark_closed("session-one")
    scheduler = BackupScheduler(
        db_path=db_path,
        backup_path=backup_dir,
        trigger=ClosedSessionsThresholdCondition(1),
        session_monitor=monitor,
        session_inactive_seconds=9999,
        poll_interval_seconds=0.1,
        cleanup_after_backup=True,
    )

    with patch("evaluations.backup_scheduler.perform_sqlite_backup") as backup_mock:
        assert scheduler.run_once() is True
        monitor.register_session("session-two")
        monitor.mark_closed("session-two")
        assert scheduler.run_once() is True
        assert backup_mock.call_count == 2
