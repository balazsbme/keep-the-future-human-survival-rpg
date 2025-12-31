# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sqlite3
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluations.backup_scheduler import (
    BackupScheduler,
    ClosedSessionsThresholdCondition,
    perform_sqlite_backup,
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
