# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sqlite3
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluations.backup_scheduler import perform_sqlite_backup


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
    connection.close()

    assert tables == []
