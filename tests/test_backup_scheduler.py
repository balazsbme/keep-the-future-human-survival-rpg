# SPDX-License-Identifier: GPL-3.0-or-later

import sqlite3

from evaluations.backup_scheduler import perform_sqlite_backup


def test_perform_sqlite_backup_accepts_directory(tmp_path):
    db_path = tmp_path / "game.sqlite"
    backup_dir = tmp_path / "backups"
    sqlite3.connect(db_path).close()

    perform_sqlite_backup(db_path, backup_dir)

    backup_files = list(backup_dir.glob("game-*.sqlite"))
    assert len(backup_files) == 1
