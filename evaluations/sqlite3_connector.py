"""Utility helpers for interacting with the SQLite evaluation database."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping

_DDL_PATH = Path(__file__).with_name("sqlite3_db.ddl")
_DEFAULT_DB_PATH = Path(os.environ.get("EVALUATION_SQLITE_PATH", "/var/lib/sqlite/main.db"))


def _ensure_directory(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def sanitize_identifier(name: str) -> str:
    """Return a SQLite-safe identifier derived from ``name``."""

    value = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower())
    value = value.strip("_") or "value"
    if value[0].isdigit():
        value = f"c_{value}"
    return value


class SQLiteConnector:
    """High level helper that owns the SQLite connection for evaluations."""

    def __init__(self, db_path: Path | str | None = None, ddl_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or _DEFAULT_DB_PATH)
        self.ddl_path = Path(ddl_path or _DDL_PATH)
        _ensure_directory(self.db_path)
        self._connection: sqlite3.Connection | None = None
        self._initialised = False

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path)
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            self._initialised = False

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        cur = self.connection.cursor()
        try:
            yield cur
        finally:
            cur.close()

    def initialise(self) -> None:
        """Execute the DDL script once per connector lifetime."""

        if self._initialised:
            return
        script = self.ddl_path.read_text(encoding="utf-8")
        self.connection.executescript(script)
        self._initialised = True

    def commit(self) -> None:
        self.connection.commit()

    # Column helpers -----------------------------------------------------
    def _table_columns(self, table: str) -> Dict[str, str]:
        with self.cursor() as cur:
            cur.execute(f"PRAGMA table_info({table})")
            return {row[1]: row[2] for row in cur.fetchall()}

    def ensure_columns(self, table: str, columns: Mapping[str, str]) -> None:
        existing = self._table_columns(table)
        for column, declaration in columns.items():
            if column not in existing:
                with self.cursor() as cur:
                    cur.execute(
                        f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
                    )

    # Serialisation helpers ----------------------------------------------
    @staticmethod
    def _serialise_json(payload: object) -> str:
        if is_dataclass(payload):
            payload = asdict(payload)
        return json.dumps(payload, sort_keys=True)

    @staticmethod
    def _prepare_payload(data: MutableMapping[str, object]) -> MutableMapping[str, object]:
        cleaned: Dict[str, object] = {}
        for key, value in data.items():
            if is_dataclass(value):
                cleaned[key] = asdict(value)
            else:
                cleaned[key] = value
        return cleaned

    def _execute_insert(self, table: str, values: Mapping[str, object]) -> int:
        if not values:
            raise ValueError("insert payload cannot be empty")
        columns = list(values.keys())
        placeholders = ", ".join(["?"] * len(columns))
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        with self.cursor() as cur:
            cur.execute(sql, [values[column] for column in columns])
            return int(cur.lastrowid)

    # Public API ---------------------------------------------------------
    def insert_execution(self, values: Mapping[str, object]) -> int:
        payload = dict(self._prepare_payload(dict(values)))
        if "config_json" in payload and not isinstance(payload["config_json"], str):
            payload["config_json"] = self._serialise_json(payload["config_json"])
        return self._execute_insert("executions", payload)

    def insert_action(self, values: Mapping[str, object]) -> int:
        payload = dict(values)
        if "option_json" in payload and not isinstance(payload["option_json"], str):
            payload["option_json"] = self._serialise_json(payload["option_json"])
        if "targets_json" in payload and not isinstance(payload["targets_json"], str):
            payload["targets_json"] = self._serialise_json(payload["targets_json"])
        return self._execute_insert("actions", payload)

    def insert_assessment(self, values: Mapping[str, object]) -> int:
        payload = dict(values)
        if "assessment_json" in payload and not isinstance(payload["assessment_json"], str):
            payload["assessment_json"] = self._serialise_json(payload["assessment_json"])
        return self._execute_insert("assessments", payload)

    def insert_credibility(self, values: Mapping[str, object]) -> int:
        payload = dict(values)
        if "credibility_json" in payload and not isinstance(payload["credibility_json"], str):
            payload["credibility_json"] = self._serialise_json(payload["credibility_json"])
        return self._execute_insert("credibility", payload)

    def insert_result(self, values: Mapping[str, object]) -> int:
        payload = dict(values)
        if "successful_execution" in payload:
            payload["successful_execution"] = int(bool(payload["successful_execution"]))
        return self._execute_insert("results", payload)

    # Dynamic schema helpers --------------------------------------------
    def ensure_assessment_columns(self, faction_triplets: Mapping[str, int]) -> None:
        columns = {
            f"{sanitize_identifier(faction)}_triplet_{index}": "INTEGER"
            for faction, count in faction_triplets.items()
            for index in range(1, count + 1)
        }
        if columns:
            self.ensure_columns("assessments", columns)

    def ensure_credibility_columns(self, targets: Iterable[str]) -> None:
        columns = {
            f"credibility_{sanitize_identifier(target)}": "INTEGER"
            for target in targets
        }
        if columns:
            self.ensure_columns("credibility", columns)

    def ensure_dynamic_schema(self, faction_triplets: Mapping[str, int], credibility_targets: Iterable[str]) -> None:
        self.ensure_assessment_columns(faction_triplets)
        self.ensure_credibility_columns(credibility_targets)


@contextmanager
def sqlite_connector(db_path: Path | str | None = None) -> Iterator[SQLiteConnector]:
    connector = SQLiteConnector(db_path=db_path)
    try:
        connector.initialise()
        yield connector
        connector.commit()
    finally:
        connector.close()


__all__ = ["SQLiteConnector", "sqlite_connector", "sanitize_identifier"]
