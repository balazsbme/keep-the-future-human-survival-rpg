"""Utility helpers for interacting with the SQLite evaluation database."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import logging
import json
import os
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping
try:
    import fcntl
except ImportError:  # pragma: no cover - platform specific
    fcntl = None

logger = logging.getLogger(__name__)

_DDL_PATH = Path(__file__).with_name("sqlite3_db.ddl")
_DB_PATH_ENV = "EVALUATION_SQLITE_PATH"
_DEFAULT_DB_PATH = Path("/var/lib/sqlite/main.db")


def _default_db_path_from_env() -> Path:
    """Resolve the SQLite path from the environment at call time."""

    env_value = os.environ.get(_DB_PATH_ENV)
    if env_value:
        return Path(env_value)
    return _DEFAULT_DB_PATH


class DatabaseLockedError(RuntimeError):
    """Raised when the SQLite file lock indicates another writer is active."""


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

    def __init__(
        self,
        db_path: Path | str | None = None,
        ddl_path: Path | str | None = None,
        lock_path: Path | str | None = None,
        *,
        require_lock: bool = True,
    ) -> None:
        self.db_path = Path(db_path) if db_path else _default_db_path_from_env()
        self.ddl_path = Path(ddl_path or _DDL_PATH)
        self.lock_path = Path(lock_path or (self.db_path.with_suffix(self.db_path.suffix + ".lock")))
        self.require_lock = require_lock
        _ensure_directory(self.db_path)
        self._connection: sqlite3.Connection | None = None
        self._initialised = False
        self._lock = threading.RLock()
        self._lock_file = None
        self._blocked_by_lock = False

    def _acquire_interprocess_lock(self) -> None:
        if not self.require_lock:
            return
        if self._blocked_by_lock:
            raise DatabaseLockedError(
                f"Database locked via {self.lock_path}; another container is writing"
            )
        if self._lock_file is not None:
            return
        _ensure_directory(self.lock_path)
        logger.info("Attempting to acquire SQLite interprocess lock at %s", self.lock_path)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT)
        file_handle = os.fdopen(fd, "r+")
        try:
            if fcntl is not None:
                try:
                    fcntl.flock(file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:  # pragma: no cover - depends on runtime
                    file_handle.close()
                    self._blocked_by_lock = True
                    logger.info(
                        "SQLite lock at %s is held by another process", self.lock_path
                    )
                    raise DatabaseLockedError(
                        f"Database locked via {self.lock_path}; another container is writing"
                    ) from exc
            file_handle.seek(0)
            file_handle.truncate()
            file_handle.write("1")
            file_handle.flush()
            os.fsync(file_handle.fileno())
        except Exception:
            file_handle.close()
            raise
        self._lock_file = file_handle
        logger.info("Acquired SQLite interprocess lock at %s", self.lock_path)

    def _release_interprocess_lock(self) -> None:
        if self._lock_file is None:
            return
        try:
            self._lock_file.seek(0)
            self._lock_file.truncate()
            self._lock_file.write("0")
            self._lock_file.flush()
            os.fsync(self._lock_file.fileno())
            if fcntl is not None:
                try:
                    fcntl.flock(self._lock_file, fcntl.LOCK_UN)
                except OSError:  # pragma: no cover - depends on runtime
                    pass
        finally:
            self._lock_file.close()
            self._lock_file = None
            self._blocked_by_lock = False
            logger.info("Released SQLite interprocess lock at %s", self.lock_path)

    @property
    def connection(self) -> sqlite3.Connection:
        with self._lock:
            self._acquire_interprocess_lock()
            if self._connection is None:
                self._connection = sqlite3.connect(
                    self.db_path, check_same_thread=False
                )
                self._connection.row_factory = sqlite3.Row
            return self._connection

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
                self._initialised = False
            self._release_interprocess_lock()

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        self._lock.acquire()
        cur = self.connection.cursor()
        try:
            yield cur
        finally:
            cur.close()
            self._lock.release()

    def initialise(self) -> None:
        """Execute the DDL script once per connector lifetime."""

        with self._lock:
            if self._initialised:
                return
            script = self.ddl_path.read_text(encoding="utf-8")
            self.connection.executescript(script)
            self._initialised = True

    def commit(self) -> None:
        with self._lock:
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

    def _execute_insert(
        self,
        table: str,
        values: Mapping[str, object],
        *,
        return_value: object | None = None,
    ) -> object:
        if not values:
            raise ValueError("insert payload cannot be empty")
        columns = list(values.keys())
        placeholders = ", ".join(["?"] * len(columns))
        sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        with self.cursor() as cur:
            cur.execute(sql, [values[column] for column in columns])
            return return_value if return_value is not None else cur.lastrowid

    @staticmethod
    def _ensure_uuid(values: MutableMapping[str, object], key: str) -> str:
        if key not in values or values[key] is None:
            values[key] = str(uuid.uuid4())
        return str(values[key])

    # Public API ---------------------------------------------------------
    def insert_execution(self, values: Mapping[str, object]) -> str:
        payload = dict(self._prepare_payload(dict(values)))
        execution_id = self._ensure_uuid(payload, "execution_id")
        if "config_json" in payload and not isinstance(payload["config_json"], str):
            payload["config_json"] = self._serialise_json(payload["config_json"])
        return str(self._execute_insert("executions", payload, return_value=execution_id))

    def insert_action(self, values: Mapping[str, object]) -> str:
        payload = dict(values)
        action_id = self._ensure_uuid(payload, "action_id")
        if "option_json" in payload and not isinstance(payload["option_json"], str):
            payload["option_json"] = self._serialise_json(payload["option_json"])
        if "targets_json" in payload and not isinstance(payload["targets_json"], str):
            payload["targets_json"] = self._serialise_json(payload["targets_json"])
        return str(self._execute_insert("actions", payload, return_value=action_id))

    def insert_assessment(self, values: Mapping[str, object]) -> str:
        payload = dict(values)
        assessment_id = self._ensure_uuid(payload, "assessment_id")
        if "assessment_json" in payload and not isinstance(payload["assessment_json"], str):
            payload["assessment_json"] = self._serialise_json(payload["assessment_json"])
        return str(self._execute_insert("assessments", payload, return_value=assessment_id))

    def insert_credibility(self, values: Mapping[str, object]) -> str:
        payload = dict(values)
        credibility_id = self._ensure_uuid(payload, "credibility_vector_id")
        if "credibility_json" in payload and not isinstance(payload["credibility_json"], str):
            payload["credibility_json"] = self._serialise_json(payload["credibility_json"])
        return str(self._execute_insert("credibility", payload, return_value=credibility_id))

    def insert_result(self, values: Mapping[str, object]) -> str:
        payload = dict(values)
        if "successful_execution" in payload:
            payload["successful_execution"] = int(bool(payload["successful_execution"]))
        execution_id = payload.get("execution_id")
        if execution_id is None:
            raise ValueError("execution_id is required for results")
        return str(self._execute_insert("results", payload, return_value=str(execution_id)))

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


__all__ = ["DatabaseLockedError", "SQLiteConnector", "sqlite_connector", "sanitize_identifier"]
