"""Session activity tracking for backup scheduling."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class SessionActivitySnapshot:
    """Summary of session activity for backup triggers."""

    active_session_count: int
    closed_since_last_backup: int


@dataclass
class _SessionRecord:
    session_id: str
    last_access: float
    created_generation: int
    closed: bool = False


class SessionActivityMonitor:
    """Track active sessions and closed session counts for scheduling."""

    def __init__(self) -> None:
        self._sessions: dict[str, _SessionRecord] = {}
        self._closed_since_last_backup = 0
        self._generation = 0
        self._lock = threading.Lock()

    def register_session(self, session_id: str, now: float | None = None) -> None:
        """Register a new session for activity tracking."""

        timestamp = now if now is not None else time.time()
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].last_access = timestamp
                return
            self._sessions[session_id] = _SessionRecord(
                session_id=session_id,
                last_access=timestamp,
                created_generation=self._generation,
            )

    def touch_session(self, session_id: str, now: float | None = None) -> None:
        """Update session activity based on a new request."""

        timestamp = now if now is not None else time.time()
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                self._sessions[session_id] = _SessionRecord(
                    session_id=session_id,
                    last_access=timestamp,
                    created_generation=self._generation,
                )
                return
            record.last_access = timestamp
            if record.closed:
                record.closed = False
                record.created_generation = self._generation

    def close_inactive_sessions(
        self, inactive_for_seconds: float, now: float | None = None
    ) -> list[str]:
        """Mark inactive sessions as closed for backup counting."""

        timestamp = now if now is not None else time.time()
        closed_sessions: list[str] = []
        with self._lock:
            for record in self._sessions.values():
                if record.closed:
                    continue
                if timestamp - record.last_access < inactive_for_seconds:
                    continue
                record.closed = True
                closed_sessions.append(record.session_id)
                if record.created_generation == self._generation:
                    self._closed_since_last_backup += 1
        return closed_sessions

    def mark_closed(self, session_id: str) -> None:
        """Explicitly mark a session as closed."""

        with self._lock:
            record = self._sessions.get(session_id)
            if record is None or record.closed:
                return
            record.closed = True
            if record.created_generation == self._generation:
                self._closed_since_last_backup += 1

    def snapshot(self) -> SessionActivitySnapshot:
        """Return the current snapshot of session activity."""

        with self._lock:
            active = sum(1 for record in self._sessions.values() if not record.closed)
            closed = self._closed_since_last_backup
        return SessionActivitySnapshot(
            active_session_count=active,
            closed_since_last_backup=closed,
        )

    def reset_for_backup(self) -> None:
        """Reset counters after a backup completes."""

        with self._lock:
            self._generation += 1
            self._closed_since_last_backup = 0
