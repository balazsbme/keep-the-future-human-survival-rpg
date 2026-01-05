from __future__ import annotations

import sqlite3
import uuid
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluations.sqlite3_connector import DatabaseLockedError, SQLiteConnector


def _open_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _get_columns(connection: sqlite3.Connection, table: str) -> dict[str, str]:
    cursor = connection.execute(f"PRAGMA table_info({table})")
    return {row[1]: row[2] for row in cursor.fetchall()}


def test_dynamic_schema_and_inserts(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite"
    connector = SQLiteConnector(db_path=db_path)
    connector.initialise()

    connector.ensure_dynamic_schema({"Governments": 2, "CivilSociety": 1}, ["Governments", "CivilSociety"])

    with _open_connection(db_path) as connection:
        columns = _get_columns(connection, "assessments")
    assert "governments_triplet_1" in columns
    assert "governments_triplet_2" in columns
    assert "civilsociety_triplet_1" in columns
    assert "session_id" in columns

    with _open_connection(db_path) as connection:
        credibility_columns = _get_columns(connection, "credibility")
    assert "credibility_governments" in credibility_columns
    assert "credibility_civilsociety" in credibility_columns
    assert "session_id" in credibility_columns

    with _open_connection(db_path) as connection:
        execution_columns = _get_columns(connection, "executions")
    assert "action_time_cost_years" in execution_columns
    assert "format_prompt_character_limit" in execution_columns
    assert "conversation_force_action_after" in execution_columns
    assert "log_filename" in execution_columns
    assert "session_id" in execution_columns

    with _open_connection(db_path) as connection:
        results_columns = _get_columns(connection, "results")
    assert "log_warning_count" in results_columns
    assert "log_error_count" in results_columns
    assert "session_id" in results_columns

    execution_id = connector.insert_execution(
        {
            "session_id": "abc",
            "player_class": "TestPlayer",
            "automated_player_class": "Auto",
            "scenario": "complete",
            "win_threshold": 10,
            "max_rounds": 5,
            "roll_success_threshold": 10,
            "action_time_cost_years": 0.5,
            "format_prompt_character_limit": 400,
            "conversation_force_action_after": 8,
        }
    )
    uuid.UUID(execution_id)

    action_id = connector.insert_action(
        {
            "execution_id": execution_id,
            "session_id": "abc",
            "actor": "NPC",
            "title": "Action",
            "option_text": "Do something",
            "option_type": "action",
            "success": 1,
            "round_number": 1,
            "option_json": {"text": "Do something"},
        }
    )

    with pytest.raises(sqlite3.IntegrityError):
        connector.insert_action(
            {
                "execution_id": execution_id,
                "conversation_id": str(uuid.uuid4()),
                "session_id": "abc",
                "actor": "NPC",
                "title": "Action",
                "option_text": "Do something",
                "option_type": "action",
                "success": 1,
                "round_number": 1,
                "option_json": {"text": "Do something"},
            }
        )
    uuid.UUID(action_id)

    with _open_connection(db_path) as connection:
        action_columns = _get_columns(connection, "actions")
    assert "conversation_id" in action_columns

    assessment_id = connector.insert_assessment(
        {
            "execution_id": execution_id,
            "action_id": action_id,
            "session_id": "abc",
            "scenario": "complete",
            "final_weighted_score": 42,
            "assessment_json": {"after": {"Governments": {"1": 50}}},
            "governments_triplet_1": 50,
            "governments_triplet_2": 20,
            "civilsociety_triplet_1": 30,
        }
    )
    uuid.UUID(assessment_id)

    credibility_id = connector.insert_credibility(
        {
            "execution_id": execution_id,
            "action_id": action_id,
            "session_id": "abc",
            "cost": 3,
            "reroll_attempt_count": 0,
            "credibility_json": {"CivilSociety": 100},
            "credibility_civilsociety": 100,
        }
    )
    uuid.UUID(credibility_id)

    conversation_id = connector.insert_conversation(
        {
            "execution_id": execution_id,
            "session_id": "abc",
            "player_character": "Player",
            "npc_character": "NPC",
            "metadata_json": {"scenario": "complete"},
        }
    )
    uuid.UUID(conversation_id)

    choice_id = connector.insert_player_conversation_choice(
        {
            "conversation_id": conversation_id,
            "order_index": 0,
            "generated_options_json": [{"text": "Hello"}],
            "selected_option_json": {"text": "Hello"},
        }
    )
    uuid.UUID(choice_id)

    npc_response_id = connector.insert_npc_response(
        {
            "conversation_id": conversation_id,
            "execution_id": execution_id,
            "session_id": "abc",
            "npc_character": "NPC",
            "response_json": [{"text": "Hi"}],
            "response_payload_json": {"text": "Hi", "type": "chat"},
            "response_text": "Hi",
            "response_type": "chat",
            "related_triplet": None,
            "related_attribute": "None",
            "order_index": 1,
        }
    )
    uuid.UUID(npc_response_id)

    interaction_id = connector.insert_web_interaction(
        {
            "session_id": "abc",
            "uri": "/start",
            "status_code": 200,
        }
    )
    uuid.UUID(interaction_id)

    connector.insert_result(
        {
            "execution_id": execution_id,
            "session_id": "abc",
            "successful_execution": True,
            "result": "Win",
            "log_warning_count": 2,
            "log_error_count": 1,
        }
    )
    with _open_connection(db_path) as connection:
        row = connection.execute(
            "SELECT result, successful_execution, log_warning_count, log_error_count, session_id FROM results WHERE execution_id = ?",
            (execution_id,),
        ).fetchone()
    assert row["result"] == "Win"
    assert row["successful_execution"] == 1
    assert row["log_warning_count"] == 2
    assert row["log_error_count"] == 1
    assert row["session_id"] == "abc"

    with _open_connection(db_path) as connection:
        action_row = connection.execute(
            "SELECT execution_id FROM actions WHERE action_id = ?",
            (action_id,),
        ).fetchone()
    assert action_row["execution_id"] == execution_id

    with _open_connection(db_path) as connection:
        assessment_row = connection.execute(
            "SELECT execution_id, action_id FROM assessments WHERE assessment_id = ?",
            (assessment_id,),
        ).fetchone()
    assert assessment_row["execution_id"] == execution_id
    assert assessment_row["action_id"] == action_id

    with _open_connection(db_path) as connection:
        credibility_row = connection.execute(
            "SELECT execution_id, action_id FROM credibility WHERE credibility_vector_id = ?",
            (credibility_id,),
        ).fetchone()
    assert credibility_row["execution_id"] == execution_id
    assert credibility_row["action_id"] == action_id

    connector.commit()
    connector.close()


def test_concurrent_inserts(tmp_path: Path) -> None:
    db_path = tmp_path / "concurrent.sqlite"
    connector = SQLiteConnector(db_path=db_path)
    connector.initialise()

    errors: list[Exception] = []

    def worker(idx: int) -> None:
        try:
            connector.insert_execution(
                {
                    "player_class": f"Player{idx}",
                    "automated_player_class": "Auto",
                    "scenario": "complete",
                    "win_threshold": 10,
                    "max_rounds": 5,
                    "roll_success_threshold": 10,
                    "notes": f"worker-{idx}",
                }
            )
            connector.commit()
        except Exception as exc:  # pragma: no cover - defensive fallback
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    with _open_connection(db_path) as connection:
        row = connection.execute("SELECT COUNT(*) FROM executions").fetchone()
        assert row[0] == 5


def test_foreign_key_enforcement_with_uuid_ids(tmp_path: Path) -> None:
    db_path = tmp_path / "fk.sqlite"
    connector = SQLiteConnector(db_path=db_path)
    connector.initialise()

    execution_id = connector.insert_execution(
        {
            "session_id": "abc",
            "player_class": "TestPlayer",
            "automated_player_class": "Auto",
            "scenario": "complete",
            "win_threshold": 10,
            "max_rounds": 5,
            "roll_success_threshold": 10,
        }
    )

    with pytest.raises(sqlite3.IntegrityError):
        connector.insert_action(
            {
                "execution_id": str(uuid.uuid4()),
                "session_id": "abc",
                "actor": "NPC",
                "title": "Action",
                "option_text": "Do something",
                "option_type": "action",
                "success": 1,
                "round_number": 1,
                "option_json": {"text": "Do something"},
            }
        )

    action_id = connector.insert_action(
        {
            "execution_id": execution_id,
            "session_id": "abc",
            "actor": "NPC",
            "title": "Action",
            "option_text": "Do something",
            "option_type": "action",
            "success": 1,
            "round_number": 1,
            "option_json": {"text": "Do something"},
        }
    )

    with pytest.raises(sqlite3.IntegrityError):
        connector.insert_assessment(
            {
                "execution_id": execution_id,
                "action_id": str(uuid.uuid4()),
                "session_id": "abc",
                "scenario": "complete",
                "final_weighted_score": 42,
                "assessment_json": {"after": {"Governments": {"1": 50}}},
            }
        )

    connector.insert_assessment(
        {
            "execution_id": execution_id,
            "action_id": action_id,
            "session_id": "abc",
            "scenario": "complete",
            "final_weighted_score": 42,
            "assessment_json": {"after": {"Governments": {"1": 50}}},
        }
    )


def test_interprocess_lock_marks_lock_file_during_cursor(tmp_path: Path) -> None:
    lock_path = tmp_path / "shared.sqlite.lock"
    db_path = tmp_path / "shared.sqlite"

    if SQLiteConnector.__module__ == "evaluations.sqlite3_connector":
        from evaluations import sqlite3_connector as sqlite3_module

        if sqlite3_module.fcntl is None:
            pytest.skip("fcntl is unavailable; interprocess lock semantics cannot be tested")

    connector = SQLiteConnector(db_path=db_path, lock_path=lock_path)
    connector.initialise()

    with connector.cursor() as cur:
        cur.execute("SELECT 1")
        assert lock_path.exists()
        assert lock_path.read_text(encoding="utf-8").strip() == "1"

    assert lock_path.read_text(encoding="utf-8").strip() == "0"
    connector.close()


def test_interprocess_lock_raises_when_held(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock_path = tmp_path / "shared.sqlite.lock"
    db_path = tmp_path / "shared.sqlite"
    connector = SQLiteConnector(db_path=db_path, lock_path=lock_path)

    if SQLiteConnector.__module__ == "evaluations.sqlite3_connector":
        from evaluations import sqlite3_connector as sqlite3_module

        if sqlite3_module.fcntl is None:
            pytest.skip("fcntl is unavailable; interprocess lock semantics cannot be tested")

        def _raise_lock(_handle, _flags):
            raise OSError("locked")

        monkeypatch.setattr(sqlite3_module.fcntl, "flock", _raise_lock)

    with pytest.raises(DatabaseLockedError):
        with connector.cursor() as cur:
            cur.execute("SELECT 1")


def test_cursor_closes_connection_after_use(tmp_path: Path) -> None:
    db_path = tmp_path / "close.sqlite"
    connector = SQLiteConnector(db_path=db_path)
    connector.initialise()

    with connector.cursor() as cur:
        cur.execute("SELECT 1")
        connection = cur.connection

    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
