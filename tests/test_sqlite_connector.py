from __future__ import annotations

import sqlite3
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluations.sqlite3_connector import SQLiteConnector


def _get_columns(connection: sqlite3.Connection, table: str) -> dict[str, str]:
    cursor = connection.execute(f"PRAGMA table_info({table})")
    return {row[1]: row[2] for row in cursor.fetchall()}


def test_dynamic_schema_and_inserts(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite"
    connector = SQLiteConnector(db_path=db_path)
    connector.initialise()

    connector.ensure_dynamic_schema({"Governments": 2, "CivilSociety": 1}, ["Governments", "CivilSociety"])

    columns = _get_columns(connector.connection, "assessments")
    assert "governments_triplet_1" in columns
    assert "governments_triplet_2" in columns
    assert "civilsociety_triplet_1" in columns
    assert "session_id" in columns

    credibility_columns = _get_columns(connector.connection, "credibility")
    assert "credibility_governments" in credibility_columns
    assert "credibility_civilsociety" in credibility_columns
    assert "session_id" in credibility_columns

    execution_columns = _get_columns(connector.connection, "executions")
    assert "action_time_cost_years" in execution_columns
    assert "format_prompt_character_limit" in execution_columns
    assert "conversation_force_action_after" in execution_columns
    assert "log_filename" in execution_columns
    assert "session_id" in execution_columns

    results_columns = _get_columns(connector.connection, "results")
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
    assert execution_id > 0

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
    assert action_id > 0

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
    assert assessment_id > 0

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
    assert credibility_id > 0

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
    row = connector.connection.execute(
        "SELECT result, successful_execution, log_warning_count, log_error_count, session_id FROM results WHERE execution_id = ?",
        (execution_id,),
    ).fetchone()
    assert row["result"] == "Win"
    assert row["successful_execution"] == 1
    assert row["log_warning_count"] == 2
    assert row["log_error_count"] == 1
    assert row["session_id"] == "abc"

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
    row = connector.connection.execute("SELECT COUNT(*) FROM executions").fetchone()
    assert row[0] == 5
