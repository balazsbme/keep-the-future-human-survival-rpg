from __future__ import annotations

import json
import sqlite3
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cli_game import load_characters
from evaluations.game_database import GameDatabaseRecorder
from evaluations.player_manager import PlayerManager
from evaluations.players import RandomPlayer
from evaluations.sqlite3_connector import SQLiteConnector
from rpg.assessment_agent import AssessmentAgent


def _response_with_text(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text)


def test_game_execution_persists_uuid_relationships(tmp_path: Path) -> None:
    db_path = tmp_path / "game.sqlite"
    log_dir = tmp_path / "logs"

    action_payload = {
        "actions": [
            {
                "text": "Coordinate a multi-faction response.",
                "type": "action",
                "related-triplet": 1,
                "related-attribute": "policy",
            }
        ]
    }

    mock_char_model = MagicMock()
    mock_char_model.generate_content.return_value = _response_with_text(
        json.dumps(action_payload)
    )

    mock_assess_model = MagicMock()
    mock_assess_model.generate_content.return_value = _response_with_text(
        "50\n50\n50\n50\n50"
    )

    with patch("rpg.character.genai") as mock_char_genai, patch(
        "rpg.assessment_agent.genai"
    ) as mock_assess_genai:
        mock_char_genai.GenerativeModel.return_value = mock_char_model
        mock_assess_genai.GenerativeModel.return_value = mock_assess_model

        characters = load_characters(scenario_name="complete")
        assessor = AssessmentAgent()
        connector = SQLiteConnector(db_path=db_path)

        def observer_factory(*_args) -> GameDatabaseRecorder:
            return GameDatabaseRecorder(connector, notes="integration-test")

        manager = PlayerManager(
            characters,
            assessor,
            log_dir,
            game_observer_factory=observer_factory,
            scenario="complete",
        )
        manager.run_sequence("random", RandomPlayer(), games=1, rounds=1)
        connector.close()

    assert db_path.exists()

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row

    execution_row = connection.execute(
        "SELECT execution_id FROM executions"
    ).fetchone()
    assert execution_row is not None
    execution_id = execution_row["execution_id"]
    uuid.UUID(execution_id)

    action_row = connection.execute(
        "SELECT action_id, execution_id FROM actions"
    ).fetchone()
    assert action_row is not None
    action_id = action_row["action_id"]
    uuid.UUID(action_id)
    assert action_row["execution_id"] == execution_id

    assessment_row = connection.execute(
        "SELECT execution_id, action_id FROM assessments"
    ).fetchone()
    assert assessment_row is not None
    assert assessment_row["execution_id"] == execution_id
    assert assessment_row["action_id"] == action_id

    credibility_row = connection.execute(
        "SELECT execution_id, action_id FROM credibility"
    ).fetchone()
    assert credibility_row is not None
    assert credibility_row["execution_id"] == execution_id
    assert credibility_row["action_id"] == action_id

    result_row = connection.execute(
        "SELECT execution_id FROM results"
    ).fetchone()
    assert result_row is not None
    assert result_row["execution_id"] == execution_id

    connection.close()
