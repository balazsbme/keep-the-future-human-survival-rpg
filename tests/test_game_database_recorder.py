from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unittest.mock import MagicMock

from evaluations.game_database import GameDatabaseRecorder
from evaluations.sqlite3_connector import SQLiteConnector
from rpg.character import ResponseOption
from rpg.config import GameConfig
from rpg.game_state import ActionAttempt


class DummyCredibility:
    def __init__(self, mapping: dict[str, dict[str, int]]) -> None:
        self._mapping = mapping

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {source: dict(targets) for source, targets in self._mapping.items()}


class DummyPlayerCharacter:
    faction = "CivilSociety"


def _build_state() -> SimpleNamespace:
    config = GameConfig(scenario="complete", win_threshold=70, max_rounds=8, roll_success_threshold=12)
    progress = {"Governments": [0, 0], "CivilSociety": [0]}
    credibility = DummyCredibility({"CivilSociety": {"Governments": 55, "CivilSociety": 100}})
    state = SimpleNamespace(
        progress=progress,
        credibility=credibility,
        player_faction="CivilSociety",
        config=config,
        player_character=DummyPlayerCharacter(),
        last_action_attempt=None,
        last_action_actor=None,
        last_reroll_count=0,
    )
    state.final_weighted_score = lambda: 75
    return state


def test_recorder_writes_expected_payloads() -> None:
    connector = MagicMock(spec=SQLiteConnector)
    connector.insert_execution.return_value = 3
    connector.insert_action.return_value = 10

    recorder = GameDatabaseRecorder(connector, notes="test-run")
    state = _build_state()

    recorder.on_game_start(
        state,
        player_key="alpha",
        player_class="PlayerChar",
        automated_player_class="AutoPlayer",
        game_index=1,
    )
    connector.ensure_dynamic_schema.assert_called_once()
    connector.insert_execution.assert_called_once()

    recorder.before_turn(state, 1)
    state.progress = {"Governments": [20, 30], "CivilSociety": [40]}
    option = ResponseOption(text="Coordinate", type="action", related_triplet=1, related_attribute="influence")
    attempt = ActionAttempt(
        success=True,
        option=option,
        label="Action 1",
        attribute="influence",
        actor_score=8,
        player_score=10,
        effective_score=10,
        roll=7,
        targets=("Governments",),
        credibility_cost=3,
        credibility_gain=5,
        failure_text=None,
    )
    state.last_action_attempt = attempt
    state.last_action_actor = "NPC"
    state.last_reroll_count = 1

    recorder.after_turn(state, 1)

    connector.insert_action.assert_called_once()
    action_payload = connector.insert_action.call_args.args[0]
    assert action_payload["round_number"] == 1
    assert action_payload["success"] == 1
    assert action_payload["credibility_cost"] == 3
    assert action_payload["option_type"] == "action"

    connector.insert_assessment.assert_called_once()
    assessment_payload = connector.insert_assessment.call_args.args[0]
    assert assessment_payload["final_weighted_score"] == 75
    assert assessment_payload["governments_triplet_1"] == 20
    assert assessment_payload["civilsociety_triplet_1"] == 40
    assert assessment_payload["assessment_json"]["before"] == {"Governments": [0, 0], "CivilSociety": [0]}

    connector.insert_credibility.assert_called_once()
    credibility_payload = connector.insert_credibility.call_args.args[0]
    assert credibility_payload["reroll_attempt_count"] == 1
    assert credibility_payload["credibility_governments"] == 55
    assert credibility_payload["credibility_civilsociety"] == 100

    recorder.on_game_end(state, result="Win", successful=True, error=None)
    connector.insert_result.assert_called_once()
    result_payload = connector.insert_result.call_args.args[0]
    assert result_payload["execution_id"] == 3
    assert result_payload["successful_execution"] is True
    assert result_payload["result"] == "Win"
    assert connector.commit.call_count >= 2


def test_recorder_records_error_outcome() -> None:
    connector = MagicMock(spec=SQLiteConnector)
    connector.insert_execution.return_value = 7

    recorder = GameDatabaseRecorder(connector)
    state = _build_state()

    recorder.on_game_start(
        state,
        player_key="beta",
        player_class="PlayerChar",
        automated_player_class="AutoPlayer",
        game_index=2,
    )

    recorder.on_game_error(state, RuntimeError("boom"))

    connector.insert_result.assert_called_once()
    result_payload = connector.insert_result.call_args.args[0]
    assert result_payload["execution_id"] == 7
    assert result_payload["successful_execution"] is False
    assert result_payload["result"] == "N/A"
    assert "boom" in result_payload["error_info"]
    connector.commit.assert_called()
