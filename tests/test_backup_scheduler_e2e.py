# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os
import sys
import time
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

with patch.dict("sys.modules", {"google.generativeai": MagicMock()}):
    from web_service import create_app

from evaluations.backup_scheduler import BackupSchedulerConfig
from rpg.character import ResponseOption, YamlCharacter
from rpg.config import GameConfig

CHARACTERS_FILE = os.path.join(
    os.path.dirname(__file__), "fixtures", "characters.yaml"
)
SCENARIO_FILE = os.path.join(
    os.path.dirname(__file__), "fixtures", "scenarios", "complete.yaml"
)


def _load_test_character() -> YamlCharacter:
    with open(CHARACTERS_FILE, "r", encoding="utf-8") as fh:
        character_payload = yaml.safe_load(fh)
    with open(SCENARIO_FILE, "r", encoding="utf-8") as fh:
        faction_payload = yaml.safe_load(fh)
    profile = character_payload["Characters"][0]
    faction_spec = faction_payload[profile["faction"]]
    return YamlCharacter(profile["name"], faction_spec, profile)


def _post_action(client) -> None:
    option = ResponseOption(
        text="Coordinate oversight teams",
        type="action",
        related_triplet=1,
        related_attribute="leadership",
    )
    payload = json.dumps(option.to_payload())
    response = client.post(
        "/actions",
        data={"character": "0", "response": payload},
    )
    assert response.status_code == 200


def test_backup_scheduler_e2e(tmp_path, monkeypatch):
    db_path = tmp_path / "game.sqlite"
    backup_path = tmp_path / "backups"
    monkeypatch.setenv("ENABLE_SQLITE_LOGGING", "1")
    monkeypatch.setenv("LOG_WEB_RUNS_TO_DB", "1")
    monkeypatch.setenv("EVALUATION_SQLITE_PATH", str(db_path))
    monkeypatch.setenv("EVALUATION_SQLITE_BACKUP_PATH", str(backup_path))
    sqlite3.connect(db_path).close()
    test_config = GameConfig(
        enabled_factions=("test_character", "CivilSociety", "ScientificCommunity")
    )
    backup_config = BackupSchedulerConfig(
        enabled=True,
        poll_interval_seconds=0.01,
        session_inactive_seconds=0.01,
        closed_sessions_threshold=2,
    )
    character = _load_test_character()

    with patch("rpg.character.genai"), patch(
        "rpg.assessment_agent.genai"
    ), patch(
        "web_service.load_characters", return_value=[character]
    ), patch(
        "web_service.current_config", test_config
    ), patch(
        "rpg.character.Character._generate_with_context",
        return_value=SimpleNamespace(text="[]"),
    ), patch(
        "rpg.assessment_agent.AssessmentAgent.assess", return_value={}
    ), patch(
        "rpg.game_state.random.randint", return_value=20
    ), patch(
        "web_service.time.sleep", return_value=None
    ), patch(
        "web_service.load_backup_scheduler_config", return_value=backup_config
    ), patch(
        "web_service.WEB_LOG_TO_DB", True
    ):
        app = create_app()
        client_one = app.test_client()
        client_two = app.test_client()

        client_one.get("/start")
        _post_action(client_one)

        client_two.get("/start")
        _post_action(client_two)

        scheduler = app.config["backup_scheduler"]
        monitor = app.config["session_activity_monitor"]
        assert scheduler is not None
        assert monitor is not None
        time.sleep(0.05)
        monitor.close_inactive_sessions(
            backup_config.session_inactive_seconds,
            now=time.time() + 1,
        )
        assert scheduler.run_once() is True

    backup_files = list(backup_path.glob("game-*.sqlite"))
    assert len(backup_files) == 1
