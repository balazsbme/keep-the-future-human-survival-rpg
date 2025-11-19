import json
import os
import threading
import time
from unittest.mock import MagicMock, patch

import yaml

from rpg.character import ResponseOption, YamlCharacter
from rpg.config import GameConfig
from web_service import create_app


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
        scenario_payload = yaml.safe_load(fh)
    profile = character_payload["Characters"][0]
    faction_spec = scenario_payload[profile["faction"]]
    return YamlCharacter(profile["name"], faction_spec, profile)


def _enabled_test_config() -> GameConfig:
    return GameConfig(
        enabled_factions=("test_character", "CivilSociety", "ScientificCommunity")
    )


@patch.dict(os.environ, {"ENABLE_PARALLELISM": "1"})
@patch("rpg.assessment_agent.genai")
@patch("rpg.character.genai")
def test_async_response_generation(mock_char_genai, mock_assess_genai):
    mock_char_genai.GenerativeModel.return_value = MagicMock()
    mock_assess_genai.GenerativeModel.return_value = MagicMock()
    character = _load_test_character()

    start_evt = threading.Event()
    finish_evt = threading.Event()

    def slow_responses(
        self,
        history,
        conversation,
        partner,
        *,
        partner_credibility=None,
        conversation_cache=None,
    ):
        start_evt.set()
        finish_evt.wait()
        return [ResponseOption(text="A", type="action")]

    with patch(
        "rpg.character.PlayerCharacter.generate_responses",
        new=slow_responses,
    ):
        test_config = _enabled_test_config()
        with patch("web_service.load_characters", return_value=[character]), patch(
            "web_service.current_config", test_config
        ):
            app = create_app()
            client = app.test_client()
            client.get("/start")
            assert start_evt.wait(timeout=1)
            resp = client.post("/actions", data={"character": "0"})
            assert b"Loading" in resp.data
            finish_evt.set()
            time.sleep(0.1)
            resp = client.get("/actions", query_string={"character": "0"})
            assert b"A" in resp.data
            resp = client.post("/actions", data={"character": "0"})
            assert b"A" in resp.data


@patch.dict(os.environ, {"ENABLE_PARALLELISM": "1"})
@patch("rpg.assessment_agent.genai")
@patch("rpg.character.genai")
def test_async_npc_responses(mock_char_genai, mock_assess_genai):
    mock_char_genai.GenerativeModel.return_value = MagicMock()
    mock_assess_genai.GenerativeModel.return_value = MagicMock()
    character = _load_test_character()

    start_evt = threading.Event()
    finish_evt = threading.Event()
    completed_evt = threading.Event()

    def player_options(
        self,
        history,
        conversation,
        partner,
        *,
        partner_credibility=None,
        conversation_cache=None,
    ):
        return [
            ResponseOption(text="Starter 1", type="chat"),
            ResponseOption(text="Starter 2", type="chat"),
            ResponseOption(text="Starter 3", type="chat"),
        ]

    observed_conversations: list[str] = []

    def slow_npc(
        self,
        history,
        conversation,
        partner,
        *,
        partner_credibility=None,
        conversation_cache=None,
        force_action=False,
    ):
        start_evt.set()
        finish_evt.wait()
        completed_evt.set()
        if conversation:
            observed_conversations.append(conversation[-1].text)
        else:
            observed_conversations.append("<empty>")
        return [ResponseOption(text="Coordinate defenses", type="action")]

    with patch(
        "rpg.character.PlayerCharacter.generate_responses",
        new=player_options,
    ):
        with patch.object(type(character), "generate_responses", new=slow_npc):
            test_config = _enabled_test_config()
            with patch(
                "web_service.load_characters", return_value=[character]
            ), patch("web_service.current_config", test_config):
                app = create_app()
                client = app.test_client()
                client.get("/start")
                resp = client.get("/actions", query_string={"character": "0"})
                assert resp.status_code == 200
                assert start_evt.wait(timeout=1)
                starter_option = ResponseOption(text="Starter 1", type="chat")
                payload = json.dumps(starter_option.to_payload())
                resp = client.post(
                    "/actions",
                    data={"character": "0", "response": payload},
                )
                assert b"Loading" in resp.data
                finish_evt.set()
                assert completed_evt.wait(timeout=5)
                attempts = 0
                resp = client.get("/actions", query_string={"character": "0"})
                while attempts < 50 and b"Coordinate defenses" not in resp.data:
                    time.sleep(0.1)
                    resp = client.get("/actions", query_string={"character": "0"})
                    attempts += 1
                assert b"Coordinate defenses" in resp.data, observed_conversations
                assert any(
                    text == "Starter 1" for text in observed_conversations
                ), observed_conversations


@patch.dict(os.environ, {"ENABLE_PARALLELISM": "1"})
@patch("rpg.assessment_agent.genai")
@patch("rpg.character.genai")
def test_assessment_background_wait(mock_char_genai, mock_assess_genai):
    mock_char_genai.GenerativeModel.return_value = MagicMock()
    mock_assess_genai.GenerativeModel.return_value = MagicMock()
    character = _load_test_character()

    action_option = ResponseOption(
        text="A",
        type="action",
        related_triplet=1,
        related_attribute="leadership",
    )

    def static_responses(
        self,
        history,
        conversation,
        partner,
        *,
        partner_credibility=None,
        conversation_cache=None,
    ):
        return [action_option]

    start_evt = threading.Event()
    finish_evt = threading.Event()

    class DummyAssess:
        def assess(self, chars, history, parallel=False):
            start_evt.set()
            finish_evt.wait()
            return {c.progress_key: [100] * len(c.triplets) for c in chars}

    with patch(
        "rpg.character.PlayerCharacter.generate_responses",
        new=static_responses,
    ):
        with patch("web_service.AssessmentAgent", return_value=DummyAssess()):
            test_config = _enabled_test_config()
            with patch("web_service.load_characters", return_value=[character]), patch(
                "web_service.current_config", test_config
            ):
                app = create_app()
                client = app.test_client()
                client.get("/start")
                payload = json.dumps(action_option.to_payload())
                with patch("rpg.game_state.random.randint", return_value=20):
                    resp = client.post(
                        "/actions",
                        data={"character": "0", "response": payload},
                    )
                assert resp.status_code == 200
                assert b"Action Outcome" in resp.data
                assert start_evt.wait(timeout=1)
                resp = client.get("/result")
                assert b"Waiting for assessments" in resp.data
                finish_evt.set()
                time.sleep(0.1)
                resp = client.get("/result")
                assert b"You won" in resp.data
