import os
import threading
import time
from unittest.mock import MagicMock, patch

import yaml

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from web_service import create_app
from rpg.character import YamlCharacter

FIXTURE_FILE = os.path.join(os.path.dirname(__file__), "fixtures", "characters.yaml")


@patch.dict(os.environ, {"ENABLE_PARALLELISM": "1"})
@patch("rpg.assessment_agent.genai")
@patch("rpg.character.genai")
def test_async_action_generation(mock_char_genai, mock_assess_genai):
    mock_char_genai.GenerativeModel.return_value = MagicMock()
    mock_assess_genai.GenerativeModel.return_value = MagicMock()
    with open(FIXTURE_FILE, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    character = YamlCharacter("test_character", data["test_character"])

    start_evt = threading.Event()
    finish_evt = threading.Event()

    def slow_actions(history):
        start_evt.set()
        finish_evt.wait()
        return ["A"]

    with patch.object(character, "generate_actions", side_effect=slow_actions):
        with patch("web_service.load_characters", return_value=[character]):
            app = create_app()
            client = app.test_client()
            client.get("/")
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
def test_assessment_background_wait(mock_char_genai, mock_assess_genai):
    mock_char_genai.GenerativeModel.return_value = MagicMock()
    mock_assess_genai.GenerativeModel.return_value = MagicMock()
    with open(FIXTURE_FILE, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    character = YamlCharacter("test_character", data["test_character"])

    start_evt = threading.Event()
    finish_evt = threading.Event()

    with patch.object(character, "generate_actions", return_value=["A"]):
        class DummyAssess:
            def assess(self, chars, htw, hist, parallel=False):
                start_evt.set()
                finish_evt.wait()
                return {c.name: [100] * len(c.triplets) for c in chars}

        with patch("web_service.AssessmentAgent", return_value=DummyAssess()):
            with patch("web_service.load_characters", return_value=[character]):
                app = create_app()
                client = app.test_client()
                client.get("/")
                resp = client.post(
                    "/perform", data={"character": "0", "action": "A"}
                )
                assert resp.status_code == 302
                assert start_evt.wait(timeout=1)
                resp = client.get("/result")
                assert b"Waiting for assessments" in resp.data
                finish_evt.set()
                time.sleep(0.1)
                resp = client.get("/result")
                assert b"You won" in resp.data
