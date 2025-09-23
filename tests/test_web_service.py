# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from web_service import create_app
from rpg.character import YamlCharacter

FIXTURE_FILE = os.path.join(os.path.dirname(__file__), "fixtures", "characters.yaml")


class WebServiceTest(unittest.TestCase):
    def test_win_and_reset_flow(self):
        with patch("rpg.character.genai") as mock_char_genai, patch(
            "rpg.assessment_agent.genai"
        ) as mock_assess_genai:
            mock_action_model = MagicMock()
            mock_assess_model = MagicMock()
            mock_action_model.generate_content.return_value = MagicMock(
                text=json.dumps(
                    [
                        {"text": "A", "related-triplet": 1},
                        {"text": "B", "related-triplet": "None"},
                        {"text": "C", "related-triplet": "None"},
                    ]
                )
            )
            mock_assess_model.generate_content.return_value = MagicMock(
                text="90\n90\n90"
            )
            mock_char_genai.GenerativeModel.return_value = mock_action_model
            mock_assess_genai.GenerativeModel.return_value = mock_assess_model
            with open(FIXTURE_FILE, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            character = YamlCharacter("test_character", data["test_character"])
            with patch("web_service.load_characters", return_value=[character]):
                app = create_app()
                client = app.test_client()

        resp = client.get("/")
        page = resp.data.decode()
        self.assertEqual(resp.status_code, 200)
        self.assertIn("AI Safety Negotiation Game", page)
        self.assertIn("Start", page)
        self.assertIn("Instructions", page)
        self.assertIn("GitHub", page)

        start_resp = client.get("/start")
        start_page = start_resp.data.decode()
        self.assertEqual(start_resp.status_code, 200)
        self.assertIn("Keep the Future Human Survival RPG", start_page)
        self.assertIn("Reset", start_page)
        self.assertIn("Instructions", start_page)
        self.assertIn("GitHub", start_page)

        actions_resp = client.post("/actions", data={"character": "0"})
        actions_page = actions_resp.data.decode()
        self.assertEqual(actions_resp.status_code, 200)
        self.assertIn("<h1>test_character</h1>", actions_page)
        self.assertIn(
            "Which action do you want test_character to perform?", actions_page
        )

        inst_resp = client.get("/instructions")
        inst_page = inst_resp.data.decode()
        self.assertEqual(inst_resp.status_code, 200)
        self.assertIn("Instructions", inst_page)
        self.assertIn("GitHub", inst_page)

        resp = client.post(
            "/perform", data={"character": "0", "action": "A"}, follow_redirects=True
        )
        page = resp.data.decode()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.request.path, "/result")
        self.assertIn("You won!", page)
        self.assertIn("Action History", page)
        self.assertIn("<li><strong>test_character</strong>: A</li>", page)
        self.assertIn("Final weighted score", page)
        self.assertIn("Reset", page)
        self.assertIn("GitHub", page)

        resp = client.post("/reset", follow_redirects=True)
        page = resp.data.decode()
        self.assertEqual(resp.request.path, "/start")
        self.assertIn("Final weighted score: 0", page)
        self.assertNotIn("Action History", page)
        self.assertIn("GitHub", page)

    def test_loss_after_ten_actions(self):
        with patch("rpg.character.genai") as mock_char_genai, patch(
            "rpg.assessment_agent.genai"
        ) as mock_assess_genai:
            mock_action_model = MagicMock()
            mock_assess_model = MagicMock()
            mock_action_model.generate_content.return_value = MagicMock(
                text=json.dumps(
                    [
                        {"text": "A", "related-triplet": 1},
                        {"text": "B", "related-triplet": "None"},
                        {"text": "C", "related-triplet": "None"},
                    ]
                )
            )
            mock_assess_model.generate_content.return_value = MagicMock(
                text="10\n20\n30"
            )
            mock_char_genai.GenerativeModel.return_value = mock_action_model
            mock_assess_genai.GenerativeModel.return_value = mock_assess_model
            with open(FIXTURE_FILE, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            character = YamlCharacter("test_character", data["test_character"])
            with patch("web_service.load_characters", return_value=[character]):
                app = create_app()
                client = app.test_client()

        for _ in range(9):
            resp = client.post(
                "/perform", data={"character": "0", "action": "A"}, follow_redirects=True
            )
            self.assertEqual(resp.request.path, "/start")

        resp = client.post(
            "/perform", data={"character": "0", "action": "A"}, follow_redirects=True
        )
        page = resp.data.decode()
        self.assertEqual(resp.request.path, "/result")
        self.assertIn("You lost!", page)
        self.assertIn("Action History", page)
        self.assertIn("Final weighted score", page)
        self.assertIn("Reset", page)
        self.assertIn("GitHub", page)


if __name__ == "__main__":
    unittest.main()
