# SPDX-License-Identifier: GPL-3.0-or-later

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
                text="1. A\n2. B\n3. C"
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
        self.assertIn("Reset", page)

        resp = client.post(
            "/perform", data={"character": "0", "action": "A"}, follow_redirects=True
        )
        page = resp.data.decode()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.request.path, "/result")
        self.assertIn("You won!", page)
        self.assertIn("History:", page)
        self.assertIn("test_character: A", page)
        self.assertIn("Final weighted score", page)
        self.assertIn("Reset", page)

        resp = client.post("/reset", follow_redirects=True)
        page = resp.data.decode()
        self.assertEqual(resp.request.path, "/")
        self.assertIn("Final weighted score: 0", page)
        self.assertNotIn("History:", page)

    def test_loss_after_ten_actions(self):
        with patch("rpg.character.genai") as mock_char_genai, patch(
            "rpg.assessment_agent.genai"
        ) as mock_assess_genai:
            mock_action_model = MagicMock()
            mock_assess_model = MagicMock()
            mock_action_model.generate_content.return_value = MagicMock(
                text="1. A\n2. B\n3. C"
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
            self.assertEqual(resp.request.path, "/")

        resp = client.post(
            "/perform", data={"character": "0", "action": "A"}, follow_redirects=True
        )
        page = resp.data.decode()
        self.assertEqual(resp.request.path, "/result")
        self.assertIn("You lost!", page)
        self.assertIn("History:", page)
        self.assertIn("Final weighted score", page)
        self.assertIn("Reset", page)


if __name__ == "__main__":
    unittest.main()
