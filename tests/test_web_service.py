# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from web_service import create_app
from rpg.character import MarkdownCharacter

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "test_character.md")


class WebServiceTest(unittest.TestCase):
    def test_html_flow(self):
        with patch("rpg.character.genai") as mock_genai:
            mock_model = MagicMock()
            mock_model.generate_content.side_effect = [
                MagicMock(text="Base context"),
                MagicMock(text="1. Guard the gate\n2. Patrol\n3. Rest"),
                MagicMock(text="I stand guard."),
            ]
            mock_genai.GenerativeModel.return_value = mock_model

            character = MarkdownCharacter("Tester", FIXTURE)

        with patch("web_service.load_characters", return_value=[character]), \
            patch("web_service.GameState") as MockState:
            mock_state = MockState.return_value
            mock_state.characters = [character]
            app = create_app()
            client = app.test_client()

            resp = client.get("/")
            self.assertEqual(resp.status_code, 200)
            self.assertIn("Tester", resp.data.decode())

            resp = client.post("/actions", data={"character": "0"})
            self.assertEqual(resp.status_code, 200)
            page = resp.data.decode()
            self.assertIn("Guard the gate", page)
            self.assertIn("Back to characters", page)

            resp = client.post(
                "/perform", data={"character": "0", "action": "Guard the gate"}
            )
            self.assertEqual(resp.status_code, 200)
            self.assertIn("I stand guard.", resp.data.decode())
            mock_state.record_action.assert_called_once_with(character, "Guard the gate")


if __name__ == "__main__":
    unittest.main()
