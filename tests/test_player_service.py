# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from player_service import create_app
from rpg.character import YamlCharacter

FIXTURE_FILE = os.path.join(os.path.dirname(__file__), "fixtures", "characters.yaml")


class PlayerServiceTest(unittest.TestCase):
    def test_progress_page(self):
        with patch("players.random.choice") as mock_choice, patch(
            "rpg.character.genai"
        ) as mock_char_genai, patch(
            "rpg.assessment_agent.genai"
        ) as mock_assess_genai, patch("players.genai") as mock_players_genai:
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
            mock_players_genai.GenerativeModel.return_value = MagicMock()
            with open(FIXTURE_FILE, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            character = YamlCharacter("test_character", data["test_character"])
            mock_choice.side_effect = [character, "A"]
            with patch("player_service.load_characters", return_value=[character]):
                app = create_app()
                client = app.test_client()
        resp = client.post(
            "/", data={"player": "random", "rounds": "1"}, follow_redirects=True
        )
        page = resp.data.decode()
        self.assertIn("Game Progress", page)
        self.assertIn("test_character", page)
        self.assertIn("Final weighted score", page)


if __name__ == "__main__":
    unittest.main()
