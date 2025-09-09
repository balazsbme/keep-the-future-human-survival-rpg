# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rpg.character import YamlCharacter

FIXTURE_FILE = os.path.join(os.path.dirname(__file__), "fixtures", "characters.yaml")


class YamlCharacterTest(unittest.TestCase):
    @patch("rpg.character.genai")
    def test_generate_and_answer(self, mock_genai):
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = [
            MagicMock(text="1. Act1\n2. Act2\n3. Act3"),
            MagicMock(text="10\n20\n30"),
        ]
        mock_genai.GenerativeModel.return_value = mock_model

        with open(FIXTURE_FILE, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        char = YamlCharacter("test_character", data["test_character"])
        actions = char.generate_actions([])
        prompt_used = mock_model.generate_content.call_args_list[0][0][0]
        self.assertIn("condition1", prompt_used)
        self.assertIn("aligned with your motivations and capabilities", prompt_used)
        self.assertEqual(len(actions), 3)
        scores = char.perform_action(actions[0], [])
        self.assertEqual(scores, [10, 20, 30])


if __name__ == "__main__":
    unittest.main()
