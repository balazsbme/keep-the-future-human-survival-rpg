# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rpg.character import YamlCharacter
from rpg.assessment_agent import AssessmentAgent

FIXTURE_FILE = os.path.join(os.path.dirname(__file__), "fixtures", "characters.yaml")


class YamlCharacterTest(unittest.TestCase):
    @patch("rpg.assessment_agent.genai")
    @patch("rpg.character.genai")
    def test_generate_and_answer(self, mock_char_genai, mock_assess_genai):
        mock_action_model = MagicMock()
        mock_assess_model = MagicMock()
        mock_action_model.generate_content.return_value = MagicMock(
            text="1. Act1\n2. Act2\n3. Act3"
        )
        mock_assess_model.generate_content.return_value = MagicMock(
            text="10\n20\n30"
        )
        mock_char_genai.GenerativeModel.return_value = mock_action_model
        mock_assess_genai.GenerativeModel.return_value = mock_assess_model

        with open(FIXTURE_FILE, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        char = YamlCharacter("test_character", data["test_character"])
        actions = char.generate_actions([])
        prompt_used = mock_action_model.generate_content.call_args_list[0][0][0]
        self.assertIn("end1", prompt_used)
        self.assertIn("size: Small", prompt_used)
        self.assertIn("aligned with your motivations and capabilities", prompt_used)
        self.assertEqual(len(actions), 3)
        assessor = AssessmentAgent()
        scores = assessor.assess([char], "baseline", [])[char.name]
        self.assertEqual(scores, [10, 20, 30])


if __name__ == "__main__":
    unittest.main()
