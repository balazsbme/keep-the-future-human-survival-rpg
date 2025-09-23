# SPDX-License-Identifier: GPL-3.0-or-later

import json
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
            text=json.dumps(
                [
                    {"text": "Act1", "related-triplet": 1},
                    {"text": "Act2", "related-triplet": "None"},
                    {"text": "Act3", "related-triplet": "None"},
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
        char = YamlCharacter("test_character", data["test_character"])
        actions = char.generate_actions([])
        prompt_used = mock_action_model.generate_content.call_args_list[0][0][0]
        self.assertIn("end1", prompt_used)
        self.assertIn("size: Small", prompt_used)
        self.assertIn("aligned with your motivations and capabilities", prompt_used)
        self.assertIn("Return the result as a JSON array", prompt_used)
        self.assertEqual(actions, ["Act1", "Act2", "Act3"])
        assessor = AssessmentAgent()
        scores = assessor.assess([char], "baseline", [])[char.name]
        self.assertEqual(scores, [10, 20, 30])

    @patch("rpg.character.genai")
    def test_generate_actions_warning_for_related_triplets(self, mock_char_genai):
        mock_action_model = MagicMock()
        mock_action_model.generate_content.return_value = MagicMock(
            text=json.dumps(
                [
                    {"text": "Act1", "related-triplet": 1},
                    {"text": "Act2", "related-triplet": 2},
                    {"text": "Act3", "related-triplet": "None"},
                ]
            )
        )
        mock_char_genai.GenerativeModel.return_value = mock_action_model

        with open(FIXTURE_FILE, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        char = YamlCharacter("test_character", data["test_character"])

        with self.assertLogs("rpg.character", level="WARNING") as log_ctx:
            actions = char.generate_actions([])

        self.assertEqual(actions, ["Act1", "Act2", "Act3"])
        self.assertTrue(
            any("Expected exactly one action referencing a triplet" in msg for msg in log_ctx.output)
        )


if __name__ == "__main__":
    unittest.main()
