# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from players import RandomPlayer, GeminiWinPlayer, GeminiGovCorpPlayer
from rpg.assessment_agent import AssessmentAgent
from rpg.game_state import GameState
from rpg.character import YamlCharacter

FIXTURE_FILE = os.path.join(os.path.dirname(__file__), "fixtures", "characters.yaml")


class PlayerTests(unittest.TestCase):
    @patch("players.random.choice")
    @patch("rpg.assessment_agent.genai")
    @patch("rpg.character.genai")
    def test_random_player_turn(self, mock_char_genai, mock_assess_genai, mock_choice):
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
        char = YamlCharacter("test_character", data["test_character"])
        state = GameState([char])
        assessor = AssessmentAgent()
        mock_choice.side_effect = [char, "A"]
        player = RandomPlayer()
        player.take_turn(state, assessor)
        self.assertEqual(state.history[0], ("test_character", "A"))
        self.assertEqual(state.progress["test_character"], [10, 20, 30])

    @patch("players.genai")
    def test_gemini_win_prompt(self, mock_genai):
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text="1")
        mock_genai.GenerativeModel.return_value = mock_model
        with open(FIXTURE_FILE, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        char = YamlCharacter("test_character", data["test_character"])
        state = GameState([char])
        player = GeminiWinPlayer()
        actions = ["A", "B", "C"]
        player.select_action(char, actions, state)
        prompt = mock_model.generate_content.call_args[0][0]
        self.assertIn(state.how_to_win.split()[0], prompt)
        self.assertIn(char.base_context.split()[0], prompt)

    @patch("players.genai")
    def test_govcorp_context(self, mock_genai):
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text="1")
        mock_genai.GenerativeModel.return_value = mock_model
        with open(FIXTURE_FILE, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        gov_ctx = corp_ctx = "CTX"
        player = GeminiGovCorpPlayer(gov_ctx, corp_ctx)
        char = YamlCharacter("test_character", data["test_character"])
        state = GameState([char])
        actions = ["A", "B", "C"]
        player.select_action(char, actions, state)
        prompt = mock_model.generate_content.call_args[0][0]
        self.assertIn("CTX", prompt)


if __name__ == "__main__":
    unittest.main()
