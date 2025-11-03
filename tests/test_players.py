# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluations.players import (
    ActionFirstRandomPlayer,
    GeminiCivilSocietyPlayer,
    GeminiCorporationPlayer,
    RandomPlayer,
)
from rpg.assessment_agent import AssessmentAgent
from rpg.game_state import GameState
from rpg.character import ResponseOption, YamlCharacter

CHARACTERS_FILE = os.path.join(
    os.path.dirname(__file__), "fixtures", "characters.yaml"
)
SCENARIO_FILE = os.path.join(
    os.path.dirname(__file__), "fixtures", "scenarios", "complete.yaml"
)

load_dotenv()


def _load_test_character() -> YamlCharacter:
    with open(CHARACTERS_FILE, "r", encoding="utf-8") as fh:
        character_payload = yaml.safe_load(fh)
    with open(SCENARIO_FILE, "r", encoding="utf-8") as fh:
        faction_payload = yaml.safe_load(fh)
    profile = character_payload["Characters"][0]
    faction_spec = faction_payload[profile["faction"]]
    return YamlCharacter(profile["name"], faction_spec, profile)


class PlayerTests(unittest.TestCase):
    @patch("evaluations.players.random.choice")
    @patch("rpg.assessment_agent.genai")
    @patch("rpg.character.genai")
    @patch("rpg.game_state.random.randint", return_value=20)
    def test_random_player_turn(
        self, mock_uniform, mock_char_genai, mock_assess_genai, mock_choice
    ):
        mock_action_model = MagicMock()
        mock_assess_model = MagicMock()
        mock_action_model.generate_content.return_value = MagicMock(
            text=json.dumps(
                [
                    {
                        "text": "A",
                        "type": "action",
                        "related-triplet": 1,
                        "related-attribute": "leadership",
                    },
                    {
                        "text": "Ask status",
                        "type": "chat",
                        "related-triplet": "None",
                        "related-attribute": "None",
                    },
                    {
                        "text": "Offer help",
                        "type": "chat",
                        "related-triplet": "None",
                        "related-attribute": "None",
                    },
                ]
            )
        )
        mock_assess_model.generate_content.return_value = MagicMock(
            text="10\n20\n30"
        )
        mock_char_genai.GenerativeModel.return_value = mock_action_model
        mock_assess_genai.GenerativeModel.return_value = mock_assess_model
        char = _load_test_character()
        state = GameState([char])
        assessor = AssessmentAgent()
        def choice_side_effect(options):
            if options and isinstance(options[0], YamlCharacter):
                return char
            for option in options:
                if getattr(option, "is_action", False):
                    return option
            return options[0]

        mock_choice.side_effect = choice_side_effect
        player = RandomPlayer()
        player.take_turn(state, assessor)
        self.assertEqual(
            state.history[0],
            (char.display_name, "A"),
        )
        self.assertEqual(state.progress[char.progress_key], [10, 20, 30])
        self.assertIsNotNone(state.last_action_attempt)
        self.assertEqual(state.last_action_attempt.attribute, "leadership")

    @patch("evaluations.players.random.choice")
    @patch("rpg.assessment_agent.genai")
    @patch("rpg.character.genai")
    @patch("rpg.game_state.random.randint", side_effect=[1, 20])
    def test_action_first_player_rerolls_until_success(
        self, mock_uniform, mock_char_genai, mock_assess_genai, mock_choice
    ):
        mock_action_model = MagicMock()
        mock_assess_model = MagicMock()
        mock_action_model.generate_content.return_value = MagicMock(
            text=json.dumps(
                [
                    {
                        "text": "A",
                        "type": "action",
                        "related-triplet": 1,
                        "related-attribute": "leadership",
                    },
                    {
                        "text": "Discuss",
                        "type": "chat",
                        "related-triplet": "None",
                        "related-attribute": "None",
                    },
                ]
            )
        )
        mock_assess_model.generate_content.return_value = MagicMock(
            text="10\n20\n30"
        )
        mock_char_genai.GenerativeModel.return_value = mock_action_model
        mock_assess_genai.GenerativeModel.return_value = mock_assess_model
        char = _load_test_character()
        state = GameState([char])
        assessor = AssessmentAgent()

        def choice_side_effect(options):
            if options and isinstance(options[0], YamlCharacter):
                return char
            for option in options:
                if getattr(option, "is_action", False):
                    return option
            return options[0]

        mock_choice.side_effect = choice_side_effect
        player = ActionFirstRandomPlayer()
        player.take_turn(state, assessor)

        self.assertEqual(state.history[0], (char.display_name, "A"))
        self.assertEqual(state.progress[char.progress_key], [10, 20, 30])
        self.assertIsNotNone(state.last_action_attempt)
        self.assertTrue(state.last_action_attempt.success)
        self.assertEqual(state.last_reroll_count, 1)
        self.assertEqual(mock_uniform.call_count, 2)

    @patch("evaluations.players.genai")
    @patch("rpg.character.genai")
    def test_gemini_win_prompt(self, mock_char_genai, mock_players_genai):
        if not os.environ.get("GEMINI_API_KEY"):
            self.skipTest("GEMINI_API_KEY environment variable not set")
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text="1")
        mock_players_genai.GenerativeModel.return_value = mock_model
        mock_char_genai.GenerativeModel.return_value = MagicMock()
        char = _load_test_character()
        state = GameState([char])
        player = GeminiCivilSocietyPlayer()
        actions = [
            ResponseOption(text="A", type="action"),
            ResponseOption(text="B", type="action"),
            ResponseOption(text="C", type="action"),
        ]
        player.select_action(char, [], actions, state)
        prompt = mock_model.generate_content.call_args[0][0]
        self.assertIn(state.how_to_win.split()[0], prompt)
        self.assertIn(char.display_name, prompt)

    @patch("evaluations.players.genai")
    @patch("rpg.character.genai")
    def test_corporation_context(self, mock_char_genai, mock_players_genai):
        if not os.environ.get("GEMINI_API_KEY"):
            self.skipTest("GEMINI_API_KEY environment variable not set")
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text="1")
        mock_players_genai.GenerativeModel.return_value = mock_model
        mock_char_genai.GenerativeModel.return_value = MagicMock()
        char = _load_test_character()
        gov_ctx = corp_ctx = "CTX"
        player = GeminiCorporationPlayer(corp_ctx)
        state = GameState([char])
        actions = [
            ResponseOption(text="A", type="action"),
            ResponseOption(text="B", type="action"),
            ResponseOption(text="C", type="action"),
        ]
        player.select_action(char, [], actions, state)
        prompt = mock_model.generate_content.call_args[0][0]
        self.assertIn("CTX", prompt)


if __name__ == "__main__":
    unittest.main()
