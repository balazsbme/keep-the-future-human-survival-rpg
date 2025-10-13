import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rpg.character import ResponseOption
from rpg.game_state import GameState


class DummyCharacter:
    def __init__(self, name: str, faction: str) -> None:
        self.name = name
        self.faction = faction
        self.progress_key = faction
        self.progress_label = faction
        self.display_name = name
        self.triplets = [("initial", "end", {"severity": "Small"})]
        self.weights = [1]

    def attribute_score(self, attribute):
        return 10

    def perform_action(self, action, history):  # pragma: no cover - not used
        return [0]


class CredibilityMatrixTests(unittest.TestCase):
    @patch("rpg.character.genai")
    @patch("rpg.game_state.random.uniform", return_value=0)
    def test_record_action_rewards_targets(self, mock_uniform, mock_genai):
        mock_genai.GenerativeModel.return_value = MagicMock()
        character = DummyCharacter("Alice", "Governments")
        state = GameState([character])
        player_faction = state.player_faction
        initial_player = state.credibility.value(player_faction, "Regulators")
        action = ResponseOption(
            text="Coordinate with regulators", type="action", related_triplet=None
        )
        state.record_action(character, action, targets=["Regulators"])
        updated_player = state.credibility.value(player_faction, "Regulators")
        self.assertEqual(updated_player, min(100, initial_player + 10))

    @patch("rpg.character.genai")
    @patch("rpg.game_state.random.uniform", return_value=0)
    def test_record_action_penalises_targets_with_triplet(
        self, mock_uniform, mock_genai
    ):
        mock_genai.GenerativeModel.return_value = MagicMock()
        character = DummyCharacter("Alice", "Governments")
        state = GameState([character])
        player_faction = state.player_faction
        initial_player = state.credibility.value(player_faction, "Regulators")
        action = ResponseOption(
            text="Enforce compute caps", type="action", related_triplet=1
        )
        state.record_action(character, action, targets=["Regulators"])
        updated_player = state.credibility.value(player_faction, "Regulators")
        self.assertEqual(updated_player, max(0, initial_player - 30))

    @patch("rpg.character.genai")
    @patch("rpg.game_state.random.uniform", return_value=0)
    def test_record_action_without_targets_applies_penalty(
        self, mock_uniform, mock_genai
    ):
        mock_genai.GenerativeModel.return_value = MagicMock()
        character = DummyCharacter("Alice", "Governments")
        state = GameState([character])
        player_faction = state.player_faction
        initial_player = state.credibility.value(player_faction, "Governments")
        action = ResponseOption(text="Limit compute", type="action", related_triplet=1)
        state.record_action(character, action)
        updated_player = state.credibility.value(player_faction, "Governments")
        self.assertEqual(updated_player, max(0, initial_player - 30))

    @patch("rpg.character.genai")
    @patch("rpg.game_state.random.uniform", return_value=0)
    def test_unknown_faction_initialises_defaults(self, mock_uniform, mock_genai):
        mock_genai.GenerativeModel.return_value = MagicMock()
        outsider = DummyCharacter("Bob", "NewFaction")
        state = GameState([outsider])
        action = ResponseOption(text="Build bridges", type="action", related_triplet=None)
        state.record_action(outsider, action, targets=["Governments"])
        player_faction = state.player_faction
        self.assertEqual(state.credibility.value(player_faction, "Governments"), 60)
        self.assertEqual(state.credibility.value("Governments", "NewFaction"), 50)


if __name__ == "__main__":
    unittest.main()
