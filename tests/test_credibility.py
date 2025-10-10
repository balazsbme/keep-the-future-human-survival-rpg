import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rpg.character import ActionOption
from rpg.game_state import PLAYER_FACTION, GameState


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

    def generate_actions(self, history):  # pragma: no cover - not used
        return []

    def perform_action(self, action, history):  # pragma: no cover - not used
        return [0]


class CredibilityMatrixTests(unittest.TestCase):
    @patch("rpg.game_state.random.uniform", return_value=0)
    def test_record_action_rewards_targets(self, mock_uniform):
        character = DummyCharacter("Alice", "Governments")
        state = GameState([character])
        initial_player = state.credibility.value(PLAYER_FACTION, "Regulators")
        initial_actor = state.credibility.value("Governments", "Regulators")
        action = ActionOption(text="Coordinate with regulators", related_triplet=None)
        state.record_action(character, action, targets=["Regulators"])
        updated_player = state.credibility.value(PLAYER_FACTION, "Regulators")
        updated_actor = state.credibility.value("Governments", "Regulators")
        self.assertEqual(updated_player, min(100, initial_player + 20))
        self.assertEqual(updated_actor, initial_actor)

    @patch("rpg.game_state.random.uniform", return_value=0)
    def test_record_action_penalises_targets_with_triplet(self, mock_uniform):
        character = DummyCharacter("Alice", "Governments")
        state = GameState([character])
        initial_player = state.credibility.value(PLAYER_FACTION, "Regulators")
        initial_actor = state.credibility.value("Governments", "Regulators")
        action = ActionOption(text="Enforce compute caps", related_triplet=1)
        state.record_action(character, action, targets=["Regulators"])
        updated_player = state.credibility.value(PLAYER_FACTION, "Regulators")
        updated_actor = state.credibility.value("Governments", "Regulators")
        self.assertEqual(updated_player, max(0, initial_player - 20))
        self.assertEqual(updated_actor, initial_actor)

    @patch("rpg.game_state.random.uniform", return_value=0)
    def test_record_action_without_targets_applies_penalty(self, mock_uniform):
        character = DummyCharacter("Alice", "Governments")
        state = GameState([character])
        initial_player = state.credibility.value(PLAYER_FACTION, "Corporations")
        initial_actor = state.credibility.value("Governments", "Corporations")
        action = ActionOption(text="Limit compute", related_triplet=1)
        state.record_action(character, action)
        updated_player = state.credibility.value(PLAYER_FACTION, "Corporations")
        updated_actor = state.credibility.value("Governments", "Corporations")
        self.assertEqual(updated_player, max(0, initial_player - 20))
        self.assertEqual(updated_actor, initial_actor)

    @patch("rpg.game_state.random.uniform", return_value=0)
    def test_unknown_faction_initialises_defaults(self, mock_uniform):
        outsider = DummyCharacter("Bob", "NewFaction")
        state = GameState([outsider])
        action = ActionOption(text="Build bridges", related_triplet=None)
        state.record_action(outsider, action, targets=["Governments"])
        self.assertEqual(state.credibility.value("NewFaction", "Governments"), 70)
        self.assertEqual(state.credibility.value("Governments", "NewFaction"), 50)


if __name__ == "__main__":
    unittest.main()
