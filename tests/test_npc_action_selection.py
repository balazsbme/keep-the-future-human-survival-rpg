from unittest.mock import MagicMock, patch

from rpg.game_state import GameState
from rpg.character import ResponseOption


class DummyCharacter:
    def __init__(self) -> None:
        self.name = "Victor"
        self.display_name = "Victor Representative"
        self.faction = "Corporations"
        self.progress_key = "Corporations"
        self.progress_label = "Corporations"
        self.triplets = [("init", "end", "gap")]
        self.weights = [1]

    def attribute_score(self, _attribute):
        return 0


@patch("rpg.character.genai")
def test_log_npc_responses_only_tracks_displayed_action(mock_genai):
    mock_model = MagicMock()
    mock_model.generate_content.return_value = MagicMock(text="[]")
    mock_genai.GenerativeModel.return_value = mock_model

    character = DummyCharacter()
    state = GameState([character])
    primary_action = ResponseOption(
        text="Deploy joint safeguards",
        type="action",
        related_triplet=1,
        related_attribute="technology",
    )
    secondary_action = ResponseOption(
        text="Launch new public relations campaign",
        type="action",
        related_triplet=1,
        related_attribute="network",
    )

    state.log_npc_responses(character, [primary_action, secondary_action])

    stored_actions = state.available_npc_actions(character)
    assert [option.text for option in stored_actions] == [primary_action.text]
    assert all(option.text != secondary_action.text for option in stored_actions)
