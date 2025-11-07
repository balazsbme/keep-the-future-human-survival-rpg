import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rpg.game_state import GameState
from rpg.character import ResponseOption
from rpg.config import GameConfig


class DummyCharacter:
    def __init__(self) -> None:
        self.name = "Ally"
        self.display_name = "Ally Representative"
        self.faction = "Allies"
        self.progress_key = "Allies"
        self.progress_label = "Allies"
        self.triplets = [("init", "end", "gap")]
        self.weights = [1]

    def attribute_score(self, _attribute: str | None) -> int:
        return 0


@patch("rpg.character.genai")
def test_log_npc_responses_records_single_entry(mock_genai):
    mock_model = MagicMock()
    mock_model.generate_content.return_value = MagicMock(text="[]")
    mock_genai.GenerativeModel.return_value = mock_model

    character = DummyCharacter()
    config = GameConfig(enabled_factions=("Allies", "CivilSociety"))
    state = GameState([character], config_override=config)
    action_option = ResponseOption(
        text="Build community shelters",
        type="action",
        related_triplet=1,
        related_attribute="network",
    )
    chat_option = ResponseOption(text="We can mobilise volunteers.", type="chat")

    entries = state.log_npc_responses(character, [action_option, chat_option])

    assert len(entries) == 1
    history = state.conversation_history(character)
    assert len(history) == 1
    label_map = state.action_label_map(character)
    label = label_map.get(action_option.text)
    assert label is not None
    assert history[0].text == f"{label}: {action_option.text}"
    assert history[0].type == "action"
    stored_actions = state.available_npc_actions(character)
    assert any(option.text == action_option.text for option in stored_actions)


@patch("rpg.character.genai")
def test_log_npc_responses_falls_back_to_chat(mock_genai):
    mock_model = MagicMock()
    mock_model.generate_content.return_value = MagicMock(text="[]")
    mock_genai.GenerativeModel.return_value = mock_model

    character = DummyCharacter()
    config = GameConfig(enabled_factions=("Allies", "CivilSociety"))
    state = GameState([character], config_override=config)
    chat_option = ResponseOption(text="We can mobilise volunteers.", type="chat")

    entries = state.log_npc_responses(character, [chat_option])

    assert len(entries) == 1
    history = state.conversation_history(character)
    assert len(history) == 1
    assert history[0].text == chat_option.text
    assert history[0].type == "chat"
