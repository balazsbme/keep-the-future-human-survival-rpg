from unittest.mock import MagicMock, patch

from rpg.game_state import GameState
from rpg.character import ResponseOption


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
    state = GameState([character])
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
    assert history[0].text == action_option.text
    assert history[0].type == "action"
    stored_actions = state.available_npc_actions(character)
    assert any(option.text == action_option.text for option in stored_actions)


@patch("rpg.character.genai")
def test_log_npc_responses_falls_back_to_chat(mock_genai):
    mock_model = MagicMock()
    mock_model.generate_content.return_value = MagicMock(text="[]")
    mock_genai.GenerativeModel.return_value = mock_model

    character = DummyCharacter()
    state = GameState([character])
    chat_option = ResponseOption(text="We can mobilise volunteers.", type="chat")

    entries = state.log_npc_responses(character, [chat_option])

    assert len(entries) == 1
    history = state.conversation_history(character)
    assert len(history) == 1
    assert history[0].text == chat_option.text
    assert history[0].type == "chat"
