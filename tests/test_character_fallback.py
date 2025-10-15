from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rpg.character import YamlCharacter


@patch("rpg.character.genai")
def test_generate_responses_strips_code_fence_on_fallback(mock_genai):
    mock_model = MagicMock()
    mock_model.generate_content.return_value = MagicMock(
        text="```json\nNot valid\n```"
    )
    mock_genai.GenerativeModel.return_value = mock_model

    spec = {
        "MarkdownContext": "Faction context",
        "end_states": ["end"],
        "initial_states": ["initial"],
        "gaps": ["gap"],
    }
    profile = {
        "name": "Test NPC",
        "faction": "Testers",
        "perks": "",
        "motivations": "",
        "background": "",
        "weaknesses": "",
    }

    character = YamlCharacter("Test NPC", spec, profile)
    partner = SimpleNamespace(
        display_name="Player", faction="Allies", triplets=character.triplets
    )

    options = character.generate_responses([], [], partner, partner_credibility=50)

    assert [option.text for option in options] == ["Not valid"]
    assert all(option.type == "chat" for option in options)
