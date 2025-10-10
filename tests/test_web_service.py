# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from web_service import create_app
from rpg.character import ResponseOption, YamlCharacter

CHARACTERS_FILE = os.path.join(
    os.path.dirname(__file__), "fixtures", "characters.yaml"
)
SCENARIO_FILE = os.path.join(
    os.path.dirname(__file__), "fixtures", "scenarios", "complete.yaml"
)


def _load_test_character() -> YamlCharacter:
    with open(CHARACTERS_FILE, "r", encoding="utf-8") as fh:
        character_payload = yaml.safe_load(fh)
    with open(SCENARIO_FILE, "r", encoding="utf-8") as fh:
        faction_payload = yaml.safe_load(fh)
    profile = character_payload["Characters"][0]
    faction_spec = faction_payload[profile["faction"]]
    return YamlCharacter(profile["name"], faction_spec, profile)


class WebServiceTest(unittest.TestCase):
    @patch("rpg.game_state.random.uniform", return_value=0)
    def test_conversation_and_win_flow(self, mock_uniform):
        with patch("rpg.character.genai") as mock_char_genai, patch(
            "rpg.assessment_agent.genai"
        ) as mock_assess_genai:
            npc_model = MagicMock()
            player_model = MagicMock()
            assess_model = MagicMock()
            player_model.generate_content.side_effect = [
                MagicMock(
                    text=json.dumps(
                        [
                            {
                                "text": "What worries you most?",
                                "type": "chat",
                                "related-triplet": "None",
                                "related-attribute": "None",
                            },
                            {
                                "text": "How can I help?",
                                "type": "chat",
                                "related-triplet": "None",
                                "related-attribute": "None",
                            },
                        ]
                    )
                ),
                MagicMock(
                    text=json.dumps(
                        [
                            {
                                "text": "Thanks for the plan.",
                                "type": "chat",
                                "related-triplet": "None",
                                "related-attribute": "None",
                            }
                        ]
                    )
                ),
            ]
            npc_action_text = "Coordinate oversight teams"
            npc_model.generate_content.return_value = MagicMock(
                text=json.dumps(
                    [
                        {
                            "text": "We should gather more intel first.",
                            "type": "chat",
                            "related-triplet": "None",
                            "related-attribute": "None",
                        },
                        {
                            "text": npc_action_text,
                            "type": "action",
                            "related-triplet": 1,
                            "related-attribute": "leadership",
                        },
                    ]
                )
            )
            assess_model.generate_content.return_value = MagicMock(
                text="95\n95\n95"
            )
            mock_char_genai.GenerativeModel.side_effect = [
                npc_model,
                player_model,
                player_model,
                player_model,
            ]
            mock_assess_genai.GenerativeModel.return_value = assess_model
            character = _load_test_character()
            with patch("web_service.load_characters", return_value=[character]):
                app = create_app()
                client = app.test_client()

            resp = client.get("/")
            page = resp.data.decode()
            self.assertEqual(resp.status_code, 200)
            self.assertIn("AI Safety Negotiation Game", page)

            start_resp = client.get("/start")
            start_page = start_resp.data.decode()
            self.assertEqual(start_resp.status_code, 200)
            self.assertIn("Talk", start_page)

            convo_resp = client.get("/actions", query_string={"character": "0"})
            convo_page = convo_resp.data.decode()
            self.assertEqual(convo_resp.status_code, 200)
            self.assertIn(f"<h1>{character.display_name}</h1>", convo_page)
            self.assertIn("No conversation yet", convo_page)
            self.assertIn("What worries you most?", convo_page)

            chat_payload = json.dumps(
                {
                    "text": "What worries you most?",
                    "type": "chat",
                    "related-triplet": "None",
                    "related-attribute": "None",
                }
            )
            follow_resp = client.post(
                "/actions",
                data={"character": "0", "response": chat_payload},
                follow_redirects=True,
            )
            follow_page = follow_resp.data.decode()
            self.assertIn("We should gather more intel first.", follow_page)
            self.assertIn("Coordinate oversight teams", follow_page)
            self.assertIn("<strong>Action:</strong>", follow_page)

            action_option = ResponseOption(
                text=npc_action_text,
                type="action",
                related_triplet=1,
                related_attribute="leadership",
            )
            action_resp = client.post(
                "/actions",
                data={
                    "character": "0",
                    "response": json.dumps(action_option.to_payload()),
                },
                follow_redirects=True,
            )
            action_page = action_resp.data.decode()
            self.assertEqual(action_resp.request.path, "/result")
            self.assertIn("You won!", action_page)
            self.assertIn(npc_action_text, action_page)

            inst_resp = client.get("/instructions")
            inst_page = inst_resp.data.decode()
            self.assertEqual(inst_resp.status_code, 200)
            self.assertIn("Instructions", inst_page)

            reset_resp = client.post("/reset", follow_redirects=True)
            reset_page = reset_resp.data.decode()
            self.assertEqual(reset_resp.request.path, "/start")
            self.assertIn("Final weighted score: 0", reset_page)


if __name__ == "__main__":
    unittest.main()
