# SPDX-License-Identifier: GPL-3.0-or-later

import json
from html import escape
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from web_service import create_app
from rpg.character import YamlCharacter

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
    def test_win_and_reset_flow(self, mock_uniform):
        with patch("rpg.character.genai") as mock_char_genai, patch(
            "rpg.assessment_agent.genai"
        ) as mock_assess_genai:
            mock_action_model = MagicMock()
            mock_assess_model = MagicMock()
            fancy_action = 'Coordinate "<AI>" & <Oversight>'
            mock_action_model.generate_content.return_value = MagicMock(
                text="```json\n"
                + json.dumps(
                    [
                        {
                            "text": fancy_action,
                            "related-triplet": 1,
                            "related-attribute": "leadership",
                        },
                        {
                            "text": "B",
                            "related-triplet": "None",
                            "related-attribute": "technology",
                        },
                        {
                            "text": "C",
                            "related-triplet": "None",
                            "related-attribute": "policy",
                        },
                    ]
                )
                + "\n```"
            )
            mock_assess_model.generate_content.return_value = MagicMock(
                text="90\n90\n90"
            )
            mock_char_genai.GenerativeModel.return_value = mock_action_model
            mock_assess_genai.GenerativeModel.return_value = mock_assess_model
            character = _load_test_character()
            with patch("web_service.load_characters", return_value=[character]):
                app = create_app()
                client = app.test_client()

        resp = client.get("/")
        page = resp.data.decode()
        self.assertEqual(resp.status_code, 200)
        self.assertIn("AI Safety Negotiation Game", page)
        self.assertIn("Start", page)
        self.assertIn("Instructions", page)
        self.assertIn("GitHub", page)

        start_resp = client.get("/start")
        start_page = start_resp.data.decode()
        self.assertEqual(start_resp.status_code, 200)
        self.assertIn("Keep the Future Human Survival RPG", start_page)
        self.assertIn("Reset", start_page)
        self.assertIn("Instructions", start_page)
        self.assertIn("GitHub", start_page)

        actions_resp = client.post("/actions", data={"character": "0"})
        actions_page = actions_resp.data.decode()
        self.assertEqual(actions_resp.status_code, 200)
        self.assertIn(f"<h1>{character.display_name}</h1>", actions_page)
        self.assertIn(
            f"Which action do you want {character.display_name} to perform?",
            actions_page,
        )
        expected_payload = json.dumps(
            {
                "text": fancy_action,
                "related-triplet": 1,
                "related-attribute": "leadership",
            }
        )
        escaped_payload = escape(expected_payload, quote=True)
        self.assertIn(f'value="{escaped_payload}"', actions_page)
        self.assertIn(
            '>Coordinate "&lt;AI&gt;" &amp; &lt;Oversight&gt;</label><br>',
            actions_page,
        )

        inst_resp = client.get("/instructions")
        inst_page = inst_resp.data.decode()
        self.assertEqual(inst_resp.status_code, 200)
        self.assertIn("Instructions", inst_page)
        self.assertIn("GitHub", inst_page)

        resp = client.post(
            "/perform",
            data={"character": "0", "action": expected_payload},
            follow_redirects=True,
        )
        page = resp.data.decode()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.request.path, "/result")
        self.assertIn("You won!", page)
        self.assertIn("Action History", page)
        self.assertIn(
            f"<li><strong>{character.display_name}</strong>: Coordinate \"&lt;AI&gt;\" &amp; &lt;Oversight&gt;</li>",
            page,
        )
        self.assertIn("Final weighted score", page)
        self.assertIn("Reset", page)
        self.assertIn("GitHub", page)

        resp = client.post("/reset", follow_redirects=True)
        page = resp.data.decode()
        self.assertEqual(resp.request.path, "/start")
        self.assertIn("Final weighted score: 0", page)
        self.assertNotIn("Action History", page)
        self.assertIn("GitHub", page)

    @patch("rpg.game_state.random.uniform", return_value=0)
    def test_loss_after_ten_actions(self, mock_uniform):
        with patch("rpg.character.genai") as mock_char_genai, patch(
            "rpg.assessment_agent.genai"
        ) as mock_assess_genai:
            mock_action_model = MagicMock()
            mock_assess_model = MagicMock()
            mock_action_model.generate_content.return_value = MagicMock(
                text="```json\n"
                + json.dumps(
                    [
                        {
                            "text": "A",
                            "related-triplet": 1,
                            "related-attribute": "leadership",
                        },
                        {
                            "text": "B",
                            "related-triplet": "None",
                            "related-attribute": "technology",
                        },
                        {
                            "text": "C",
                            "related-triplet": "None",
                            "related-attribute": "policy",
                        },
                    ]
                )
                + "\n```"
            )
            mock_assess_model.generate_content.return_value = MagicMock(
                text="10\n20\n30"
            )
            mock_char_genai.GenerativeModel.return_value = mock_action_model
            mock_assess_genai.GenerativeModel.return_value = mock_assess_model
            character = _load_test_character()
            with patch("web_service.load_characters", return_value=[character]):
                app = create_app()
                client = app.test_client()

        action_payload = json.dumps(
            {
                "text": "A",
                "related-triplet": 1,
                "related-attribute": "leadership",
            }
        )
        for _ in range(9):
            resp = client.post(
                "/perform",
                data={"character": "0", "action": action_payload},
                follow_redirects=True,
            )
            self.assertEqual(resp.request.path, "/start")

        resp = client.post(
            "/perform",
            data={"character": "0", "action": action_payload},
            follow_redirects=True,
        )
        page = resp.data.decode()
        self.assertEqual(resp.request.path, "/result")
        self.assertIn("You lost!", page)
        self.assertIn("Action History", page)
        self.assertIn("Final weighted score", page)
        self.assertIn("Reset", page)
        self.assertIn("GitHub", page)


if __name__ == "__main__":
    unittest.main()
