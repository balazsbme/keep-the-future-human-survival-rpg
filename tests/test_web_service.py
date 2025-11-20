# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from flask import Flask
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

with patch.dict("sys.modules", {"google.generativeai": MagicMock()}):
    from web_service import SESSION_COOKIE_NAME, create_app
from rpg.character import ResponseOption, YamlCharacter
from rpg.config import GameConfig

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
    def test_scenario_dropdown_lists_all_yaml_files(self):
        scenario_dir = Path(__file__).resolve().parent.parent / "scenarios"
        expected_values = {p.stem.lower() for p in scenario_dir.glob("*.yaml")}
        self.assertTrue(expected_values, "Expected scenario YAML fixtures to exist")
        hidden = {"complete"}
        selectable = expected_values - hidden
        test_config = GameConfig(
            enabled_factions=("test_character", "CivilSociety", "ScientificCommunity")
        )
        with patch("rpg.character.genai"), patch("rpg.assessment_agent.genai"):
            character = _load_test_character()
            with patch("web_service.load_characters", return_value=[character]), patch(
                "web_service.current_config", test_config
            ):
                app = create_app()
                client = app.test_client()
                resp = client.get("/free-play")
                html = resp.data.decode()
                for name in selectable:
                    self.assertIn(f"value='{name}'", html)
                for name in hidden:
                    self.assertNotIn(f"value='{name}'", html)

    @patch("rpg.game_state.random.randint", return_value=20)
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
                            "text": npc_action_text,
                            "type": "action",
                            "related-triplet": 1,
                            "related-attribute": "leadership",
                        },
                        {
                            "text": "We should gather more intel first.",
                            "type": "chat",
                            "related-triplet": "None",
                            "related-attribute": "None",
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
            test_config = GameConfig(
                enabled_factions=(
                    "test_character",
                    "CivilSociety",
                    "ScientificCommunity",
                )
            )
            with patch("web_service.load_characters", return_value=[character]), patch(
                "web_service.current_config", test_config
            ):
                app = create_app()
                client = app.test_client()

                resp = client.get("/")
                page = resp.data.decode()
                self.assertEqual(resp.status_code, 200)
                self.assertIn("Keep the Future Human Survival RPG", page)
                self.assertIn("Free Play", page)
                self.assertIn("Campaign", page)

                start_resp = client.get("/start")
                start_page = start_resp.data.decode()
                self.assertEqual(start_resp.status_code, 200)
                self.assertIn("Talk", start_page)
                self.assertIn("Scenario Overview", start_page)
                self.assertIn("Test scenario summary.", start_page)

                convo_resp = client.get("/actions", query_string={"character": "0"})
                convo_page = convo_resp.data.decode()
                self.assertEqual(convo_resp.status_code, 200)
                self.assertIn("Conversation with", convo_page)
                self.assertIn(character.display_name, convo_page)
                self.assertIn("No conversation yet", convo_page)
                starter_text = "What worries you most?"
                self.assertIn(starter_text, convo_page)

                chat_payload = json.dumps(
                    ResponseOption(text=starter_text, type="chat").to_payload()
                )
                follow_resp = client.post(
                    "/actions",
                    data={"character": "0", "response": chat_payload},
                    follow_redirects=True,
                )
                follow_page = follow_resp.data.decode()
                self.assertIn("<em>(chat)</em>", follow_page)
                self.assertIn("Action 1 [Leadership]", follow_page)
                self.assertIn("title='Coordinate oversight teams'", follow_page)

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
                self.assertEqual(action_resp.status_code, 200)
                self.assertIn("Action Outcome", action_page)
                self.assertIn("Succeeded", action_page)
                self.assertIn(npc_action_text, action_page)

                inst_resp = client.get("/instructions")
                inst_page = inst_resp.data.decode()
                self.assertEqual(inst_resp.status_code, 200)
                self.assertIn("Instructions", inst_page)
                self.assertIn("Reference material", inst_page)

                reset_resp = client.post("/reset", follow_redirects=True)
                reset_page = reset_resp.data.decode()
                self.assertEqual(reset_resp.request.path, "/")
                self.assertIn("Free Play", reset_page)


class SessionHandlingTest(unittest.TestCase):
    def setUp(self):
        self.character = _load_test_character()
        self.test_config = GameConfig(
            enabled_factions=(
                "test_character",
                "CivilSociety",
                "ScientificCommunity",
            )
        )

    def _app(self) -> Flask:
        with patch("rpg.character.genai"), patch("rpg.assessment_agent.genai"), patch(
            "web_service.genai"
        ):
            with patch("web_service.load_characters", return_value=[self.character]), patch(
                "web_service.current_config", self.test_config
            ):
                return create_app()

    def test_actions_redirects_when_character_invalid_and_no_session(self):
        app = self._app()
        client = app.test_client()

        resp = client.get("/actions", query_string={"character": "999"})

        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp.location, "/start")
        cookies = resp.headers.getlist("Set-Cookie")
        self.assertTrue(
            any(SESSION_COOKIE_NAME in cookie for cookie in cookies),
            "Expected a new session cookie to be issued",
        )

    def test_session_cookie_reused_between_requests(self):
        app = self._app()
        client = app.test_client()

        first = client.get("/")
        first_cookies = first.headers.getlist("Set-Cookie")
        self.assertTrue(any(SESSION_COOKIE_NAME in cookie for cookie in first_cookies))

        second = client.get("/start")
        self.assertFalse(
            any(SESSION_COOKIE_NAME in cookie for cookie in second.headers.getlist("Set-Cookie")),
            "Existing session should be reused without setting a new cookie",
        )

    def test_invalid_session_cookie_is_replaced(self):
        app = self._app()
        client = app.test_client()
        client.set_cookie(SESSION_COOKIE_NAME, "invalid-session", domain="localhost")

        resp = client.get("/")

        replacement_cookies = [
            cookie
            for cookie in resp.headers.getlist("Set-Cookie")
            if cookie.startswith(f"{SESSION_COOKIE_NAME}=")
        ]
        self.assertTrue(replacement_cookies)
        self.assertTrue(
            all("invalid-session" not in cookie for cookie in replacement_cookies),
            "Invalid sessions should trigger a new session cookie",
        )


if __name__ == "__main__":
    unittest.main()
