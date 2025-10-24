# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os
import re
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import tempfile
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluations.player_service import create_app
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


class PlayerServiceTest(unittest.TestCase):
    def test_progress_page_lists_all_scores_and_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("evaluations.players.random.choice") as mock_choice, patch(
                "rpg.character.genai"
            ) as mock_char_genai, patch(
                "rpg.assessment_agent.genai"
            ) as mock_assess_genai, patch(
                "evaluations.players.genai"
            ) as mock_players_genai:
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
                                "text": "B",
                                "type": "action",
                                "related-triplet": "None",
                                "related-attribute": "technology",
                            },
                            {
                                "text": "C",
                                "type": "action",
                                "related-triplet": "None",
                                "related-attribute": "policy",
                            },
                        ]
                    )
                )
                mock_assess_model.generate_content.return_value = MagicMock(
                    text="10\n20\n30"
                )
                mock_char_genai.GenerativeModel.return_value = mock_action_model
                mock_assess_genai.GenerativeModel.return_value = mock_assess_model
                mock_players_genai.GenerativeModel.return_value = MagicMock()
                character = _load_test_character()

                def choice_side_effect(options):
                    if options and isinstance(options[0], YamlCharacter):
                        return character
                    return options[0]

                mock_choice.side_effect = choice_side_effect
                with patch(
                    "evaluations.player_service.load_characters",
                    return_value=[character],
                ):
                    with patch("rpg.game_state.random.randint", return_value=20):
                        app = create_app(log_dir=tmpdir)
                        client = app.test_client()
                        resp = client.post(
                            "/",
                            data={
                                "player": "action-first",
                                "rounds": "1",
                                "games": "2",
                            },
                            follow_redirects=True,
                )

                page = resp.data.decode()
                self.assertIn("Player Manager Progress", page)
                self.assertIn("Run 1", page)
                self.assertIn("Action-first opportunist", page)
                self.assertIn("games: 2", page)
                self.assertRegex(page, r"Game 1: \d+ rounds completed")
                self.assertRegex(page, r"Game 2: \d+ rounds completed")
                self.assertIn(">Log</a>", page)
                self.assertNotIn("Selected player:", page)
                log_links = re.findall(r"/logs/([^\"']+)", page)
                self.assertGreaterEqual(len(log_links), 2)
                log_resp = client.get(f"/logs/{log_links[0]}")
                self.assertEqual(log_resp.status_code, 200)
                self.assertTrue(log_resp.data)

    def test_evaluation_buttons_present(self):
        with patch(
            "evaluations.player_service.load_characters",
            return_value=[],
        ), patch(
            "evaluations.players.genai"
        ), patch("rpg.assessment_agent.genai"):
            app = create_app()
            client = app.test_client()
        resp = client.get("/")
        page = resp.data.decode()
        self.assertIn("Baseline Assessment", page)
        self.assertIn("Consistency Assessment", page)
        self.assertIn("Action-first opportunist", page)

    def test_batch_runs_execute_multiple_configurations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("evaluations.players.random.choice") as mock_choice, patch(
                "rpg.character.genai"
            ) as mock_char_genai, patch(
                "rpg.assessment_agent.genai"
            ) as mock_assess_genai, patch(
                "evaluations.players.genai"
            ) as mock_players_genai:
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
                                "text": "B",
                                "type": "action",
                                "related-triplet": "None",
                                "related-attribute": "technology",
                            },
                            {
                                "text": "C",
                                "type": "action",
                                "related-triplet": "None",
                                "related-attribute": "policy",
                            },
                        ]
                    )
                )
                mock_assess_model.generate_content.return_value = MagicMock(
                    text="10\n20\n30"
                )
                mock_char_genai.GenerativeModel.return_value = mock_action_model
                mock_assess_genai.GenerativeModel.return_value = mock_assess_model
                mock_players_genai.GenerativeModel.return_value = MagicMock()
                character = _load_test_character()

                def choice_side_effect(options):
                    if options and isinstance(options[0], YamlCharacter):
                        return character
                    return options[0]

                mock_choice.side_effect = choice_side_effect
                with patch(
                    "evaluations.player_service.load_characters",
                    return_value=[character],
                ) as mock_load_characters:
                    with patch("rpg.game_state.random.randint", return_value=20):
                        app = create_app(log_dir=tmpdir)
                        client = app.test_client()
                        batch_payload = json.dumps(
                            [
                                {
                                    "player": "action-first",
                                    "rounds": 1,
                                    "games": 1,
                                    "scenario": "complete",
                                },
                                {
                                    "player": "random",
                                    "rounds": 1,
                                    "games": 1,
                                    "scenario": "complete",
                                    "player_config": {"label": "second"},
                                },
                            ]
                        )
                        resp = client.post(
                            "/",
                            data={"batch_runs": batch_payload},
                            follow_redirects=True,
                        )

                self.assertEqual(mock_load_characters.call_count, 2)
                page = resp.data.decode()
                self.assertIn("Total configured runs: 2", page)
                self.assertIn("Run 1", page)
                self.assertIn("Run 2", page)
                self.assertIn("Action-first opportunist", page)
                self.assertIn("Random (uniform choice)", page)
                self.assertIn("log to DB: No", page)
                self.assertNotIn("Player configuration:", page)

    def test_parallel_runs_execute_concurrently(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            thread_ids: list[int] = []
            release_event = threading.Event()
            state_lock = threading.Lock()
            state = {"first": True}

            class StubManager:
                def __init__(self, *args, **kwargs):
                    pass

                def run_sequence(self, player_key, player_instance, games, rounds):
                    thread_ids.append(threading.get_ident())
                    with state_lock:
                        is_first = state["first"]
                        if is_first:
                            state["first"] = False
                        else:
                            release_event.set()
                    if is_first:
                        if not release_event.wait(timeout=1):
                            raise AssertionError(
                                "Parallel run did not start a second worker in time"
                            )
                    return [
                        {
                            "game_number": 1,
                            "rounds": [],
                            "final_score": 0,
                            "result": "Lose",
                            "iterations": rounds,
                            "actions": 0,
                            "log_filename": None,
                        }
                    ]

            fake_characters = [
                SimpleNamespace(base_context="context", faction="Corporations")
            ]

            with patch("evaluations.player_service.PlayerManager", StubManager), patch(
                "evaluations.player_service.AssessmentAgent"
            ) as mock_assessor_cls, patch(
                "evaluations.player_service.load_characters",
                return_value=fake_characters,
            ), patch(
                "evaluations.players.genai"
            ), patch(
                "rpg.assessment_agent.genai"
            ):
                mock_assessor_cls.return_value = MagicMock()
                app = create_app(log_dir=tmpdir)
                client = app.test_client()
                batch_payload = json.dumps(
                    [
                        {
                            "player": "random",
                            "rounds": 5,
                            "games": 1,
                            "scenario": "complete",
                        },
                        {
                            "player": "action-first",
                            "rounds": 5,
                            "games": 1,
                            "scenario": "complete",
                        },
                    ]
                )
                resp = client.post(
                    "/",
                    data={"batch_runs": batch_payload, "parallel_runs": "2"},
                    follow_redirects=True,
                )

            self.assertTrue(release_event.is_set())
            self.assertEqual(len(thread_ids), 2)
            page = resp.data.decode()
            self.assertIn("Run 1", page)
            self.assertIn("Run 2", page)
            self.assertRegex(page, r"Game 1: 5 rounds completed")
            self.assertLess(page.index("Run 1"), page.index("Run 2"))


if __name__ == "__main__":
    unittest.main()
