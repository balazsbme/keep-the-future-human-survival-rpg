# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cli_game import load_characters
from types import SimpleNamespace

from rpg.character import PlayerCharacter, YamlCharacter
from rpg.assessment_agent import AssessmentAgent
from rpg.config import GameConfig

CHARACTERS_FILE = os.path.join(
    os.path.dirname(__file__), "fixtures", "characters.yaml"
)
SCENARIO_FILE = os.path.join(
    os.path.dirname(__file__), "fixtures", "scenarios", "complete.yaml"
)
FACTIONS_FILE = os.path.join(
    os.path.dirname(__file__), "fixtures", "factions.yaml"
)


class YamlCharacterTest(unittest.TestCase):
    @patch("rpg.assessment_agent.genai")
    @patch("rpg.character.genai")
    def test_generate_and_answer(self, mock_char_genai, mock_assess_genai):
        mock_action_model = MagicMock()
        mock_assess_model = MagicMock()
        mock_action_model.generate_content.return_value = MagicMock(
            text="```json\n"
            + json.dumps(
                [
                    {
                        "text": "Act1",
                        "type": "action",
                        "related-triplet": 1,
                        "related-attribute": "leadership",
                    },
                    {
                        "text": "Tell me about your readiness.",
                        "type": "chat",
                        "related-triplet": "None",
                        "related-attribute": "None",
                    },
                    {
                        "text": "How could allies help?",
                        "type": "chat",
                        "related-triplet": "None",
                        "related-attribute": "None",
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

        with open(CHARACTERS_FILE, "r", encoding="utf-8") as fh:
            character_payload = yaml.safe_load(fh)
        with open(SCENARIO_FILE, "r", encoding="utf-8") as fh:
            faction_payload = yaml.safe_load(fh) or {}
        with open(FACTIONS_FILE, "r", encoding="utf-8") as fh:
            faction_contexts = yaml.safe_load(fh) or {}
        profile = character_payload["Characters"][0]
        faction_spec = dict(faction_payload[profile["faction"]])
        faction_context = faction_contexts.get(profile["faction"], {})
        if isinstance(faction_context, dict) and faction_context.get("MarkdownContext"):
            faction_spec["MarkdownContext"] = faction_context["MarkdownContext"]
        char = YamlCharacter(profile["name"], faction_spec, profile)
        partner = SimpleNamespace(
            display_name="Player", faction="CivilSociety", triplets=char.triplets
        )
        with self.assertLogs("rpg.character", level="WARNING") as log_ctx:
            actions = char.generate_responses([], [], partner, partner_credibility=50)
        prompt_used = mock_action_model.generate_content.call_args_list[0][0][0]
        self.assertIn("end1", prompt_used)
        self.assertIn("size: Small", prompt_used)
        self.assertIn("Base context for test character.", prompt_used)
        self.assertIn("aligned with your motivations and capabilities", prompt_used)
        self.assertIn("Perks: Detailed planner", prompt_used)
        self.assertIn("Weaknesses: Struggles to prioritize", prompt_used)
        self.assertIn("Return the result as a JSON array", prompt_used)
        self.assertIn("related-attribute", prompt_used)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].type, "action")
        self.assertEqual(actions[0].related_triplet, 1)
        self.assertEqual(actions[0].related_attribute, "leadership")
        self.assertTrue(
            any("using top suggestions" in entry for entry in log_ctx.output),
            log_ctx.output,
        )
        assessor = AssessmentAgent()
        scores = assessor.assess([char], [])[char.progress_key]
        self.assertEqual(scores, [10, 20, 30])

    @patch("rpg.character.get_cache_manager")
    @patch("rpg.character.genai")
    def test_generate_responses_uses_cached_context(self, mock_char_genai, mock_cache_mgr):
        cache_config = object()
        fake_manager = MagicMock()
        fake_manager.get_cached_config.return_value = cache_config
        mock_cache_mgr.return_value = fake_manager

        mock_action_model = MagicMock()
        mock_action_model.generate_content.return_value = MagicMock(
            text=json.dumps(
                [
                    {
                        "text": "Discuss the latest request.",
                        "type": "chat",
                        "related-triplet": "None",
                        "related-attribute": "None",
                    }
                ]
            )
        )
        mock_char_genai.GenerativeModel.return_value = mock_action_model

        with open(CHARACTERS_FILE, "r", encoding="utf-8") as fh:
            character_payload = yaml.safe_load(fh)
        with open(SCENARIO_FILE, "r", encoding="utf-8") as fh:
            faction_payload = yaml.safe_load(fh) or {}
        with open(FACTIONS_FILE, "r", encoding="utf-8") as fh:
            faction_contexts = yaml.safe_load(fh) or {}
        profile = character_payload["Characters"][0]
        faction_spec = dict(faction_payload[profile["faction"]])
        faction_context = faction_contexts.get(profile["faction"], {})
        if isinstance(faction_context, dict) and faction_context.get("MarkdownContext"):
            faction_spec["MarkdownContext"] = faction_context["MarkdownContext"]
        char = YamlCharacter(profile["name"], faction_spec, profile)
        partner = SimpleNamespace(
            display_name="Player", faction="CivilSociety", triplets=char.triplets
        )
        _ = char.generate_responses([], [], partner, partner_credibility=80)
        fake_manager.get_cached_config.assert_called_once()
        called_args, called_kwargs = mock_action_model.generate_content.call_args
        self.assertIn("Use the cached persona", called_args[0])
        self.assertEqual(called_kwargs.get("config"), cache_config)

    @patch("rpg.character.genai")
    def test_generate_responses_warning_for_related_triplets(self, mock_char_genai):
        mock_action_model = MagicMock()
        mock_action_model.generate_content.return_value = MagicMock(
            text=json.dumps(
                [
                    {
                        "text": "Act1",
                        "type": "action",
                        "related-triplet": 1,
                        "related-attribute": "leadership",
                    },
                    {
                        "text": "Act2",
                        "type": "action",
                        "related-triplet": 2,
                        "related-attribute": "technology",
                    },
                    {
                        "text": "Act3",
                        "type": "action",
                        "related-triplet": 3,
                        "related-attribute": "policy",
                    },
                ]
            )
        )
        mock_char_genai.GenerativeModel.return_value = mock_action_model

        with open(CHARACTERS_FILE, "r", encoding="utf-8") as fh:
            character_payload = yaml.safe_load(fh)
        with open(SCENARIO_FILE, "r", encoding="utf-8") as fh:
            faction_payload = yaml.safe_load(fh) or {}
        with open(FACTIONS_FILE, "r", encoding="utf-8") as fh:
            faction_contexts = yaml.safe_load(fh) or {}
        profile = character_payload["Characters"][0]
        faction_spec = dict(faction_payload[profile["faction"]])
        faction_context = faction_contexts.get(profile["faction"], {})
        if isinstance(faction_context, dict) and faction_context.get("MarkdownContext"):
            faction_spec["MarkdownContext"] = faction_context["MarkdownContext"]
        char = YamlCharacter(profile["name"], faction_spec, profile)

        partner = SimpleNamespace(
            display_name="Player", faction="CivilSociety", triplets=char.triplets
        )
        with self.assertLogs("rpg.character", level="WARNING") as log_ctx:
            actions = char.generate_responses([], [], partner, partner_credibility=50)

        self.assertEqual([action.text for action in actions], ["Act1"])
        self.assertTrue(
            any("using top suggestions" in msg for msg in log_ctx.output)
        )

    def test_player_character_uses_configured_faction(self):
        config = GameConfig(player_faction="ScientificCommunity")
        player = PlayerCharacter(config=config)
        self.assertEqual(player.faction, "ScientificCommunity")
        self.assertEqual(player.name, "Dr. Maya Ibarra")
        self.assertIn("scientific", player.faction_descriptor.lower())
        self.assertIn("scientific", player.guidance.lower())

    @patch("rpg.assessment_agent.get_cache_manager")
    @patch("rpg.assessment_agent.genai")
    @patch("rpg.character.genai")
    def test_assessment_agent_uses_cached_context(
        self, mock_char_genai, mock_assess_genai, mock_cache_mgr
    ):
        cache_config = object()
        fake_manager = MagicMock()
        fake_manager.get_cached_config.return_value = cache_config
        mock_cache_mgr.return_value = fake_manager

        mock_char_genai.GenerativeModel.return_value = MagicMock()
        assess_model = MagicMock()
        assess_model.generate_content.return_value = MagicMock(text="10\n20\n30")
        mock_assess_genai.GenerativeModel.return_value = assess_model

        with open(CHARACTERS_FILE, "r", encoding="utf-8") as fh:
            character_payload = yaml.safe_load(fh)
        with open(SCENARIO_FILE, "r", encoding="utf-8") as fh:
            faction_payload = yaml.safe_load(fh) or {}
        with open(FACTIONS_FILE, "r", encoding="utf-8") as fh:
            faction_contexts = yaml.safe_load(fh) or {}
        profile = character_payload["Characters"][0]
        faction_spec = dict(faction_payload[profile["faction"]])
        faction_context = faction_contexts.get(profile["faction"], {})
        if isinstance(faction_context, dict) and faction_context.get("MarkdownContext"):
            faction_spec["MarkdownContext"] = faction_context["MarkdownContext"]
        char = YamlCharacter(profile["name"], faction_spec, profile)

        agent = AssessmentAgent()
        agent.assess([char], [])

        fake_manager.get_cached_config.assert_called()
        args, kwargs = assess_model.generate_content.call_args
        self.assertIn(
            "Use the cached facts and triplet definitions when computing progress scores.",
            args[0],
        )
        self.assertEqual(kwargs.get("config"), cache_config)

    @patch("rpg.character.genai")
    def test_player_character_generates_responses(self, mock_char_genai):
        mock_player_model = MagicMock()
        mock_player_model.generate_content.return_value = MagicMock(
            text=json.dumps(
                [
                    {
                        "text": "How can you leverage allies?",
                        "type": "chat",
                        "related-triplet": "None",
                        "related-attribute": "None",
                    },
                    {
                        "text": "Coordinate oversight teams",
                        "type": "action",
                        "related-triplet": 1,
                        "related-attribute": "leadership",
                    },
                ]
            )
        )
        mock_char_genai.GenerativeModel.return_value = mock_player_model
        player = PlayerCharacter()
        partner = SimpleNamespace(
            display_name="NPC", faction="Allies", triplets=[(1, 2, 3)]
        )
        options = player.generate_responses([], [], partner, partner_credibility=50)
        expected_texts = [
            "It's good to connect, NPC. What's top of mind for you today?",
            "I'd love to hear your priorities right now, NPC.",
            "Where do you see the biggest opportunity to move forward, NPC?",
        ]
        self.assertEqual([opt.text for opt in options], expected_texts)
        self.assertTrue(all(option.type == "chat" for option in options))

    @patch("rpg.character.genai")
    def test_load_characters_merges_markdown_context(self, mock_char_genai):
        mock_char_genai.GenerativeModel.return_value = MagicMock()
        config = GameConfig(enabled_factions=("test_character",))
        characters = load_characters(
            character_file=CHARACTERS_FILE,
            scenario_file=SCENARIO_FILE,
            factions_file=FACTIONS_FILE,
            config=config,
        )
        self.assertEqual(len(characters), 1)
        self.assertEqual(
            characters[0].base_context.strip(), "Base context for test character."
        )
        self.assertEqual(characters[0].gaps[0]["explanation"], "gap1")


if __name__ == "__main__":
    unittest.main()
