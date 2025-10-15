# SPDX-License-Identifier: GPL-3.0-or-later
"""Command-line demo for YAML-defined RPG characters."""

import logging
import os
from typing import List, Sequence

import yaml

from rpg.character import YamlCharacter
from rpg.config import GameConfig, load_game_config
from rpg.game_state import GameState
from rpg.assessment_agent import AssessmentAgent


logger = logging.getLogger(__name__)


def _load_yaml(path: str) -> object:
    """Return parsed YAML content from ``path`` or an empty structure."""

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _character_entries(payload: object) -> Sequence[dict]:
    """Return iterable of character profile dictionaries from YAML payload."""

    if isinstance(payload, dict):
        if "Characters" in payload and isinstance(payload["Characters"], list):
            return [entry for entry in payload["Characters"] if isinstance(entry, dict)]
        # Backwards compatibility for legacy YAML mapping name->profile
        return [dict({"name": name}, **profile) for name, profile in payload.items() if isinstance(profile, dict)]
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    return []


def _faction_mapping(payload: object) -> dict:
    """Return a mapping of faction specifications from YAML payload."""

    if isinstance(payload, dict):
        factions = payload.get("factions")
        if isinstance(factions, dict):
            return {k: v for k, v in factions.items() if isinstance(v, dict)}
        return {k: v for k, v in payload.items() if isinstance(v, dict)}
    return {}


def load_characters(
    character_file: str | None = None,
    factions_file: str | None = None,
    *,
    scenario_file: str | None = None,
    scenario_name: str | None = None,
    config: GameConfig | None = None,
) -> List[YamlCharacter]:
    """Load characters linked to factions from scenario and context files."""

    base_dir = os.path.dirname(__file__)
    cfg = config or load_game_config()
    characters_path = character_file or os.path.join(base_dir, "characters.yaml")
    scenario_dir = os.path.join(base_dir, "scenarios")
    context_path = os.path.join(base_dir, "factions.yaml")
    resolved_scenario: str | None = None
    if scenario_file:
        resolved_scenario = scenario_file
    elif factions_file:
        logger.warning(
            "'factions_file' parameter is deprecated; please supply a scenario file instead."
        )
        resolved_scenario = factions_file
    else:
        scenario_name = (scenario_name or cfg.scenario).lower()
        resolved_scenario = os.path.join(scenario_dir, f"{scenario_name}.yaml")
    if factions_file and scenario_file:
        context_path = factions_file

    characters_payload = _load_yaml(characters_path)
    scenario_payload = _load_yaml(resolved_scenario) if resolved_scenario else {}
    context_payload = _load_yaml(context_path) if context_path else {}
    triplet_specs = _faction_mapping(scenario_payload)
    context_specs = _faction_mapping(context_payload)
    scenario_summary = ""
    if isinstance(scenario_payload, dict):
        summary_value = scenario_payload.get("summary") or scenario_payload.get("Summary")
        if isinstance(summary_value, str):
            scenario_summary = summary_value.strip()
        elif isinstance(summary_value, list):
            parts = [str(item).strip() for item in summary_value if str(item).strip()]
            scenario_summary = "\n".join(parts)
    if not triplet_specs:
        logger.error("No faction definitions found in scenario file %s", resolved_scenario)
    characters: List[YamlCharacter] = []
    for entry in _character_entries(characters_payload):
        name = entry.get("name")
        faction_name = entry.get("faction")
        if not name or not faction_name:
            logger.warning("Skipping character with missing name or faction: %s", entry)
            continue
        faction_spec = triplet_specs.get(faction_name)
        if not isinstance(faction_spec, dict):
            logger.warning(
                "No faction specification found for %s (character %s)",
                faction_name,
                name,
            )
            continue
        combined_spec = dict(faction_spec)
        context_spec = context_specs.get(faction_name)
        if isinstance(context_spec, dict):
            if context_spec.get("MarkdownContext"):
                combined_spec["MarkdownContext"] = context_spec["MarkdownContext"]
            for key, value in context_spec.items():
                combined_spec.setdefault(key, value)
        character = YamlCharacter(name, combined_spec, entry)
        if scenario_summary:
            setattr(character, "scenario_summary", scenario_summary)
        characters.append(character)
    return characters


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    state = GameState(load_characters())
    assessor = AssessmentAgent()
    player = state.player_character
    for idx, char in enumerate(state.characters, 1):
        print(f"{idx}. {char.display_name}")
    choice = int(input("Choose a character: ")) - 1
    char = state.characters[choice]
    while True:
        convo = state.conversation_history(char)
        for entry in convo:
            print(f"{entry.speaker}: {entry.text} ({entry.type})")
        responses = player.generate_responses(state.history, convo, char)
        npc_actions = list(state.available_npc_actions(char))
        combined_options = list(responses)
        existing_texts = {opt.text for opt in combined_options}
        for action in npc_actions:
            if action.text not in existing_texts:
                combined_options.append(action)
                existing_texts.add(action.text)
        for idx, opt in enumerate(combined_options, 1):
            prefix = "Action: " if opt.is_action else ""
            print(f"{idx}. {prefix}{opt.text}")
        choice = int(input("Choose a response: ")) - 1
        option = combined_options[choice]
        state.log_player_response(char, option)
        if option.is_action:
            attempt = state.attempt_action(char, option)
            if attempt.success:
                break
            while True:
                failure_text = attempt.failure_text or (
                    f"Failed '{option.text}' (attribute {attempt.attribute or 'none'}: {attempt.effective_score}, roll={attempt.roll:.2f})"
                )
                print(failure_text)
                cost = state.next_reroll_cost(char, option)
                if cost > 0:
                    prompt = f"Reroll at a credibility cost of {cost}? (y/n): "
                else:
                    prompt = "Reroll for free? (y/n): "
                reroll = input(prompt).strip().lower()
                if reroll.startswith("y"):
                    attempt = state.reroll_action(char, option)
                    if attempt.success:
                        break
                    continue
                state.finalize_failed_action(char, option)
                break
            break
        npc_responses = char.generate_responses(state.history, state.conversation_history(char), player)
        state.log_npc_responses(char, npc_responses)
        npc_actions = list(state.available_npc_actions(char))
        if npc_actions:
            print("Available actions:")
            for idx, action in enumerate(npc_actions, 1):
                print(f"{idx}. {action.text}")
    scores = assessor.assess(state.characters, state.how_to_win, state.history)
    state.update_progress(scores)


if __name__ == "__main__":
    main()
