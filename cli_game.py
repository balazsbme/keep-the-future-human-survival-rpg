# SPDX-License-Identifier: GPL-3.0-or-later
"""Command-line demo for YAML-defined RPG characters."""

import logging
import os
from typing import List, Sequence

import yaml

from rpg.character import YamlCharacter
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


def load_characters(
    character_file: str | None = None, factions_file: str | None = None
) -> List[YamlCharacter]:
    """Load characters linked to factions from YAML specifications."""

    base_dir = os.path.dirname(__file__)
    factions_path = factions_file or os.path.join(base_dir, "factions.yaml")
    characters_path = character_file or os.path.join(base_dir, "characters.yaml")
    factions_payload = _load_yaml(factions_path)
    characters_payload = _load_yaml(characters_path)
    factions = factions_payload if isinstance(factions_payload, dict) else {}
    characters: List[YamlCharacter] = []
    for entry in _character_entries(characters_payload):
        name = entry.get("name")
        faction_name = entry.get("faction")
        if not name or not faction_name:
            logger.warning("Skipping character with missing name or faction: %s", entry)
            continue
        faction_spec = factions.get(faction_name)
        if not isinstance(faction_spec, dict):
            logger.warning(
                "No faction specification found for %s (character %s)",
                faction_name,
                name,
            )
            continue
        characters.append(YamlCharacter(name, faction_spec, entry))
    return characters


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    state = GameState(load_characters())
    assessor = AssessmentAgent()
    for idx, char in enumerate(state.characters, 1):
        print(f"{idx}. {char.display_name}")
    choice = int(input("Choose a character: ")) - 1
    char = state.characters[choice]
    options = char.generate_actions(state.history)
    for idx, act in enumerate(options, 1):
        print(f"{idx}. {act}")
    action = options[int(input("Choose an action: ")) - 1]
    state.record_action(char, action)
    scores = assessor.assess(state.characters, state.how_to_win, state.history)
    state.update_progress(scores)


if __name__ == "__main__":
    main()
