# SPDX-License-Identifier: GPL-3.0-or-later
"""Command-line demo for YAML-defined RPG characters."""

import logging
import os
from typing import List

import yaml

from rpg.character import YamlCharacter
from rpg.game_state import GameState
from rpg.assessment_agent import AssessmentAgent


def load_characters() -> List[YamlCharacter]:
    file_path = os.path.join(os.path.dirname(__file__), "characters.yaml")
    with open(file_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return [YamlCharacter(name, spec) for name, spec in data.items()]


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    state = GameState(load_characters())
    assessor = AssessmentAgent()
    for idx, char in enumerate(state.characters, 1):
        print(f"{idx}. {char.name}")
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
