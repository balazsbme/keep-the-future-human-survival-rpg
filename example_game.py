"""Command-line demo for folder-defined RPG characters."""

# SPDX-License-Identifier: GPL-3.0-or-later

import os
from typing import List

from rpg.character import FolderCharacter
from rpg.game_state import GameState


def load_characters() -> List[FolderCharacter]:
    base = os.path.join(os.path.dirname(__file__), "characters")
    chars: List[FolderCharacter] = []
    for name in os.listdir(base):
        path = os.path.join(base, name)
        if os.path.isdir(path):
            chars.append(FolderCharacter(path))
    return chars


def main() -> None:
    state = GameState(load_characters())
    for idx, char in enumerate(state.characters, 1):
        print(f"{idx}. {char.name}")
    choice = int(input("Choose a character: ")) - 1
    char = state.characters[choice]
    options = char.generate_actions(state.history)
    for idx, act in enumerate(options, 1):
        print(f"{idx}. {act}")
    action = options[int(input("Choose an action: ")) - 1]
    result, scores = char.perform_action(action, state.history)
    print(result)
    state.record_action(char, action, scores)


if __name__ == "__main__":
    main()
