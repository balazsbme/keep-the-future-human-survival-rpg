"""Command-line demo for Markdown-based RPG characters."""

# SPDX-License-Identifier: GPL-3.0-or-later

import os
from typing import List

from rpg.character import MarkdownCharacter
from rpg.game_state import GameState


def load_characters() -> List[MarkdownCharacter]:
    base = os.path.join(os.path.dirname(__file__), "characters")
    return [
        MarkdownCharacter("Alice", os.path.join(base, "alice.md")),
        MarkdownCharacter("Bob", os.path.join(base, "bob.md")),
    ]


def main() -> None:
    state = GameState(load_characters())
    for idx, char in enumerate(state.characters, 1):
        print(f"{idx}. {char.name}")
    choice = int(input("Choose a character: ")) - 1
    char = state.characters[choice]
    options = char.generate_actions()
    for idx, act in enumerate(options, 1):
        print(f"{idx}. {act}")
    action = options[int(input("Choose an action: ")) - 1]
    print(char.perform_action(action))
    state.record_action(char, action)


if __name__ == "__main__":
    main()
