"""Game state storage for characters and their action history."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .character import Character


@dataclass
class GameState:
    """Store characters, their progress, and a history of actions."""

    characters: List[Character]
    history: List[Tuple[str, str]] = field(default_factory=list)
    progress: Dict[str, List[int]] = field(init=False)
    how_to_win: str = field(init=False)

    def __post_init__(self) -> None:
        self.progress = {c.name: [0] * len(c.triplets) for c in self.characters}
        win_path = os.path.join(os.path.dirname(__file__), "..", "how-to-win.md")
        with open(win_path, "r", encoding="utf-8") as f:
            self.how_to_win = f.read()

    def record_action(
        self, character: Character, action: str, scores: List[int]
    ) -> None:
        """Record that ``character`` performed ``action`` and update progress."""
        self.history.append((character.name, action))
        if character.name in self.progress:
            current = self.progress[character.name]
            for idx, score in enumerate(scores):
                if idx < len(current):
                    current[idx] = score

    def render_state(self) -> str:
        lines = []
        for name, scores in self.progress.items():
            lines.append(f"{name}: {scores}")
        return "<div id='state'>" + "<br>".join(lines) + "</div>"
