"""Game state storage for characters and their action history."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .character import Character


logger = logging.getLogger(__name__)


@dataclass
class GameState:
    """Store characters, their progress, and a history of actions."""

    characters: List[Character]
    history: List[Tuple[str, str]] = field(default_factory=list)
    progress: Dict[str, List[int]] = field(init=False)
    how_to_win: str = field(init=False)

    def __post_init__(self) -> None:
        """Initialize progress tracking and load "how to win" instructions.

        Returns:
            None.
        """
        logger.info("Initializing game state")
        self.progress = {c.name: [0] * len(c.triplets) for c in self.characters}
        win_path = os.path.join(os.path.dirname(__file__), "..", "how-to-win.md")
        with open(win_path, "r", encoding="utf-8") as f:
            self.how_to_win = f.read()

    def record_action(self, character: Character, action: str) -> None:
        """Record an action taken by a character.

        Args:
            character: The actor performing the action.
            action: The action undertaken.

        Returns:
            None.
        """
        logger.info("Recording action '%s' for %s", action, character.name)
        self.history.append((character.name, action))

    def update_progress(self, scores: Dict[str, List[int]]) -> None:
        """Update progress scores for all characters.

        Args:
            scores: Mapping of character name to list of progress values.

        Returns:
            None.
        """
        for name, new_scores in scores.items():
            if name not in self.progress:
                continue
            current = self.progress[name]
            for idx, score in enumerate(new_scores):
                if idx < len(current):
                    current[idx] = score

    def render_state(self) -> str:
        """Return HTML rendering of the current game state.

        Returns:
            HTML string describing character progress and history.
        """
        logger.info("Rendering game state")
        lines = [f"{name}: {scores}" for name, scores in self.progress.items()]
        if self.history:
            lines.append("History:")
            lines.extend(f"{n}: {a}" for n, a in self.history)
        return "<div id='state'>" + "<br>".join(lines) + "</div>"
