"""Game state storage for characters and their action history."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .character import Character


@dataclass
class GameState:
    """Store characters and a history of performed actions."""

    characters: List[Character]
    history: List[Tuple[str, str]] = field(default_factory=list)

    def record_action(self, character: Character, action: str) -> None:
        """Record that ``character`` performed ``action``."""
        self.history.append((character.name, action))
