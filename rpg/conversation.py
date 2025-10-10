"""Conversation related data structures used across the game."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ConversationType = Literal["chat", "action"]


@dataclass(frozen=True)
class ConversationEntry:
    """Represents a single utterance in the ongoing conversation."""

    speaker: str
    text: str
    type: ConversationType

    def __post_init__(self) -> None:
        """Validate the entry type is one of the accepted values."""

        if self.type not in ("chat", "action"):
            raise ValueError(f"Invalid conversation entry type: {self.type!r}")

