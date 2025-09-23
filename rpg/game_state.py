"""Game state storage for characters and their action history."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from html import escape
from typing import Dict, List, Tuple

from .character import ActionOption, Character
from .config import GameConfig, load_game_config


logger = logging.getLogger(__name__)


GAME_CONFIG = load_game_config()
WIN_THRESHOLD = GAME_CONFIG.win_threshold
MAX_ROUNDS = GAME_CONFIG.max_rounds


@dataclass
class GameState:
    """Store characters, their progress, and a history of actions."""

    characters: List[Character]
    history: List[Tuple[str, str]] = field(default_factory=list)
    progress: Dict[str, List[int]] = field(init=False)
    weights: Dict[str, List[int]] = field(init=False)
    how_to_win: str = field(init=False)
    faction_labels: Dict[str, str] = field(init=False)
    config: GameConfig = field(init=False)

    def __post_init__(self) -> None:
        """Initialize progress tracking and load "how to win" instructions.

        Returns:
            None.
        """
        logger.info("Initializing game state")
        self.progress = {}
        self.weights = {}
        self.faction_labels = {}
        self.config = GAME_CONFIG
        for character in self.characters:
            key = character.progress_key
            if key not in self.progress:
                self.progress[key] = [0] * len(character.triplets)
                self.weights[key] = getattr(
                    character, "weights", [1] * len(character.triplets)
                )
                self.faction_labels[key] = character.progress_label
        win_path = os.path.join(os.path.dirname(__file__), "..", "how-to-win.md")
        with open(win_path, "r", encoding="utf-8") as f:
            self.how_to_win = f.read()

    def record_action(
        self, character: Character, action: ActionOption | str
    ) -> bool:
        """Attempt to record an action taken by a character.

        The action succeeds only if a random draw from a uniform [0, 10]
        distribution is strictly below the relevant character attribute score.

        Args:
            character: The faction-aligned character performing the action.
            action: The proposed action or its textual description.

        Returns:
            ``True`` if the action is recorded as successful, ``False`` otherwise.
        """

        option = action if isinstance(action, ActionOption) else ActionOption(text=str(action))
        logger.info("Evaluating action '%s' for %s", option.text, character.name)
        attribute_name = option.related_attribute
        attribute_score = character.attribute_score(attribute_name)
        if attribute_name:
            logger.info(
                "Action related attribute for %s: %s", character.name, attribute_name
            )
        else:
            logger.info(
                "Action for %s has no related attribute; defaulting score to 0",
                character.name,
            )
        logger.info(
            "Using attribute score %s for success threshold", attribute_score
        )
        sampled_value = random.uniform(0, 10)
        logger.info("Sampled %.2f from uniform[0, 10]", sampled_value)
        success = sampled_value < attribute_score
        if success:
            logger.info("Action succeeded; recording in history")
            self.history.append((character.display_name, option.text))
        else:
            logger.info("Action failed; recording failure entry")
            label = attribute_name or "none"
            failure_text = (
                f"Failed '{option.text}' (attribute {label}: {attribute_score}, roll={sampled_value:.2f})"
            )
            self.history.append((character.display_name, failure_text))
        return success

    def update_progress(self, scores: Dict[str, List[int]]) -> None:
        """Update progress scores for all characters.

        Args:
            scores: Mapping of character name to list of progress values.

        Returns:
            None.
        """
        for key, new_scores in scores.items():
            if key not in self.progress:
                continue
            current = self.progress[key]
            for idx, score in enumerate(new_scores):
                if idx < len(current):
                    current[idx] = score

    def _faction_weighted_score(self, key: str) -> int:
        """Return weighted score for a single faction."""

        scores = self.progress.get(key, [])
        weights = self.weights.get(key, [])
        total = sum(weights)
        if not scores or total == 0:
            return 0
        return round(sum(s * w for s, w in zip(scores, weights)) / total)

    def final_weighted_score(self) -> int:
        """Return weighted score across all factions."""

        totals = []
        for key in self.progress:
            weight_total = sum(self.weights.get(key, []))
            faction_score = self._faction_weighted_score(key)
            totals.append((faction_score, weight_total))
        grand_total = sum(w for _, w in totals)
        if grand_total == 0:
            return 0
        return round(sum(score * w for score, w in totals) / grand_total)

    def render_state(self) -> str:
        """Return HTML rendering of the current game state.

        Returns:
            HTML string describing character progress and history.
        """
        logger.info("Rendering game state")
        lines = []
        for key, scores in self.progress.items():
            label = escape(self.faction_labels.get(key, key), quote=False)
            lines.append(
                f"{label}: {scores} (weighted: {self._faction_weighted_score(key)})"
            )
        if self.history:
            hist_items = "".join(
                f"<li><strong>{escape(n, quote=False)}</strong>: {escape(a, quote=False)}</li>"
                for n, a in self.history
            )
            lines.append(f"<h2>Action History</h2><ol>{hist_items}</ol>")
        lines.append(f"Final weighted score: {self.final_weighted_score()}")
        return "<div id='state'>" + "<br>".join(lines) + "</div>"
