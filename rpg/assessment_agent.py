"""Centralized assessment agent for evaluating character progress."""

# SPDX-License-Identifier: GPL-3.0-or-later

import logging
from typing import Dict, List, Tuple

try:  # pragma: no cover - optional dependency
    import google.generativeai as genai
except ModuleNotFoundError:  # pragma: no cover
    genai = None

from .character import Character

logger = logging.getLogger(__name__)


class AssessmentAgent:
    """Assess progress for all characters after each action."""

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        """Initialize the assessment agent.

        Args:
            model: Generative model identifier.
        """
        if genai is None:  # pragma: no cover - environment without dependency
            raise ModuleNotFoundError("google-generativeai not installed")
        self._model = genai.GenerativeModel(model)

    def assess(
        self,
        characters: List[Character],
        how_to_win: str,
        history: List[Tuple[str, str]],
    ) -> Dict[str, List[int]]:
        """Assess all triplets for each character.

        Args:
            characters: List of game characters.
            how_to_win: Baseline script content.
            history: List of (actor, action) tuples performed so far.

        Returns:
            Mapping of character name to list of progress scores.
        """
        actor_list = ", ".join(c.name for c in characters)
        history_text = "\n".join(f"{actor}: {act}" for actor, act in history) or "None"
        results: Dict[str, List[int]] = {}
        for char in characters:
            context = f"{char.base_context}\n{char._triplet_text()}"
            prompt = (
                "You are the Game Master for the 'Keep the future human' survival RPG. "
                "The player is interacting with the characters and convinces them to take actions. "
                f"You assess of the following character's 'initial state - end state - gap' triplets with a 0-100 integer: {actor_list}, "
                "based on the baseline script and the performed actions.\n"
                f"The baseline script: {how_to_win}\n"
                f"Performed actions: {history_text}\n"
                f"Assess all triplets of the character {context}.\n"
                "Output ONLY an ordered list of 0-100 integers one for each triplet line-by-line."
            )
            logger.debug("Assessment prompt for %s: %s", char.name, prompt)
            response = self._model.generate_content(prompt)
            text = getattr(response, "text", "")
            logger.info("Assessment for %s: %s", char.name, text[:50])
            scores: List[int] = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                num = ''.join(ch for ch in line if ch.isdigit())
                if not num:
                    continue
                scores.append(max(0, min(100, int(num))))
                if len(scores) == len(char.triplets):
                    break
            results[char.name] = scores
        return results
