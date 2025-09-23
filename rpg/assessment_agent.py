"""Centralized assessment agent for evaluating character progress."""

# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
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
        # Store the genai module reference so patched versions persist per instance
        self._genai = genai
        # keep only the model name and create a per-thread model on demand
        self._model_name = model
        self._local = threading.local()

    def _get_model(self):
        """Return a thread-local GenerativeModel instance."""
        if not hasattr(self._local, "model"):
            self._local.model = self._genai.GenerativeModel(self._model_name)
        return self._local.model

    def _assess_single(
        self,
        char: Character,
        faction_list: str,
        how_to_win: str,
        history_text: str,
    ) -> List[int]:
        """Assess ``char`` returning a list of progress scores.

        This helper exists so assessments for multiple characters can be
        executed in parallel using threads when requested by the caller.
        """

        context = f"{char.base_context}\n{char._triplet_text()}"
        prompt = (
            "You are the Game Master for the 'Keep the future human' survival RPG. "
            "The player is interacting with the characters and convinces them to take actions. "
            f"Assess the progress for the following factions' 'initial state - end state - gap' triplets with a 0-100 integer: {faction_list}, "
            "based on the baseline script and the performed actions.\n"
            f"The baseline script: {how_to_win}\n"
            f"Performed actions: {history_text}\n"
            f"Assess all triplets for {char.progress_label} using the context below:\n{context}\n"
            "Output ONLY an ordered list of 0-100 integers one for each triplet line-by-line. For example, 0 means that no relevant actions have been performed for a triplet (i.e. still the 'initial state' stands), while ~50 means that the 'gap' has been reduced by a lot, but significant gap remains, finally 100 means that the performed actions equivalently describe the 'end state'."
        )
        logger.debug("Assessment prompt for %s: %s", char.name, prompt)
        response = self._get_model().generate_content(prompt)
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
        return scores

    def assess(
        self,
        characters: List[Character],
        how_to_win: str,
        history: List[Tuple[str, str]],
        parallel: bool = False,
    ) -> Dict[str, List[int]]:
        """Assess all triplets for each character.

        Args:
            characters: List of game characters.
            how_to_win: Baseline script content.
            history: List of (character label, action) tuples performed so far.

        Returns:
            Mapping of character name to list of progress scores.
        """
        faction_list = ", ".join(c.progress_label for c in characters)
        history_text = "\n".join(f"{label}: {act}" for label, act in history) or "None"

        if parallel:
            with ThreadPoolExecutor(max_workers=len(characters)) as executor:
                future_map = {
                    executor.submit(
                        self._assess_single, char, faction_list, how_to_win, history_text
                    ): char.progress_key
                    for char in characters
                }
                return {name: fut.result() for fut, name in future_map.items()}

        results: Dict[str, List[int]] = {}
        for char in characters:
            results[char.progress_key] = self._assess_single(
                char, faction_list, how_to_win, history_text
            )
        return results
