"""Centralized assessment agent for evaluating character progress."""

# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

try:  # pragma: no cover - optional dependency
    import google.generativeai as genai
except ModuleNotFoundError:  # pragma: no cover
    genai = None

from .character import Character
from .genai_cache import get_cache_manager

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

        base_context = str(char.base_context or "")
        triplet_text = char._triplet_text()
        context = f"{base_context}\n{triplet_text}"
        manager = get_cache_manager()
        cache_config = None
        cache_instruction = ""
        if manager:
            cache_segments = [f"Baseline script for evaluation:\n{how_to_win}"]
            base_context_clean = base_context.strip()
            if base_context_clean:
                cache_segments.append(
                    f"Persona context for {char.progress_label}:\n{base_context_clean}"
                )
            if triplet_text:
                cache_segments.append(
                    f"Triplet definitions for {char.progress_label}:\n{triplet_text}"
                )
            cache_hash = f"{abs(hash(how_to_win)) & 0xFFFFFFFF:08x}"
            cache_key = re.sub(
                r"[^a-z0-9_-]+",
                "-",
                f"assessment-{char.progress_key}-{cache_hash}".lower(),
            )
            cache_instruction = (
                "Use the cached baseline script, persona context, and triplet definitions when computing progress scores.\n"
            )
            try:
                cache_config = manager.get_cached_config(
                    display_name=f"assessment::{cache_key}",
                    model=self._model_name,
                    texts=cache_segments,
                    system_instruction=(
                        "You are the Game Master for the 'Keep the future human' survival RPG. "
                        "Reference the cached baseline script, persona context, and triplet definitions when producing assessment scores."
                    ),
                )
            except Exception as exc:  # pragma: no cover - cache service failure
                logger.warning(
                    "Failed to prepare assessment cache for %s: %s", char.name, exc
                )
                cache_config = None
        if cache_config is not None:
            baseline_context = cache_instruction
        else:
            baseline_context = (
                f"The baseline script: {how_to_win}\n"
                f"Assess all triplets for {char.progress_label} using the context below:\n{context}\n"
            )
        prompt = (
            "You are the Game Master for the 'Keep the future human' survival RPG. "
            "The player is interacting with the characters and convinces them to take actions. "
            f"Assess the progress for the following factions' 'initial state - end state - gap' triplets with a 0-100 integer: {faction_list}, "
            "based on the baseline script and the performed actions.\n"
            f"{baseline_context}"
            f"Performed actions: {history_text}\n"
            "Output ONLY an ordered list of 0-100 integers one for each triplet line-by-line. For example, 0 means that no relevant actions have been performed for a triplet (i.e. still the 'initial state' stands), while ~50 means that the 'gap' has been reduced by a lot, but significant gap remains, finally 100 means that the performed actions equivalently describe the 'end state'."
        )
        logger.debug("Assessment prompt for %s: %s", char.name, prompt)
        if cache_config is not None:
            response = self._get_model().generate_content(prompt, config=cache_config)
        else:
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
