"""Character abstractions backed by YAML-defined context."""

# SPDX-License-Identifier: GPL-3.0-or-later

import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

try:  # pragma: no cover - optional dependency
    import google.generativeai as genai
except ModuleNotFoundError:  # pragma: no cover
    genai = None


logger = logging.getLogger(__name__)


class Character(ABC):
    """Abstract base class defining character interactions."""

    def __init__(self, name: str, context: str, model: str = "gemini-2.5-flash"):
        """Initialize a character.

        Args:
            name: Character name.
            context: Base context used for prompt generation.
            model: Generative model identifier.

        Returns:
            None.
        """
        self.name = name
        self.context = context
        if genai is None:  # pragma: no cover - env without dependency
            raise ModuleNotFoundError("google-generativeai not installed")
        self._model = genai.GenerativeModel(model)

    @abstractmethod
    def generate_actions(self, history: List[Tuple[str, str]]) -> List[str]:
        """Return three possible actions a player might request.

        Args:
            history: Prior actions taken in the game.

        Returns:
            A list of up to three proposed actions.
        """

    @abstractmethod
    def perform_action(
        self, action: str, history: List[Tuple[str, str]]
    ) -> List[int]:
        """Assess progress after performing ``action``.

        Args:
            action: The requested action to perform.
            history: Prior actions taken in the game.

        Returns:
            Updated progress scores for the character.
        """


class YamlCharacter(Character):
    """Character defined by a YAML entry containing context and triplets."""

    def __init__(self, name: str, spec: dict, model: str = "gemini-2.5-flash"):
        """Create a character from YAML ``spec`` data.

        Args:
            name: Character name.
            spec: Dictionary with ``MarkdownContext``, ``conditions``,
                ``current_state`` and ``gaps`` keys.
            model: Generative model identifier.

        Returns:
            None.
        """
        base_context = spec.get("MarkdownContext", "")
        conditions = spec.get("conditions", [])
        current = spec.get("current_state", [])
        gaps = spec.get("gaps", [])
        if not all(isinstance(lst, list) for lst in (conditions, current, gaps)):
            raise ValueError(
                "conditions, current_state, and gaps must be lists"
            )
        self.current = current
        self.conditions = conditions
        self.gaps = gaps
        self.triplets = list(zip(self.current, self.conditions, self.gaps))
        super().__init__(name, base_context, model)
        self.base_context = base_context

    def _triplet_text(self) -> str:
        """Return a textual representation of triplet data.

        Returns:
            A multi-line string describing current, condition, and gap.
        """
        lines = []
        for idx, (cur, cond, gap) in enumerate(self.triplets, 1):
            gap_text = gap if isinstance(gap, str) else gap.get("explanation", str(gap))
            lines.append(
                f"{idx}. Initial state: {cur}\n   End state: {cond}\n   Gap: {gap_text}"
            )
        return "\n".join(lines)

    def _history_text(self, history: List[Tuple[str, str]]) -> str:
        """Format the action history into a readable string.

        Args:
            history: List of (actor, action) tuples.

        Returns:
            A multi-line string of past actions or 'None' if empty.
        """
        return "\n".join(f"{actor}: {act}" for actor, act in history) or "None"

    def generate_actions(self, history: List[Tuple[str, str]]) -> List[str]:
        logger.info("Generating actions for %s", self.name)
        prompt = (
            "You are an NPC character/actor in the 'Keep the Future Human' survival RPG game."
            "Your attributes, personality, preferences, motivations and relationship to other characters are described in the following section, called 'MarkdownContext'"
            f"**MarkdownContext**\n{self.base_context}\n**End of MarkdownContext**"
            f"Throughout the game you are acting related to the following numbered list of triplets, describing the initial state at the start of the game, end state and the gap between them:\n{self._triplet_text()}\n"
            f"Previous actions taken by you or other actors:\n{self._history_text(history)}\n"
            "List three numbered actions you might take next. "
            "The actions must be aligned with your motivations and capabilities, "
            "but at least one of them should address closing the gap between any of the above triplets."
        )
        logger.debug("Prompt: %s", prompt)
        response = self._model.generate_content(prompt)
        response_text = getattr(response, "text", "")
        logger.info("Generated: %s", response_text[:50])
        logger.debug("Response: %s", response_text)
        lines = [line.strip() for line in response_text.splitlines() if line.strip()]
        actions: List[str] = []
        for line in lines:
            if line[0].isdigit():
                parts = line.split(".", 1)
                act = parts[1].strip() if len(parts) > 1 else line
                actions.append(act)
            else:
                if actions:
                    actions.append(line)
            if len(actions) == 3:
                break
        return actions

    def perform_action(
        self, action: str, history: List[Tuple[str, str]]
    ) -> List[int]:
        logger.info("Performing action '%s' for %s", action, self.name)
        full_history = history + [(self.name, action)]
        context_block = (
            f"{self.base_context}\n{self._triplet_text()}\n"
            f"Action history:\n{self._history_text(full_history)}\n"
        )
        assess_prompt = (
            f"{context_block}"
            "Provide progress (0-100) for each triplet on separate lines."
        )
        logger.debug("Assess prompt: %s", assess_prompt)
        assess_resp = self._model.generate_content(assess_prompt)
        assess_text = getattr(assess_resp, "text", "")
        logger.info("Assessment: %s", assess_text[:50])
        logger.debug("Assess response: %s", assess_text)
        scores: List[int] = []
        for line in assess_text.splitlines():
            line = line.strip()
            if not line:
                continue
            num = ''.join(ch for ch in line if ch.isdigit())
            if not num:
                continue
            scores.append(max(0, min(100, int(num))))
            if len(scores) == len(self.triplets):
                break
        return scores
