"""Character abstractions backed by YAML-defined context."""

# SPDX-License-Identifier: GPL-3.0-or-later

import json
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

    def __init__(
        self,
        name: str,
        context: str,
        model: str = "gemini-2.5-flash",
        *,
        faction: str | None = None,
        perks: str = "",
        motivations: str = "",
        background: str = "",
        weaknesses: str = "",
    ):
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
        self.faction = faction
        self.perks = perks or ""
        self.motivations = motivations or ""
        self.background = background or ""
        self.weaknesses = weaknesses or ""
        self.progress_key = faction or name
        if faction and name != faction:
            self.progress_label = f"{faction} faction"
            self.display_name = f"{name} ({faction} faction)"
        else:
            self.progress_label = self.progress_key
            self.display_name = name
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

    def __init__(
        self,
        name: str,
        spec: dict,
        profile: dict | None = None,
        model: str = "gemini-2.5-flash",
    ):
        """Create a character from YAML ``spec`` data.

        Args:
            name: Character name.
            spec: Dictionary with ``MarkdownContext``, ``initial_states``,
                ``end_states`` and ``gaps`` keys.
            model: Generative model identifier.

        Returns:
            None.
        """
        profile = profile or {}
        base_context = spec.get("MarkdownContext", "")
        end_states = spec.get("end_states", [])
        initial_states = spec.get("initial_states", [])
        gaps = spec.get("gaps", [])
        if not all(
            isinstance(lst, list) for lst in (end_states, initial_states, gaps)
        ):
            raise ValueError(
                "end_states, initial_states, and gaps must be lists"
            )
        self.initial_states = initial_states
        self.end_states = end_states
        self.gaps = gaps
        self.triplets = list(zip(self.initial_states, self.end_states, self.gaps))
        # Pre-compute weight of each gap based on its severity/size
        weight_map = {"Critical": 4, "Large": 3, "Moderate": 2, "Small": 1}
        self.weights: List[int] = []
        for gap in self.gaps:
            if isinstance(gap, dict):
                sev = gap.get("severity") or gap.get("size")
                self.weights.append(weight_map.get(sev, 1))
            else:
                self.weights.append(1)
        super().__init__(
            name,
            base_context,
            model,
            faction=profile.get("faction"),
            perks=str(profile.get("perks", "") or ""),
            motivations=str(profile.get("motivations", "") or ""),
            background=str(profile.get("background", "") or ""),
            weaknesses=str(profile.get("weaknesses", "") or ""),
        )
        self.base_context = base_context
        self.profile = profile

    def _triplet_text(self) -> str:
        """Return a textual representation of triplet data.

        Returns:
            A multi-line string describing initial state, end state, gap, and size.
        """
        lines = []
        for idx, (initial, end, gap) in enumerate(self.triplets, 1):
            if isinstance(gap, str):
                gap_text = gap
                gap_size = ""
            else:
                gap_text = gap.get("explanation", str(gap))
                gap_size = gap.get("severity") or gap.get("size")
            size_part = f" (size: {gap_size})" if gap_size else ""
            lines.append(
                f"{idx}. Initial state: {initial}\n   End state: {end}\n   Gap: {gap_text}{size_part}"
            )
        return "\n".join(lines)

    def _history_text(self, history: List[Tuple[str, str]]) -> str:
        """Format the action history into a readable string.

        Args:
            history: List of (character label, action) tuples.

        Returns:
            A multi-line string of past actions or 'None' if empty.
        """

        return "\n".join(f"{label}: {act}" for label, act in history) or "None"

    def _profile_text(self) -> str:
        """Return a textual description of the character persona."""

        lines = ["### Character Profile"]
        if self.faction:
            lines.append(f"Faction: {self.faction}")
        if self.background:
            lines.append(f"Background: {self.background}")
        if self.perks:
            lines.append(f"Perks: {self.perks}")
        if self.weaknesses:
            lines.append(f"Weaknesses: {self.weaknesses}")
        if self.motivations:
            lines.append(f"Motivations: {self.motivations}")
        return "\n".join(lines)

    def generate_actions(self, history: List[Tuple[str, str]]) -> List[str]:
        logger.info("Generating actions for %s", self.name)
        prompt = (
            f"You are {self.display_name} in the 'Keep the Future Human' survival RPG game. "
            "The player is consulting you directly about the next actions you will personally attempt to take to advance your faction's goals."
            f"\n{self._profile_text()}\n"
            "Ground your thinking in this persona and the faction context below before proposing actions."
            f"\n**MarkdownContext**\n{self.base_context}\n**End of MarkdownContext**\n"
            f"Throughout the game you are acting related to the following numbered list of triplets, describing the initial state at the start of the game, end state and the gap between them:\n{self._triplet_text()}\n"
            f"Previous actions taken by you or other faction representatives:\n{self._history_text(history)}\n"
            "Provide exactly three possible actions you might take next. "
            "The actions MUST be aligned with your motivations and capabilities, "
            "but at least one of them should address closing the gap between any of the above triplets."
            "Do not mention in the output the triplets nor any of their parts directly."
            "Return the result as a JSON array with three objects in order. Each object must contain the keys 'text' and 'related-triplet'. "
            "The 'text' field holds the action description. The 'related-triplet' field must contain the 1-based index of the triplet primarily addressed by the action or the string 'None' if the action is not focused on a single triplet."
            "Output only the JSON without additional commentary."
        )
        logger.debug("Prompt: %s", prompt)
        response = self._model.generate_content(prompt)
        response_text = getattr(response, "text", "").strip()
        logger.info("Generated for %s: %s", self.name, response_text[:50])
        logger.debug("Response: %s", response_text)
        actions: List[str] = []
        related_triplet_count: int | None = None

        if response_text:
            try:
                payload = json.loads(response_text)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Failed to parse action JSON for %s: %s", self.name, exc
                )
            else:
                max_index = len(self.triplets)

                def normalize_related(value: object) -> int | None:
                    if isinstance(value, str):
                        cleaned = value.strip()
                        if cleaned.lower() == "none":
                            return None
                        try:
                            value = int(cleaned)
                        except ValueError:
                            return None
                    if isinstance(value, (int, float)):
                        candidate = int(value)
                        if 1 <= candidate <= max_index:
                            return candidate
                        return None
                    return None

                if isinstance(payload, dict) and "actions" in payload:
                    payload = payload["actions"]

                if isinstance(payload, list):
                    related_triplet_count = 0
                    for entry in payload:
                        if not isinstance(entry, dict):
                            continue
                        text_value = entry.get("text")
                        if not isinstance(text_value, str):
                            continue
                        text = text_value.strip()
                        if not text:
                            continue
                        related_value = normalize_related(entry.get("related-triplet"))
                        if related_value is not None:
                            related_triplet_count += 1
                        actions.append(text)
                        if len(actions) == 3:
                            break
                else:
                    logger.warning(
                        "Unexpected JSON structure for %s actions: %r",
                        self.name,
                        payload,
                    )

        if related_triplet_count is not None and related_triplet_count != 1:
            logger.warning(
                "Expected exactly one action referencing a triplet for %s but got %d",
                self.name,
                related_triplet_count,
            )

        if not actions:
            lines = [line.strip() for line in response_text.splitlines() if line.strip()]
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
        return actions[:3]

    def perform_action(
        self, action: str, history: List[Tuple[str, str]]
    ) -> List[int]:
        logger.info("Performing action '%s' for %s", action, self.name)
        full_history = history + [(self.display_name, action)]
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
