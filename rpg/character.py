"""Character abstractions backed by YAML-defined context."""

"""Character abstractions backed by YAML-defined context."""

# SPDX-License-Identifier: GPL-3.0-or-later

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Tuple

try:  # pragma: no cover - optional dependency
    import google.generativeai as genai
except ModuleNotFoundError:  # pragma: no cover
    genai = None


logger = logging.getLogger(__name__)


def _summarize_action_payload(payload: object) -> str:
    """Return a concise summary of an action payload for logging."""

    def preview_text(text: Any) -> str:
        text_str = str(text or "")
        truncated = text_str[:20]
        if len(text_str) > 20:
            truncated += "\u2026"
        return truncated

    def format_attributes(action: dict) -> str:
        extras = [
            f"{key}={value!r}" for key, value in action.items() if key != "text"
        ]
        return ", ".join(extras) if extras else "no additional fields"

    if isinstance(payload, dict):
        actions = payload.get("actions")
        if isinstance(actions, list):
            action_items = actions
        else:
            action_items = [payload]
    elif isinstance(payload, list):
        action_items = payload
    else:
        return f"Unstructured payload: {payload!r}"

    parts = []
    for index, action in enumerate(action_items, 1):
        if isinstance(action, dict):
            text_preview = preview_text(action.get("text", ""))
            attrs = format_attributes(action)
            parts.append(f"{index}. '{text_preview}' ({attrs})")
        else:
            parts.append(f"{index}. {action!r}")
    return "; ".join(parts)


@dataclass(frozen=True)
class ResponseOption:
    """Container describing an action proposed by a character."""

    text: str
    # TODO: implement type: either 'action' or 'chat'
    type: str
    related_triplet: int | None = None
    related_attribute: str | None = None

    def to_payload(self) -> dict:
        """Return a JSON-serialisable representation of the option."""

        payload = {"text": self.text}
        payload["related-triplet"] = self.related_triplet
        payload["related-attribute"] = self.related_attribute
        return payload

    @classmethod
    def from_payload(cls, data: dict) -> "ResponseOption":
        """Create an :class:`ResponseOption` from a JSON payload dictionary."""

        text = str(data.get("text", "")).strip()
        raw_triplet = data.get("related-triplet")
        related_triplet: int | None
        if isinstance(raw_triplet, str):
            cleaned = raw_triplet.strip()
            if cleaned.lower() == "none":
                related_triplet = None
            else:
                try:
                    related_triplet = int(cleaned)
                except ValueError:
                    related_triplet = None
        elif isinstance(raw_triplet, (int, float)):
            related_triplet = int(raw_triplet)
        else:
            related_triplet = None
        raw_attribute = data.get("related-attribute")
        related_attribute = (
            str(raw_attribute).strip().lower() if isinstance(raw_attribute, str) else None
        )
        return cls(
            text=text,
            related_triplet=related_triplet,
            related_attribute=related_attribute or None,
        )


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

    def attribute_score(self, attribute: str | None) -> int:
        """Return the numeric score for ``attribute`` if available."""

        return 0

    # TODO: add conversation history and the character instance of the conversation partner
    @abstractmethod
    def generate_response(self, history: List[Tuple[str, str]]) -> List[ResponseOption]:
        """Return three possible actions a player might request.

        Args:
            history: Prior actions taken in the game.
            conversation: Full conversation history between this character and another.

        Returns:
            A list response options.
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

    # TODO: construct base and format prompt and use in subclasses
    def common_base_prompt(self) -> str:
            
        base_prompt = f"You are {self.display_name} in the 'Keep the Future Human' survival RPG game. "
            f"You are having a conversation with <<ADD HERE CONVERSATION PARTNER AND ITS FACTION>> about the next actions you will personally attempt to take to advance your faction's goals."
            f"Your persona is described below: \n{self._profile_text()}\n"
            "Ground your thinking in this persona and the faction context below before proposing responses."
            f"\n**MarkdownContext**\n{self.base_context}\n**End of MarkdownContext**\n"
            f"Throughout the game you are acting related to the following numbered list of triplets, describing the initial state at the start of the game, end state and the gap between them:\n{self._triplet_text()}\n"
            f"Previous actions taken by you or other faction representatives:\n{self._history_text(history)}\n"
            f"Full conversation history that you are now having with <<ADD HERE CONVERSATION PARTNER AND ITS FACTION>>: \n ADD HERE THE VARIABLE \n"

        return base_prompt

    def format_prompt(self) -> str:
        format_prompt = "Return the result as a JSON array with objects in order. Each object must contain the keys 'text', 'type', 'related-triplet', and 'related-attribute'. "
            "The 'text' field holds the action description. The 'type' filed is either 'action' or 'chat'. The 'related-triplet' field must contain the 1-based index of the triplet primarily addressed by the action or the string 'None' if the action is not focused on a single triplet. "
            "The 'related-attribute' field must be one of leadership, technology, policy, or network indicating which of your attributes best aligns with the action. "
            "Do not mention in the output the triplets nor any of their parts directly. Output only the JSON without additional commentary."

        return format_prompt

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
        self._attribute_scores: dict[str, int] = {}
        for attr in ("leadership", "technology", "policy", "network"):
            raw_value = profile.get(attr)
            if raw_value is None and attr.capitalize() in profile:
                raw_value = profile.get(attr.capitalize())
            try:
                numeric = int(raw_value)
            except (TypeError, ValueError):
                numeric = 0
            self._attribute_scores[attr] = max(0, min(10, numeric))

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

    def generate_response(self, history: List[Tuple[str, str]]) -> List[ResponseOption]:
        logger.info("Generating actions for %s", self.name)
        # TODO: fix the prompt construction, and refactor the result handling.
        prompt = (
            f"{self.common_base_prompt()}"
            " Provide a single response to continue the ongoing conversation. Consider proposing an action or just continue the chat by providing answer to a question or ask <<ADD HERE CONVERSATION PARTNER AND ITS FACTION>> something that might interest you. "
            "Your proposed actions MUST be aligned with your motivations and capabilities (mark 'related-triplet' as 'None'), and you SHOULD try to propose an action that clearly works on closing a gap from the numbered triplets (mark the 'related-triplet' its index). "

            f"{self.format_prompt()}"
        )
        logger.debug("Prompt: %s", prompt)
        response = self._model.generate_content(prompt)
        response_text = getattr(response, "text", "").strip()
        logger.debug("Response: %s", response_text)
        actions: List[ResponseOption] = []
        related_triplet_count: int | None = None
        unrelated_triplet_count: int | None = None
        parsed_payload: object | None = None
        if response_text:
            json_candidate = response_text
            if json_candidate.startswith("```"):
                fence_match = re.match(
                    r"^```[a-zA-Z0-9_-]*\s*\n(?P<body>.*)",
                    json_candidate,
                    re.DOTALL,
                )
                if fence_match:
                    body = fence_match.group("body")
                    closing_index = body.rfind("```")
                    if closing_index != -1 and body[closing_index:].strip().startswith(
                        "```"
                    ):
                        body = body[:closing_index]
                    json_candidate = body.strip()
            try:
                parsed_payload = json.loads(json_candidate)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Failed to parse action JSON for %s: %s",
                    self.name,
                    exc,
                )
        if parsed_payload is not None:
            logger.info(
                "Generated for %s: %s",
                self.name,
                _summarize_action_payload(parsed_payload),
            )
        else:
            logger.info(
                "Generated for %s (raw preview): %s",
                self.name,
                response_text[:50],
            )
        actions: List[ActionOption] = []
        related_triplet_count: int | None = None
        unrelated_triplet_count: int | None = None

        payload = parsed_payload
        if payload is not None:
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

            def normalize_attribute(value: object) -> str | None:
                if isinstance(value, str):
                    cleaned = value.strip().lower()
                    if cleaned in self._attribute_scores:
                        return cleaned
                return None

            if isinstance(payload, dict) and "actions" in payload:
                payload = payload["actions"]

# TODO: finish merging this file
<<<<<<< HEAD
                if isinstance(payload, list):
                    related_triplet_count = 0
                    unrelated_triplet_count = 0
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
                        attribute_value = normalize_attribute(
                            entry.get("related-attribute")
                        )
                        if related_value is not None:
                            related_triplet_count += 1
                        else:
                            if unrelated_triplet_count is not None:
                                unrelated_triplet_count += 1
                        actions.append(
                            ResponseOption(
                                text=text,
                                related_triplet=related_value,
                                related_attribute=attribute_value,
                            )
                        )
                        if len(actions) == 3:
                            break
                else:
                    logger.warning(
                        "Unexpected JSON structure for %s actions: %r",
                        self.name,
                        payload,
=======
            if isinstance(payload, list):
                related_triplet_count = 0
                unrelated_triplet_count = 0
                for entry in payload:
                    if not isinstance(entry, dict):
                        continue
                    text = str(entry.get("text", "")).strip()
                    if not text:
                        continue
                    related_value = normalize_related(entry.get("related-triplet"))
                    attribute_value = normalize_attribute(
                        entry.get("related-attribute")
>>>>>>> c875c726d9766eaae87a2a997fd3c5e1c29f4c1f
                    )
                    if related_value is None:
                        unrelated_triplet_count = (unrelated_triplet_count or 0) + 1
                    else:
                        related_triplet_count = (related_triplet_count or 0) + 1
                    actions.append(
                        ActionOption(
                            text=text,
                            related_triplet=related_value,
                            related_attribute=attribute_value,
                        )
                    )
                    if len(actions) == 3:
                        break
            else:
                logger.warning(
                    "Unexpected JSON structure for %s actions: %r",
                    self.name,
                    payload,
                )

        if related_triplet_count is not None:
            unrelated_value = unrelated_triplet_count or 0
            if related_triplet_count < 1 or unrelated_value < 1:
                logger.warning(
                    "Expected at least one triplet-related and one unrelated action for %s but got %d related and %d unrelated",
                    self.name,
                    related_triplet_count,
                    unrelated_value,
                )

        if not actions:
            lines = [line.strip() for line in response_text.splitlines() if line.strip()]
            for line in lines:
                if line[0].isdigit():
                    parts = line.split(".", 1)
                    act = parts[1].strip() if len(parts) > 1 else line
                    actions.append(ResponseOption(text=act))
                else:
                    if actions:
                        actions.append(ResponseOption(text=line))
                if len(actions) == 3:
                    break
        return actions[:3]

    def attribute_score(self, attribute: str | None) -> int:
        if not attribute:
            return 0
        return self._attribute_scores.get(attribute.lower(), 0)

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

class PlayerCharacter(Character):

    def generate_response(self, history):

        player_prompt = "Provide exactly three responses you might say to continue the ongoing conversation. You are trying to persuade the <<ADD HERE CONVERSATION PARTNER AND ITS FACTION>> to propose actions that they are willing to take. You may ask about their list of triplets. Also you might ask about their background, motivations, goals or their views on their own or other factions. Whatever you think they could make them propose actions."
        
        full_prompt = self.common_base_prompt() + player_prompt + self.format_prompt()
        # TODO: call Gemini API with the full prompt

