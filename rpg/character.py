"""Character abstractions backed by YAML-defined context."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from .conversation import ConversationEntry, ConversationType
from .config import load_game_config

import yaml

try:  # pragma: no cover - optional dependency
    import google.generativeai as genai
except ModuleNotFoundError:  # pragma: no cover
    genai = None


logger = logging.getLogger(__name__)


def _summarize_response_payload(payload: object) -> str:
    """Return a concise summary of a response payload for logging."""

    def preview(text: object) -> str:
        value = str(text or "")
        snippet = value[:20]
        if len(value) > 20:
            snippet += "\u2026"
        return snippet

    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("actions") or payload.get("responses") or [payload]
    else:
        return f"Unstructured payload: {payload!r}"

    parts: List[str] = []
    for idx, item in enumerate(items, 1):
        if isinstance(item, dict):
            text_preview = preview(item.get("text", ""))
            kind = item.get("type", "chat")
            if str(kind).lower() == "action":
                triplet = item.get("related-triplet", "None")
                attribute = item.get("related-attribute", "None")
                parts.append(
                    f"{idx}. [action] '{text_preview}' "
                    f"(triplet={triplet}, attribute={attribute})"
                )
            else:
                parts.append(f"{idx}. [{kind}] '{text_preview}'")
        else:
            parts.append(f"{idx}. {preview(item)}")
    return "; ".join(parts)


@dataclass(frozen=True)
class ResponseOption:
    """Container describing a response proposed by a character."""

    text: str
    type: ConversationType
    related_triplet: int | None = None
    related_attribute: str | None = None

    def to_payload(self) -> dict:
        """Return a JSON-serialisable representation of the option."""

        payload = {
            "text": self.text,
            "type": self.type,
            "related-triplet": self.related_triplet,
            "related-attribute": self.related_attribute,
        }
        return payload

    @property
    def is_action(self) -> bool:
        """Return ``True`` if this option represents an actionable choice."""

        return self.type == "action"

    @classmethod
    def from_payload(cls, data: dict) -> "ResponseOption":
        """Create a :class:`ResponseOption` from a JSON payload dictionary."""

        text = str(data.get("text", "")).strip()
        kind = str(data.get("type", "chat")).strip().lower() or "chat"
        if kind not in ("chat", "action"):
            kind = "chat"

        def _normalize_triplet(value: object) -> int | None:
            if kind != "action":
                return None
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned.lower() == "none":
                    return None
                try:
                    value = int(cleaned)
                except ValueError:
                    return None
            if isinstance(value, (int, float)):
                integer = int(value)
                return integer if integer > 0 else None
            return None

        def _normalize_attribute(value: object) -> str | None:
            if kind != "action":
                return None
            if isinstance(value, str):
                cleaned = value.strip().lower()
                if cleaned:
                    return cleaned
            return None

        triplet = _normalize_triplet(data.get("related-triplet"))
        attribute = _normalize_attribute(data.get("related-attribute"))
        return cls(text=text, type=kind, related_triplet=triplet, related_attribute=attribute)


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
        if not hasattr(self, "triplets"):
            self.triplets: Sequence[object] = []

    def attribute_score(self, attribute: str | None) -> int:
        """Return the numeric score for ``attribute`` if available."""

        return 0

    @abstractmethod
    def generate_responses(
        self,
        history: Sequence[Tuple[str, str]],
        conversation: Sequence[ConversationEntry],
        partner: "Character",
    ) -> List[ResponseOption]:
        """Return responses the character might give next."""

    def _conversation_text(self, conversation: Sequence[ConversationEntry]) -> str:
        """Format conversation entries for use in prompts."""

        if not conversation:
            return "None"
        return "\n".join(
            f"{entry.speaker}: {entry.text} [{entry.type}]"
            for entry in conversation
        )

    def _history_text(self, history: Sequence[Tuple[str, str]]) -> str:
        if not history:
            return "None"
        return "\n".join(f"{label}: {action}" for label, action in history)

    def _profile_text(self) -> str:
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

    def _format_prompt_instructions(self) -> str:
        return (
            "Return the result as a JSON array with objects in order. Each object must "
            "contain the keys 'text', 'type', 'related-triplet', and 'related-attribute'. "
            "The 'text' field holds the natural language response. The 'type' field must be "
            "either 'chat' or 'action'. For 'action' types provide the 1-based index in "
            "'related-triplet' or the string 'None' if the action is unrelated to a "
            "specific triplet, and set 'related-attribute' to one of leadership, technology, "
            "policy, or network. For 'chat' types set the related fields to 'None'. Do not "
            "include any additional commentary beyond the JSON."
        )

    def _parse_response_payload(
        self, response_text: str, max_triplet_index: int
    ) -> List[ResponseOption]:
        """Parse Gemini output into ``ResponseOption`` objects."""

        parsed_payload: object | None = None
        if response_text:
            json_candidate = response_text.strip()
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
                logger.warning("Failed to parse response JSON for %s: %s", self.name, exc)

        if parsed_payload is not None:
            logger.info(
                "Generated for %s: %s",
                self.name,
                _summarize_response_payload(parsed_payload),
            )
        else:
            logger.info(
                "Generated for %s (raw preview): %s",
                self.name,
                response_text[:50],
            )

        responses: List[ResponseOption] = []
        payload = parsed_payload
        if isinstance(payload, dict) and "actions" in payload:
            payload = payload["actions"]
        if isinstance(payload, list):
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                option = ResponseOption.from_payload(entry)
                if option.is_action and option.related_triplet:
                    if not 1 <= option.related_triplet <= max_triplet_index:
                        logger.debug(
                            "Discarding out-of-range related triplet %s for %s",
                            option.related_triplet,
                            self.name,
                        )
                        option = ResponseOption(
                            text=option.text,
                            type=option.type,
                            related_triplet=None,
                            related_attribute=option.related_attribute,
                        )
                responses.append(option)
        if not responses:
            lines = [line.strip() for line in response_text.splitlines() if line.strip()]
            if lines:
                logger.info(
                    "Falling back to line-by-line responses for %s", self.name
                )
            for line in lines:
                responses.append(ResponseOption(text=line, type="chat"))
        return responses[:3]


class YamlCharacter(Character):
    """Character defined by a YAML entry containing context and triplets."""

    def __init__(
        self,
        name: str,
        spec: dict,
        profile: dict | None = None,
        model: str = "gemini-2.5-flash",
    ):
        profile = profile or {}
        base_context = spec.get("MarkdownContext", "")
        end_states = spec.get("end_states", [])
        initial_states = spec.get("initial_states", [])
        gaps = spec.get("gaps", [])
        if not all(isinstance(lst, list) for lst in (end_states, initial_states, gaps)):
            raise ValueError("end_states, initial_states, and gaps must be lists")
        self.initial_states = initial_states
        self.end_states = end_states
        self.gaps = gaps
        self.triplets = list(zip(self.initial_states, self.end_states, self.gaps))
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

    def attribute_score(self, attribute: str | None) -> int:
        if not attribute:
            return 0
        return self._attribute_scores.get(attribute.lower(), 0)

    def _triplet_text(self) -> str:
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

    def generate_responses(
        self,
        history: Sequence[Tuple[str, str]],
        conversation: Sequence[ConversationEntry],
        partner: Character,
    ) -> List[ResponseOption]:
        logger.info("Generating responses for %s", self.name)
        partner_label = partner.display_name
        faction_focus = self.faction or "your personal priorities"
        base_prompt = (
            f"You are {self.display_name} in the 'Keep the Future Human' survival RPG game. "
            f"You are having a conversation with {partner_label}. "
            "Respond in a way that prioritizes your own goals or those of your faction before entertaining new requests.\n"
            f"Keep your response grounded in {faction_focus}, your motivations, and your capabilities.\n"
            f"Your persona is described below:\n{self._profile_text()}\n"
            "Ground your thinking in this persona and the faction context below before proposing responses.\n"
            f"**MarkdownContext**\n{self.base_context}\n**End of MarkdownContext**\n"
            "Throughout the game you are acting related to the following numbered list of triplets, describing the initial state at the start of the game, end state and the gap between them:\n"
            f"{self._triplet_text()}\n"
            "Previous actions taken by you or other faction representatives:\n"
            f"{self._history_text(history)}\n"
            "Full conversation history you are now having with the player:\n"
            f"{self._conversation_text(conversation)}\n"
            "Provide exactly one JSON response. Default to a 'chat' and 'action' type replies reinforcing your own or faction priorities unless the player's case persuades you to propose an 'action' related to the numbered triplets."
        )
        prompt = f"{base_prompt}\n{self._format_prompt_instructions()}"
        logger.debug("Prompt for %s: %s", self.name, prompt)
        response = self._model.generate_content(prompt)
        response_text = getattr(response, "text", "").strip()
        logger.debug("Raw response for %s: %s", self.name, response_text)
        options = self._parse_response_payload(response_text, len(self.triplets))
        if not options:
            logger.warning("Model returned no usable responses for %s", self.name)
            return []
        if len(options) > 1:
            logger.info(
                "Model returned %d options for %s; using top suggestions",
                len(options),
                self.name,
            )
        return options[:3]

    def perform_action(
        self,
        action: str,
        history: List[Tuple[str, str]],
    ) -> List[int]:
        logger.info("Performing action '%s' for %s", action, self.name)
        full_history = history + [(self.display_name, action)]
        context_block = (
            f"{self.base_context}\n{self._triplet_text()}\n"
            f"Action history:\n{self._history_text(full_history)}\n"
        )
        assess_prompt = (
            f"{context_block}Provide progress (0-100) for each triplet on separate lines."
        )
        logger.debug("Assess prompt: %s", assess_prompt)
        assess_resp = self._model.generate_content(assess_prompt)
        assess_text = getattr(assess_resp, "text", "")
        logger.info("Assessment for %s: %s", self.name, assess_text[:50])
        logger.debug("Assess response: %s", assess_text)
        scores: List[int] = []
        for line in assess_text.splitlines():
            line = line.strip()
            if not line:
                continue
            digits = "".join(ch for ch in line if ch.isdigit())
            if not digits:
                continue
            scores.append(max(0, min(100, int(digits))))
            if len(scores) == len(self.triplets):
                break
        return scores


class PlayerCharacter(Character):
    """Representation of the human player as a character."""

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        base_dir = Path(__file__).resolve().parent
        profile_path = base_dir / "player_character.yaml"
        profile_entries = _character_entries(_load_yaml(profile_path))
        if not profile_entries:
            raise ValueError("player_character.yaml must define at least one character entry")
        profile = dict(profile_entries[0])
        faction = str(profile.get("faction", "CivilSociety")) or "CivilSociety"
        name = str(profile.get("name", "Player")) or "Player"

        config = load_game_config()
        scenario_path = base_dir.parent / "scenarios" / f"{config.scenario}.yaml"
        try:
            scenario_specs = _mapping_from_payload(_load_yaml(scenario_path))
        except FileNotFoundError:
            logger.warning("Scenario file %s not found; player triplets unavailable", scenario_path)
            scenario_specs = {}
        context_path = base_dir.parent / "factions.yaml"
        try:
            context_specs = _mapping_from_payload(_load_yaml(context_path))
        except FileNotFoundError:
            logger.warning("Faction context file %s not found; using empty context", context_path)
            context_specs = {}

        faction_spec = scenario_specs.get(faction, {})
        context_spec = context_specs.get(faction, {})
        if isinstance(context_spec, dict) and context_spec.get("MarkdownContext"):
            faction_spec = dict(faction_spec)
            faction_spec["MarkdownContext"] = context_spec["MarkdownContext"]
        base_context = str(faction_spec.get("MarkdownContext", "")).strip()

        end_states = faction_spec.get("end_states", [])
        initial_states = faction_spec.get("initial_states", [])
        gaps = faction_spec.get("gaps", [])
        if not all(isinstance(lst, list) for lst in (end_states, initial_states, gaps)):
            end_states, initial_states, gaps = [], [], []
        self.initial_states = list(initial_states)
        self.end_states = list(end_states)
        self.gaps = list(gaps)
        self.triplets = list(zip(self.initial_states, self.end_states, self.gaps))
        weight_map = {"Critical": 4, "Large": 3, "Moderate": 2, "Small": 1}
        self.weights = []
        for gap in self.gaps:
            if isinstance(gap, dict):
                sev = gap.get("severity") or gap.get("size")
                self.weights.append(weight_map.get(sev, 1))
            else:
                self.weights.append(1)

        faction_descriptor = re.sub(r"(?<!^)(?=[A-Z])", " ", faction).strip() or faction
        faction_lower = faction_descriptor.lower()
        persona_lines = [
            f"You are {name}, a {faction_lower} strategist navigating AI governance negotiations.",
            str(profile.get("background", "")),
            "Use your coalition-building strengths to elicit concrete commitments from partners.",
        ]
        if base_context:
            persona_lines.append(
                f"Ground yourself in the detailed {faction_lower} context provided."
            )
        persona = " ".join(line for line in persona_lines if line)

        super().__init__(
            name=name,
            context=persona,
            model=model,
            faction=faction,
            perks=str(profile.get("perks", "") or ""),
            motivations=str(profile.get("motivations", "") or ""),
            background=str(profile.get("background", "") or ""),
            weaknesses=str(profile.get("weaknesses", "") or ""),
        )
        self.profile = profile
        self.base_context = base_context
        self._attribute_scores: dict[str, int] = {}
        self._faction_descriptor = faction_descriptor
        for attr in ("leadership", "technology", "policy", "network"):
            raw_value = profile.get(attr)
            if raw_value is None and attr.capitalize() in profile:
                raw_value = profile.get(attr.capitalize())
            try:
                numeric = int(raw_value)
            except (TypeError, ValueError):
                numeric = 0
            self._attribute_scores[attr] = max(0, min(10, numeric))

    def attribute_score(self, attribute: str | None) -> int:
        if not attribute:
            return 0
        return self._attribute_scores.get(attribute.lower(), 0)

    def _faction_context(self) -> str:
        if not self.base_context:
            return f"No additional {self._faction_descriptor.lower()} context provided."
        return self.base_context

    def generate_responses(
        self,
        history: Sequence[Tuple[str, str]],
        conversation: Sequence[ConversationEntry],
        partner: Character,
    ) -> List[ResponseOption]:
        logger.info("Generating player responses against %s", partner.display_name)
        partner_label = partner.display_name
        attribute_summary = ", ".join(
            f"{key.title()}: {value}" for key, value in self._attribute_scores.items()
        )
        base_prompt = (
            f"You are the {self._faction_descriptor.lower()} player in the 'Keep the Future Human' survival RPG. "
            f"You are speaking with {partner_label} from the {partner.faction or 'independent'} faction. "
            "Draw on your coalition strengths to encourage them to state concrete actions they can take. "
            "Offer exactly three concise 'chat' responses that keep the conversation moving without proposing actions yourself."
        )
        context_prompt = (
            f"\n### Your {self._faction_descriptor} Context\n{self._faction_context()}\n"
            if self.base_context
            else ""
        )
        prompt = (
            f"{base_prompt}{context_prompt}\n"
            f"Your capabilities: {attribute_summary}.\n"
            f"Previous gameplay actions:\n{self._history_text(history)}\n"
            f"Conversation so far:\n{self._conversation_text(conversation)}\n"
            f"Your profile:\n{self._profile_text()}\n"
            f"{self._format_prompt_instructions()}"
        )
        logger.debug("Player prompt: %s", prompt)
        response = self._model.generate_content(prompt)
        response_text = getattr(response, "text", "").strip()
        logger.debug("Raw player response: %s", response_text)
        options = self._parse_response_payload(response_text, len(partner.triplets))
        chat_options: List[ResponseOption] = []
        seen_texts: set[str] = set()
        partner_name = getattr(partner, "name", getattr(partner, "display_name", "partner"))
        for option in options:
            raw_text = option.text.strip()
            if not raw_text:
                text = "Can you elaborate on that?"
                logger.info(
                    "Player model produced blank text; using default prompt for %s",
                    partner_name,
                )
            else:
                text = raw_text
            if text.lower() in seen_texts:
                continue
            if option.is_action:
                logger.info(
                    "Player model suggested action '%s'; reclassifying as chat option",
                    text,
                )
                chat_option = ResponseOption(text=text, type="chat")
            else:
                chat_option = ResponseOption(text=text, type="chat")
            chat_options.append(chat_option)
            seen_texts.add(text.lower())
            if len(chat_options) == 3:
                break

        starter_templates = [
            "It's good to connect, {name}. What's top of mind for you today?",
            "I'd love to hear your priorities right now, {name}.",
            "Where do you see the biggest opportunity to move forward, {name}?",
        ]
        fallback_templates = [
            "Could you elaborate on your approach, {name}?",
            "What support would help you move forward, {name}?",
            "How do you see this unfolding next, {name}?",
        ]
        supplemental_templates = list(starter_templates if not conversation else [])
        supplemental_templates.extend(fallback_templates)

        if not chat_options:
            if not conversation:
                logger.info(
                    "No generated starters for %s; using fallback templates",
                    partner_name,
                )
            else:
                logger.warning(
                    "No chat options generated for player; using fallback prompts"
                )

        if len(chat_options) < 3:
            logger.info(
                "Supplementing player options with fallback templates for %s",
                partner_name,
            )
        for template in supplemental_templates:
            if len(chat_options) >= 3:
                break
            text = template.format(name=partner_name)
            if text.lower() in seen_texts:
                continue
            chat_options.append(ResponseOption(text=text, type="chat"))
            seen_texts.add(text.lower())
            logger.info(
                "Added fallback chat template response for %s: %s",
                partner_name,
                text,
            )

        while len(chat_options) < 3:
            text = f"Tell me more about your priorities, {partner_name}."
            if text.lower() in seen_texts:
                text = "I'm interested in what you think comes next."
            chat_options.append(ResponseOption(text=text, type="chat"))
            seen_texts.add(text.lower())
            logger.info(
                "Added final hardcoded chat prompt for %s to reach three options",
                partner_name,
            )

        return chat_options[:3]


def _load_yaml(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _character_entries(payload: object) -> Sequence[dict]:
    if isinstance(payload, dict):
        if "Characters" in payload and isinstance(payload["Characters"], list):
            return [entry for entry in payload["Characters"] if isinstance(entry, dict)]
        return [
            dict({"name": name}, **profile)
            for name, profile in payload.items()
            if isinstance(profile, dict)
        ]
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    return []


def _mapping_from_payload(payload: object) -> dict:
    if isinstance(payload, dict):
        factions = payload.get("factions")
        if isinstance(factions, dict):
            return {k: v for k, v in factions.items() if isinstance(v, dict)}
        return {k: v for k, v in payload.items() if isinstance(v, dict)}
    return {}

