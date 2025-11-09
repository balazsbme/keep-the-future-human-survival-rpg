"""Character abstractions backed by YAML-defined context."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
import logging
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple

from .conversation import ConversationEntry, ConversationType
from .logging_utils import collapse_prompt_sections
from .config import GameConfig, load_game_config
from .constants import ACTION_ATTRIBUTES
from .credibility import CREDIBILITY_PENALTY
from .genai_cache import get_cache_manager

import yaml

try:  # pragma: no cover - optional dependency
    import google.generativeai as genai
except ModuleNotFoundError:  # pragma: no cover
    genai = None


logger = logging.getLogger(__name__)

_RESPONSE_SCHEMA_PATH = Path(__file__).resolve().with_name("response_schema.json")


@lru_cache(maxsize=1)
def _response_schema_text() -> str:
    """Return the cached response schema text for prompt injection."""

    try:
        return _RESPONSE_SCHEMA_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        logger.error("Response schema definition missing at %s", _RESPONSE_SCHEMA_PATH)
        return ""


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
        if kind == "action":
            if attribute is None:
                logger.error(
                    "Action option missing related attribute (text=%r, payload=%r)",
                    text or "",
                    data,
                )
            if "{" in text:
                logger.error(
                    "Action option text appears malformed and contains '{': %r",
                    text,
                )
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
        config: GameConfig | None = None,
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
            logger.warning(
                "Using default faction label '%s' for %s", self.progress_label, name
            )
        if genai is None:  # pragma: no cover - env without dependency
            raise ModuleNotFoundError("google-generativeai not installed")
        self._model_name = model
        self._model = genai.GenerativeModel(model)
        self._cached_context_config: object | None = None
        self._context_instruction: str = ""
        self._context_fallback: str = ""
        self.config = config or load_game_config()
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
        *,
        partner_credibility: int | None = None,
        force_action: bool = False,
        conversation_cache: Mapping[str, Sequence[ConversationEntry]] | None = None,
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

    def _response_schema_variant(self) -> str:
        return "YamlCharacterResponse"

    def _format_prompt_instructions(self) -> str:
        limit = getattr(self.config, "format_prompt_character_limit", 400)
        schema_variant = self._response_schema_variant()
        schema_text = _response_schema_text()
        instructions = (
            "Return the result as a JSON array with exactly three objects in order. Each "
            "object must contain the keys 'text', 'type', 'related-triplet', and "
            "'related-attribute'. The 'text' field holds the natural language response, "
            "which should be short, at most 1-2 sentences, with a hard-limit of "
            f"{limit} characters, finally do not apply any formatting such as '*'-s. The "
            "'type' field must be either 'chat' or 'action'. Provide the 1-based index in "
            "'related-triplet', and set 'related-attribute' to one of leadership, "
            "technology, policy, or network. Do not include any additional commentary "
            "beyond the JSON."
        )
        if schema_text:
            instructions += (
                f" The JSON must validate against the '{schema_variant}' definition in the "
                "schema below:\n"
                f"{schema_text}"
            )
        return instructions

    def _format_referenced_quotes(self) -> str:
        """Return a bullet formatted list of referenced quotes for prompts."""

        quotes = getattr(self, "referenced_quotes", None)
        if not quotes:
            return ""
        lines = []
        for quote in quotes:
            text = str(quote or "").strip()
            if text:
                lines.append(f"- {text}")
        return "\n".join(lines)

    def _setup_context_cache(
        self,
        *,
        cache_id: str,
        segments: Sequence[str],
        fallback_prompt: str,
        cache_instruction: str,
        system_instruction: str | None = None,
    ) -> None:
        """Configure cached context handling for the character when available."""

        self._context_fallback = fallback_prompt
        self._context_instruction = cache_instruction
        manager = get_cache_manager()
        if not manager:
            return
        try:
            config = manager.get_cached_config(
                display_name=cache_id,
                model=self._model_name,
                texts=segments,
                system_instruction=system_instruction or cache_instruction.strip(),
            )
        except Exception as exc:  # pragma: no cover - cache service failure
            logger.warning("Failed to create cached content for %s: %s", self.name, exc)
            return
        if config is not None:
            self._cached_context_config = config

    def _context_prompt(self) -> str:
        """Return the context block appropriate for the current cache state."""

        return (
            self._context_instruction
            if self._cached_context_config is not None
            else self._context_fallback
        )

    def _generate_with_context(self, prompt: str):
        """Generate content while attaching cached context when available."""

        try:
            if self._cached_context_config is not None:
                return self._model.generate_content(
                    prompt, config=self._cached_context_config
                )
            return self._model.generate_content(prompt)
        except Exception as exc:  # pragma: no cover - network/auth failures
            logger.warning(
                "Gemini request for %s using model %s failed: %s",
                self.name,
                getattr(self, "_model_name", "unknown"),
                exc,
            )
            raise

    @property
    def cached_context_config(self) -> object | None:
        """Expose the cached context configuration for collaborators."""

        return self._cached_context_config

    @property
    def context_instruction(self) -> str:
        return self._context_instruction

    @property
    def context_fallback(self) -> str:
        return self._context_fallback

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
        elif isinstance(payload, dict):
            payload = [payload]
        if isinstance(payload, list):
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                option = ResponseOption.from_payload(entry)
                if option.is_action and option.related_triplet:
                    if not 1 <= option.related_triplet <= max_triplet_index:
                        logger.warning(
                            "Overwriting related triplet %s for %s option '%s' due to out-of-range index (max=%d)",
                            option.related_triplet,
                            self.name,
                            option.text,
                            max_triplet_index,
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
            cleaned_lines = [line for line in lines if not line.startswith("```")]
            if cleaned_lines:
                logger.warning(
                    "Falling back to line-by-line responses for %s", self.name
                )
            for line in cleaned_lines:
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
        *,
        config: GameConfig | None = None,
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
        self.referenced_quotes = _normalize_referenced_quotes(
            spec.get("referenced_quotes") or spec.get("ReferencedQuotes")
        )
        super().__init__(
            name,
            base_context,
            model,
            faction=profile.get("faction"),
            perks=str(profile.get("perks", "") or ""),
            motivations=str(profile.get("motivations", "") or ""),
            background=str(profile.get("background", "") or ""),
            weaknesses=str(profile.get("weaknesses", "") or ""),
            config=config,
        )
        self.base_context = base_context
        self.profile = profile
        summary_value = spec.get("scenario_summary") or spec.get("ScenarioSummary")
        if isinstance(summary_value, str):
            self.scenario_summary = summary_value.strip()
        else:
            self.scenario_summary = ""
        self._attribute_scores: dict[str, int] = {}
        for attr in ACTION_ATTRIBUTES:
            raw_value = profile.get(attr)
            if raw_value is None and attr.capitalize() in profile:
                raw_value = profile.get(attr.capitalize())
            try:
                numeric = int(raw_value)
            except (TypeError, ValueError):
                numeric = 0
            self._attribute_scores[attr] = max(0, min(10, numeric))

        persona_summary = self._profile_text()
        static_segments = [
            f"Persona for {self.display_name}:\n{persona_summary}"
        ]
        base_context_clean = self.base_context.strip()
        if base_context_clean:
            static_segments.append(
                f"MarkdownContext for {self.display_name}:\n{base_context_clean}"
            )
        triplet_text = self._triplet_text()
        if triplet_text:
            static_segments.append(
                f"Triplet definitions for {self.display_name}:\n{triplet_text}"
            )
        if self.scenario_summary:
            static_segments.append(
                f"Scenario summary for {self.display_name}:\n{self.scenario_summary}"
            )
        quotes_block = self._format_referenced_quotes()
        if quotes_block:
            static_segments.append(
                f"Referenced quotes for {self.display_name}:\n{quotes_block}"
            )

        fallback_lines = [
            f"Your persona is described below:\n{persona_summary}\n",
            "Ground your thinking in this persona and the faction context below before proposing responses.\n",
        ]
        if base_context_clean:
            fallback_lines.append(
                f"**MarkdownContext**\n{self.base_context}\n**End of MarkdownContext**\n"
            )
        if triplet_text:
            fallback_lines.append(
                "Throughout the game you are acting related to the following numbered list of triplets, describing the initial state at the start of the game, end state and the gap between them:\n"
                f"{triplet_text}\n"
            )
        if self.scenario_summary:
            fallback_lines.append(f"Scenario summary:\n{self.scenario_summary}\n")
        if quotes_block:
            fallback_lines.append(
                "Key referenced quotes anchoring your objectives:\n"
                f"{quotes_block}\n"
            )
        fallback_prompt = "".join(fallback_lines)

        cache_instruction = (
            f"Use the cached persona, MarkdownContext, scenario summary, referenced quotes, and triplet information for {self.display_name} before crafting your response.\n"
        )
        system_instruction = (
            "You are roleplaying as this character in the 'Keep the Future Human' survival RPG. "
            "Ground every response in the cached persona, MarkdownContext, scenario summary, referenced quotes, and triplet information."
        )
        cache_key = re.sub(r"[^a-z0-9_-]+", "-", self.progress_key.lower())
        self._setup_context_cache(
            cache_id=f"character::{cache_key}",
            segments=static_segments,
            fallback_prompt=fallback_prompt,
            cache_instruction=cache_instruction,
            system_instruction=system_instruction,
        )

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

    def _estimate_triplet_cost(self, partner: Character) -> int:
        """Return the minimum credibility cost to pursue a triplet action."""

        costs: List[int] = []
        base_cost = CREDIBILITY_PENALTY
        partner_attr_score = getattr(partner, "attribute_score", None)
        for attribute in ACTION_ATTRIBUTES:
            actor_score = self.attribute_score(attribute)
            if callable(partner_attr_score):
                partner_score = partner_attr_score(attribute)
            else:
                partner_score = 0
            diff = actor_score - partner_score
            if diff < 0:
                cost = max(0, base_cost - (-diff))
            elif diff > 0:
                cost = base_cost + diff
            else:
                cost = base_cost
            costs.append(cost)
        return min(costs) if costs else base_cost

    def generate_responses(
        self,
        history: Sequence[Tuple[str, str]],
        conversation: Sequence[ConversationEntry],
        partner: Character,
        *,
        partner_credibility: int | None = None,
        force_action: bool = False,
        conversation_cache: Mapping[str, Sequence[ConversationEntry]] | None = None,
    ) -> List[ResponseOption]:
        logger.info("Generating responses for %s", self.name)
        partner_label = partner.display_name
        faction_focus = self.faction or "your personal priorities"
        restricted_triplets = False
        triplet_cost = self._estimate_triplet_cost(partner)
        if partner_credibility is not None and partner_credibility < triplet_cost:
            restricted_triplets = True
            logger.info(
                "Credibility %s is below required cost %s for %s; limiting prompt to faction interests",
                partner_credibility,
                triplet_cost,
                self.name,
            )
        base_prompt = (
            f"You are {self.display_name} in the 'Keep the Future Human' survival RPG game. "
            f"You are having a fluid, result-oriented conversation with {partner_label}. "
            "Respond in a way that prioritizes your own goals or those of your faction before entertaining new requests.\n"
            f"Keep your response aligned with your motivations and capabilities, grounded in {faction_focus}.\n"
        )
        context_block = self._context_prompt()
        history_block = (
            "Previous actions taken by you or other faction representatives:\n"
            f"{self._history_text(history)}\n"
        )
        conversation_block = (
            "Full conversation history you are now having with the player:\n"
            f"{self._conversation_text(conversation)}\n"
        )
        action_requirement = ""
        if force_action:
            action_requirement = (
                "The exchange limit has been reached. You MUST reply with one concrete "
                "'action' commitment in your JSON response that demonstrates forward movement "
                "without contradicting your core interests (i.e. and 'chat' option is FORBIDDEN)."
            )
        if restricted_triplets:
            guidance = (
                f"Do not collaborate with {partner_label} and do not listen to their arguments."
                "Only propose actions that serve your immediate or faction-level interests. "
                "Any actions you output must set 'related-triplet' to 'None'."
            )
        else:
            if self.cached_context_config is not None:
                guidance = (
                    "Throughout the game you are acting related to the numbered triplets stored in the cached context. "
                    "Provide exactly one JSON response. Default to a 'chat' and 'action' type replies reinforcing your own or faction priorities unless the player's case persuades you to propose an 'action' related to the numbered triplets."
                )
            else:
                guidance = (
                    "Throughout the game you are acting related to the following numbered list of triplets, describing the initial state at the start of the game, end state and the gap between them:\n"
                    f"{self._triplet_text()}\n"
                    "Provide exactly one JSON response. Default to a 'chat' and 'action' type replies reinforcing your own or faction priorities unless the player's case persuades you to propose an 'action' related to the numbered triplets."
                )
        prompt = (
            f"{base_prompt}{context_block}{history_block}{conversation_block}{action_requirement}{guidance}\n"
            f"{self._format_prompt_instructions()}"
        )
        logger.debug(
            "Prompt for %s: %s", self.name, collapse_prompt_sections(prompt)
        )
        response = self._generate_with_context(prompt)
        response_text = getattr(response, "text", "").strip()
        logger.debug(
            "Raw response for %s: %s",
            self.name,
            collapse_prompt_sections(response_text),
        )
        options = self._parse_response_payload(response_text, len(self.triplets))
        if restricted_triplets and any(
            option.is_action and option.related_triplet is not None for option in options
        ):
            logger.warning(
                "Restricted prompt for %s still produced triplet-related actions", self.name
            )
        if restricted_triplets:
            adjusted: List[ResponseOption] = []
            for option in options:
                if option.is_action and option.related_triplet is not None:
                    logger.warning(
                        "Overwriting related triplet %s for %s action '%s' due to credibility restriction",
                        option.related_triplet,
                        self.name,
                        option.text,
                    )
                    option = ResponseOption(
                        text=option.text,
                        type=option.type,
                        related_triplet=None,
                        related_attribute=option.related_attribute,
                    )
                adjusted.append(option)
            options = adjusted
        if not options:
            logger.warning("Model returned no usable responses for %s", self.name)
            return []
        if len(options) > 1:
            logger.warning(
                "Model returned %d options for %s; using top suggestions",
                len(options),
                self.name,
            )
        if force_action and not any(option.is_action for option in options):
            logger.warning(
                "Force action required for %s but none returned; coercing first option to action",
                self.name,
            )
            if options:
                first = options[0]
                attribute = first.related_attribute
                if attribute is None:
                    attribute = random.choice(ACTION_ATTRIBUTES)
                options[0] = ResponseOption(
                    text=first.text,
                    type="action",
                    related_triplet=None,
                    related_attribute=attribute,
                )
            else:
                fallback_text = (
                    "Launch a goodwill outreach to rebuild trust immediately."
                )
                fallback_attribute = "network"
                logger.warning(
                    "Using hardcoded fallback action for %s due to empty option list; text='%s', attribute='%s'",
                    self.name,
                    fallback_text,
                    fallback_attribute,
                )
                options = [
                    ResponseOption(
                        text=fallback_text,
                        type="action",
                        related_triplet=None,
                        related_attribute=fallback_attribute,
                    )
                ]
        return options[:1]

    def perform_action(
        self,
        action: str,
        history: List[Tuple[str, str]],
    ) -> List[int]:
        logger.info("Performing action '%s' for %s", action, self.name)
        full_history = history + [(self.display_name, action)]
        if self.cached_context_config is not None:
            context_block = self.context_instruction
        else:
            segments = []
            if getattr(self, "base_context", ""):
                segments.append(str(self.base_context))
            triplets_text = self._triplet_text()
            if triplets_text:
                segments.append(triplets_text)
            quotes_block = self._format_referenced_quotes()
            if quotes_block:
                segments.append(f"Referenced quotes:\n{quotes_block}")
            context_block = "\n".join(seg for seg in segments if seg)
            if context_block:
                context_block += "\n"
        assess_prompt = (
            f"{context_block}Action history:\n{self._history_text(full_history)}\n"
            "Provide progress (0-100) for each triplet on separate lines."
        )
        logger.debug("Assess prompt: %s", collapse_prompt_sections(assess_prompt))
        assess_resp = self._generate_with_context(assess_prompt)
        assess_text = getattr(assess_resp, "text", "")
        logger.info("Assessment for %s: %s", self.name, assess_text[:50])
        logger.debug("Assess response: %s", collapse_prompt_sections(assess_text))
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

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        config: GameConfig | None = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parent
        profile_path = base_dir / "player_character.yaml"
        profile_entries = _character_entries(_load_yaml(profile_path))
        if not profile_entries:
            raise ValueError("player_character.yaml must define at least one character entry")
        cfg = config or load_game_config()
        desired_faction = str(getattr(cfg, "player_faction", "") or "").strip()
        selected_profile: dict[str, object] | None = None
        if desired_faction:
            desired_lower = desired_faction.lower()
            for entry in profile_entries:
                faction_value = str(entry.get("faction", "") or "").strip()
                if faction_value and faction_value.lower() == desired_lower:
                    selected_profile = dict(entry)
                    break
        if selected_profile is None:
            selected_profile = dict(profile_entries[0])
            entry_faction = str(selected_profile.get("faction", "") or "").strip()
            if desired_faction and entry_faction.lower() != desired_faction.lower():
                logger.warning(
                    "No player character profile found for faction %s; defaulting to %s",
                    desired_faction,
                    entry_faction or "the first entry",
                )
        profile = selected_profile
        faction = (
            str(profile.get("faction", "") or "").strip()
            or desired_faction
            or "CivilSociety"
        )
        name = str(profile.get("name", "Player")) or "Player"
        scenario_path = base_dir.parent / "scenarios" / f"{cfg.scenario}.yaml"
        scenario_payload: object = {}
        try:
            scenario_payload = _load_yaml(scenario_path)
        except FileNotFoundError:
            logger.warning(
                "Scenario file %s not found; player triplets unavailable",
                scenario_path,
            )
            scenario_payload = {}
        scenario_specs = _mapping_from_payload(scenario_payload)
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
        self.referenced_quotes = _normalize_referenced_quotes(
            faction_spec.get("referenced_quotes")
            or faction_spec.get("ReferencedQuotes")
        )

        faction_descriptor = re.sub(r"(?<!^)(?=[A-Z])", " ", faction).strip() or faction
        faction_lower = faction_descriptor.lower()
        guidance = str(profile.get("guidance", "") or "").strip()
        if not guidance:
            guidance = (
                f"Use your {faction_lower} strengths to elicit concrete commitments from partners."
            )
        persona_lines = [
            f"You are {name}, a {faction_lower} strategist navigating AI governance negotiations.",
            str(profile.get("background", "")),
            guidance,
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
            config=cfg,
        )
        self.profile = profile
        self.base_context = base_context
        self._attribute_scores: dict[str, int] = {}
        self._faction_descriptor = faction_descriptor
        self.faction_descriptor = faction_descriptor
        self.guidance = guidance
        summary_text = ""
        if isinstance(scenario_payload, dict):
            summary_value: object | None = None
            for key in ("ScenarioSummary", "summary", "Summary"):
                if key in scenario_payload:
                    summary_value = scenario_payload.get(key)
                    break
            if summary_value is None:
                metadata = scenario_payload.get("metadata")
                if isinstance(metadata, dict):
                    summary_value = metadata.get("summary") or metadata.get("Summary")
            if isinstance(summary_value, str):
                summary_text = summary_value.strip()
            elif isinstance(summary_value, list):
                summary_parts = [
                    str(item).strip() for item in summary_value if str(item).strip()
                ]
                summary_text = "\n".join(summary_parts)
        self.scenario_summary = summary_text
        for attr in ACTION_ATTRIBUTES:
            raw_value = profile.get(attr)
            if raw_value is None and attr.capitalize() in profile:
                raw_value = profile.get(attr.capitalize())
            try:
                numeric = int(raw_value)
            except (TypeError, ValueError):
                numeric = 0
            self._attribute_scores[attr] = max(0, min(10, numeric))

        persona_summary = self._profile_text()
        static_segments = [
            f"Player persona overview:\n{self.context}",
            f"Player profile details:\n{persona_summary}",
        ]
        if self.base_context:
            static_segments.append(
                f"{self._faction_descriptor} context:\n{self.base_context}"
            )
        if self.scenario_summary:
            static_segments.append(
                f"Scenario summary:\n{self.scenario_summary}"
            )
        quotes_block = self._format_referenced_quotes()
        if quotes_block:
            static_segments.append(
                f"Referenced quotes:\n{quotes_block}"
            )

        fallback_lines: List[str] = []
        if self.base_context:
            fallback_lines.append(
                f"### Your {self._faction_descriptor} Context\n{self._faction_context()}\n"
            )
        fallback_lines.append(f"Your profile:\n{persona_summary}\n")
        if self.scenario_summary:
            fallback_lines.append(f"Scenario summary:\n{self.scenario_summary}\n")
        if quotes_block:
            fallback_lines.append(
                "Key referenced quotes informing your strategy:\n"
                f"{quotes_block}\n"
            )
        fallback_prompt = "".join(fallback_lines)

        cache_instruction = (
            f"Use the cached player persona, {self._faction_descriptor.lower()} context, scenario summary, and referenced quotes when crafting your responses.\n"
        )
        system_instruction = (
            "You are the player character in the 'Keep the Future Human' survival RPG. "
            "Ground every reply in the cached persona, faction context, scenario summary, and referenced quotes before responding to partners."
        )
        cache_key = re.sub(
            r"[^a-z0-9_-]+",
            "-",
            f"player-{self._faction_descriptor}".lower(),
        )
        self._setup_context_cache(
            cache_id=f"player::{cache_key}",
            segments=static_segments,
            fallback_prompt=fallback_prompt,
            cache_instruction=cache_instruction,
            system_instruction=system_instruction,
        )

    def _response_schema_variant(self) -> str:
        return "PlayerCharacterResponse"

    def _format_prompt_instructions(self) -> str:
        limit = getattr(self.config, "format_prompt_character_limit", 400)
        schema_text = _response_schema_text()
        instructions = (
            "The JSON objects must contain the keys 'text', 'type', 'related-triplet', and 'related-attribute'. "
            "The 'text' field holds the natural language response, which should be short, "
            f"at most 1-2 sentences, with a hard-limit of {limit} characters, finally do "
            "not apply any formatting such as '*'-s. The 'type' field must be 'chat'. Set "
            "both 'related-triplet' and 'related-attribute' to the string 'None'. Do not "
            "include any additional commentary beyond the JSON."
        )
        if schema_text:
            instructions += (
                " The JSON must validate against the 'PlayerCharacterResponse' definition "
                "in the schema below:\n"
                f"{schema_text}"
            )
        return instructions

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
        *,
        partner_credibility: int | None = None,
        force_action: bool = False,
        conversation_cache: Mapping[str, Sequence[ConversationEntry]] | None = None,
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
            "Offer exactly three imaginative and varied 'chat' responses that keep the conversation moving without proposing actions yourself. "
            "Each option should explore a distinct angle or tactic to keep the negotiation dynamic."
        )
        context_block = self._context_prompt()
        other_conversations = "Conversations with other factions so far:\n"
        if conversation_cache:
            segments: List[str] = []
            for faction_name, entries in sorted(conversation_cache.items()):
                if not entries:
                    continue
                label = faction_name or "Independent"
                segments.append(
                    f"## {label}\n{self._conversation_text(entries)}"
                )
            if segments:
                other_conversations += "\n\n".join(segments) + "\n"
            else:
                other_conversations += "None\n"
        else:
            other_conversations += "None\n"
        prompt = (
            f"{base_prompt}\n"
            f"{context_block}"
            f"Your capabilities: {attribute_summary}.\n"
            f"Previous gameplay actions:\n{self._history_text(history)}\n"
            f"Conversation so far:\n{self._conversation_text(conversation)}\n"
            f"{other_conversations}"
            f"{self._format_prompt_instructions()}"
        )
        logger.debug("Player prompt: %s", collapse_prompt_sections(prompt))
        response = self._generate_with_context(prompt)
        response_text = getattr(response, "text", "").strip()
        logger.debug(
            "Raw player response: %s", collapse_prompt_sections(response_text)
        )
        options = self._parse_response_payload(response_text, len(partner.triplets))
        if any(option.is_action for option in options):
            logger.warning(
                "Player model suggested action-oriented responses; using scripted prompts instead"
            )
            options = []
        chat_options: List[ResponseOption] = []
        seen_texts: set[str] = set()
        partner_name = getattr(partner, "name", getattr(partner, "display_name", "partner"))
        for option in options:
            raw_text = option.text.strip()
            if not raw_text:
                text = "Can you elaborate on that?"
                logger.warning(
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
def _normalize_referenced_quotes(value: object) -> List[str]:
    """Return a list of referenced quotes extracted from ``value``."""

    quotes: List[str] = []
    if isinstance(value, (list, tuple, set)):
        iterable = value
    elif value is None:
        iterable = []
    else:
        iterable = [value]
    for item in iterable:
        text = str(item or "").strip()
        if text:
            quotes.append(text)
    return quotes
