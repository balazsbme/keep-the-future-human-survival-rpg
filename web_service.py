"""Flask web service exposing the RPG demo over HTTP."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
import logging
import os
import re
import threading
from urllib.parse import quote
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from functools import lru_cache
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from flask import Flask, Response, redirect, request, send_from_directory
from dotenv import load_dotenv
import google.generativeai as genai
import yaml

from cli_game import load_characters
from rpg.assessment_agent import AssessmentAgent
from rpg.character import Character, ResponseOption
from rpg.config import GameConfig, load_game_config
from rpg.conversation import ConversationEntry
from rpg.game_state import ActionAttempt, GameState


GITHUB_URL = "https://github.com/balazsbme/keep-the-future-human-survival-rpg"


WEB_RESOURCES_DIR = Path(__file__).resolve().parent / "web"
SNIPPET_DIR = WEB_RESOURCES_DIR / "snippets"
STYLE_BASENAME = "style.css"
TOOLTIP_CONFIG_PATH = WEB_RESOURCES_DIR / "tooltips.yaml"

DEFAULT_ATTRIBUTE_TOOLTIPS = {
    "leadership": "Leadership captures how well you coordinate allies and keep coalitions focused on shared commitments.",
    "technology": "Technology reflects your technical literacy for evaluating safeguards and translating expert findings.",
    "policy": "Policy measures your ability to craft enforceable agreements and navigate governance trade-offs.",
    "network": "Network gauges access to relationships that unlock cooperation and surface new opportunities.",
}

DEFAULT_CREDIBILITY_TOOLTIP = (
    "Credibility measures how much latitude this faction gives you. "
    "High scores lower reroll costs and keep triplet-aligned commitments on the table. "
    "When credibility drops below the triplet cost, the faction stops collaborating and only pushes its own agenda."
)

DEFAULT_CONFIG_FIELD_HELP = {
    "scenario": "Select the narrative scenario used for free play sessions.",
    "win_threshold": "Score needed to achieve a win at the end of the run.",
    "max_rounds": "Maximum number of conversation rounds before the game ends.",
    "roll_success_threshold": "Minimum roll total required for an action to succeed.",
    "action_time_cost_years": "Years of in-game time that pass whenever you attempt an action.",
    "format_prompt_character_limit": "Maximum characters allowed when prompts are formatted for the model.",
    "conversation_force_action_after": "Force an action to be offered after this many exchanges without one.",
    "enabled_factions": "Factions that can appear in free play encounters.",
    "player_faction": "Faction alignment assigned to your player character.",
}


logger = logging.getLogger(__name__)


PLAYER_PERSONA_PATH = Path(__file__).resolve().parent / "rpg" / "player_character.yaml"
FACTION_IMAGE_MAP: dict[str, str] = {
    "governments": "gov-1-neutral.jpg",
    "corporations": "corp-1-neutral.jpg",
    "hardwaremanufacturers": "hwman-1-neutral.jpg",
    "regulators": "reg-1-neutral.jpg",
    "civilsociety": "civ-1-neutral.jpg",
    "scientificcommunity": "sci-1-neutral.jpg",
}


def _normalize_key(value: str | None) -> str:
    """Return a lowercase slug used for persona lookups."""

    if not value:
        return ""
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned


def _load_player_personas() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load player persona definitions from ``player_character.yaml``."""

    try:
        with open(PLAYER_PERSONA_PATH, "r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        logger.warning("Player persona definition missing at %s", PLAYER_PERSONA_PATH)
        return {}, {}
    entries = payload.get("Characters") if isinstance(payload, Mapping) else None
    if not isinstance(entries, Sequence):
        return {}, {}
    by_faction: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        profile = {str(key): value for key, value in entry.items()}
        faction_key = _normalize_key(str(profile.get("faction", "") or ""))
        name_key = _normalize_key(str(profile.get("name", "") or ""))
        if faction_key:
            by_faction[faction_key] = profile
        if name_key:
            by_name[name_key] = profile
    return by_faction, by_name


player_personas_by_faction, player_personas_by_name = _load_player_personas()


def _player_profile_by_faction(faction: str | None) -> dict[str, Any] | None:
    """Return a persona profile for the provided faction, if available."""

    normalized = _normalize_key(faction or "")
    if not normalized:
        return None
    profile = player_personas_by_faction.get(normalized)
    return dict(profile) if profile else None


def _player_persona_path(key: str) -> str:
    """Return the routed URL for a persona profile."""

    normalized = _normalize_key(key)
    if not normalized:
        return "/player/profile"
    return f"/player/personas/{normalized}"


def _format_faction_label(name: str | None) -> str:
    """Return a human readable faction label."""

    cleaned = (name or "").strip()
    if not cleaned:
        return ""
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", cleaned).strip()
    return spaced or cleaned


def _profile_image_html(
    name: str,
    *,
    css_class: str = "profile-photo",
    alt_label: str | None = None,
    faction: str | None = None,
) -> str:
    """Return HTML for a profile photo placeholder or image."""

    alt_text = alt_label or f"Portrait of {name or 'the player persona'}"
    filename = FACTION_IMAGE_MAP.get(_normalize_key(faction or "")) if faction else None
    if filename:
        src = f"/assets/character-pictures/{filename}"
        return (
            f"<div class='{css_class}'>"
            + f"<img src='{escape(src, quote=True)}' alt='{escape(alt_text, quote=True)}'>"
            + "</div>"
        )
    initials = "".join(part[0].upper() for part in str(name).split() if part)
    initials = initials[:2] or "KTFH"
    return f"<div class='{css_class}'>{escape(initials, False)}</div>"


def _persona_attribute_items(
    lookup: Mapping[str, Any] | Character,
    value_getter: Callable[[Mapping[str, Any] | Character, str], int],
) -> list[str]:
    items: list[str] = []
    for label in ("Leadership", "Technology", "Policy", "Network"):
        key = label.lower()
        score = value_getter(lookup, key)
        items.append(
            "<li>"
            + f"<span class='attribute-label'>{escape(label, False)}</span>"
            + f"<span class='attribute-value'>{int(score)}</span>"
            + "</li>"
        )
    return items


def _persona_card_from_profile(profile: Mapping[str, Any]) -> str:
    """Return persona markup for a raw profile mapping."""

    name = str(profile.get("name", "Player Persona") or "Player Persona")
    faction = _format_faction_label(str(profile.get("faction", "")))
    guidance = str(profile.get("guidance", "") or "").strip()
    header = (
        "<div class='persona-header'>"
        + _profile_image_html(name, css_class="persona-photo", faction=str(profile.get("faction", "")))
        + "<div>"
        + f"<h3>{escape(name, False)}</h3>"
        + (f"<p class='persona-faction'>{escape(faction, False)}</p>" if faction else "")
        + (f"<p class='persona-guidance'>{escape(guidance, False)}</p>" if guidance else "")
        + "</div></div>"
    )

    def _score_lookup(_: Mapping[str, Any], key: str) -> int:
        raw = profile.get(key) or profile.get(key.capitalize())
        try:
            return max(0, min(10, int(raw)))
        except (TypeError, ValueError):
            return 0

    attributes = _persona_attribute_items(profile, _score_lookup)
    attribute_block = (
        "<ul class='persona-attributes'>" + "".join(attributes) + "</ul>"
    )
    body_parts: list[str] = []
    for label, key in (
        ("Background", "background"),
        ("Perks", "perks"),
        ("Motivations", "motivations"),
        ("Weaknesses", "weaknesses"),
    ):
        text = str(profile.get(key, "") or "").strip()
        if text:
            body_parts.append(
                f"<p><strong>{escape(label, False)}:</strong> {escape(text, False)}</p>"
            )
    body = "<div class='persona-body'>" + "".join(body_parts) + "</div>"
    return "<article class='persona-card'>" + header + attribute_block + body + "</article>"


def _persona_card_for_character(character: Character) -> str:
    """Return persona markup for a :class:`Character`."""

    name = getattr(character, "display_name", getattr(character, "name", ""))
    faction = _format_faction_label(getattr(character, "faction", ""))
    guidance = getattr(character, "guidance", "") or getattr(character, "motivations", "")

    def _character_score(_: Character, key: str) -> int:
        return int(character.attribute_score(key))

    header = (
        "<div class='persona-header'>"
        + _profile_image_html(name, css_class="persona-photo", faction=getattr(character, "faction", ""))
        + "<div>"
        + f"<h3>{escape(name, False)}</h3>"
        + (f"<p class='persona-faction'>{escape(faction, False)}</p>" if faction else "")
        + (f"<p class='persona-guidance'>{escape(str(guidance), False)}</p>" if guidance else "")
        + "</div></div>"
    )
    attributes = _persona_attribute_items(character, _character_score)
    attribute_block = "<ul class='persona-attributes'>" + "".join(attributes) + "</ul>"
    body_parts: list[str] = []
    for label, attr_name in (
        ("Background", "background"),
        ("Perks", "perks"),
        ("Motivations", "motivations"),
        ("Weaknesses", "weaknesses"),
    ):
        value = getattr(character, attr_name, "")
        if value:
            body_parts.append(
                f"<p><strong>{escape(label, False)}:</strong> {escape(str(value), False)}</p>"
            )
    body = "<div class='persona-body'>" + "".join(body_parts) + "</div>"
    return "<article class='persona-card'>" + header + attribute_block + body + "</article>"


def _sector_preview_block(
    profile: Mapping[str, Any],
    *,
    label: str,
    profile_url: str | None = None,
) -> str:
    """Render the compact persona preview used on the campaign sector cards."""

    name = str(profile.get("name", "Player Persona") or "Player Persona")
    snippet = str(profile.get("guidance", "") or "").strip()
    if not snippet:
        snippet = str(profile.get("background", "") or "").strip()
    if len(snippet) > 140:
        snippet = snippet[:137].rstrip() + "\u2026"
    photo = _profile_image_html(
        name,
        css_class="preview-photo",
        faction=str(profile.get("faction", "") or label),
        alt_label=f"Portrait of {name}",
    )
    link_html = (
        f"<a href='{escape(profile_url, quote=True)}'>View full profile</a>"
        if profile_url
        else ""
    )
    body_parts = [f"<p class='preview-name'>{escape(name, False)}</p>"]
    if snippet:
        body_parts.append(f"<p>{escape(snippet, False)}</p>")
    body_parts.append(link_html)
    return (
        "<div class='sector-player-preview'>"
        + photo
        + "<div class='sector-player-preview-text'>"
        + "".join(part for part in body_parts if part)
        + "</div></div>"
    )


@lru_cache(maxsize=None)
def _load_snippet(snippet_name: str) -> str:
    """Return the snippet text stored under ``snippet_name``."""

    path = SNIPPET_DIR / snippet_name
    if not path.exists():
        raise FileNotFoundError(f"Missing snippet file: {snippet_name}")
    return path.read_text(encoding="utf-8")


def _load_tooltip_texts() -> tuple[dict[str, str], str, dict[str, str]]:
    """Load tooltip strings from ``tooltips.yaml`` (with defaults)."""

    try:
        with open(TOOLTIP_CONFIG_PATH, "r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        payload = {}
    attribute_payload = payload.get("attribute_tooltips") if isinstance(payload, dict) else None
    config_payload = payload.get("config_field_help") if isinstance(payload, dict) else None
    credibility_payload = payload.get("credibility") if isinstance(payload, dict) else None

    attribute_tooltips = dict(DEFAULT_ATTRIBUTE_TOOLTIPS)
    if isinstance(attribute_payload, Mapping):
        for key, value in attribute_payload.items():
            if value is not None:
                attribute_tooltips[str(key)] = str(value)

    config_tooltips = dict(DEFAULT_CONFIG_FIELD_HELP)
    if isinstance(config_payload, Mapping):
        for key, value in config_payload.items():
            if value is not None:
                config_tooltips[str(key)] = str(value)

    credibility_text = DEFAULT_CREDIBILITY_TOOLTIP
    if isinstance(credibility_payload, str):
        credibility_text = credibility_payload

    return attribute_tooltips, credibility_text, config_tooltips


load_dotenv()


def _configure_gemini_client() -> None:
    """Configure the Gemini SDK once so downstream imports can reuse it."""

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set. "
            "Set it (e.g., in .env) before starting the web service."
        )
    genai.configure(api_key=api_key)


_configure_gemini_client()


# Expose the active configuration at module scope so tests can patch it.
current_config: GameConfig = load_game_config()


PRIVATE_SECTOR_FACTIONS: tuple[str, ...] = (
    "Corporations",
    "HardwareManufacturers",
    "ScientificCommunity",
)
PUBLIC_SECTOR_FACTIONS: tuple[str, ...] = (
    "Governments",
    "Regulators",
    "CivilSociety",
)
CAMPAIGN_SCENARIOS: tuple[str, ...] = (
    "01-race-to-contain-power",
    "02-building-the-gates",
    "03-keep-the-future-human",
)
FREE_PLAY_HIDDEN_SCENARIOS: set[str] = {"complete"}


@dataclass
class CampaignState:
    """Track the player's progress through the structured campaign."""

    active: bool = False
    current_level: int = 0
    sector_choice: str | None = None
    level_outcomes: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    def reset(self) -> None:
        self.active = True
        self.current_level = 0
        self.sector_choice = None
        self.level_outcomes.clear()


def _option_from_payload(raw: str) -> ResponseOption:
    """Return a :class:`ResponseOption` parsed from ``raw`` JSON."""

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.debug("Falling back to chat option for raw payload: %s", raw)
        return ResponseOption(text=str(raw), type="chat")
    if isinstance(payload, dict):
        return ResponseOption.from_payload(payload)
    return ResponseOption(text=str(payload), type="chat")


def create_app() -> Flask:
    """Return a configured Flask application ready to serve the game."""

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    app = Flask(__name__)
    global current_config
    config_in_use = current_config
    campaign_state = CampaignState()
    current_mode = "free_play"
    scenario_roots = [
        Path(__file__).resolve().parent / "scenarios",
        Path(__file__).resolve().parent / "rpg" / "scenarios",
    ]
    discovered_scenarios = set()
    for root in scenario_roots:
        if root.exists():
            discovered_scenarios.update(p.stem.lower() for p in root.glob("*.yaml"))
    if not discovered_scenarios:
        available_scenarios = [config_in_use.scenario]
    else:
        discovered_scenarios.add(config_in_use.scenario)
        available_scenarios = sorted(discovered_scenarios)
    faction_path = Path(__file__).resolve().parent / "factions.yaml"
    try:
        with open(faction_path, "r", encoding="utf-8") as fh:
            faction_payload = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        faction_payload = {}
    if isinstance(faction_payload, dict):
        known_factions = sorted(str(name) for name in faction_payload.keys())
    else:
        known_factions = []
    if not known_factions:
        fallback = sorted(set(PRIVATE_SECTOR_FACTIONS + PUBLIC_SECTOR_FACTIONS))
        known_factions = fallback
    initial_characters = load_characters(config=config_in_use)
    game_state = GameState(list(initial_characters), config_override=config_in_use)
    assessor = AssessmentAgent()
    enable_parallel = os.environ.get("ENABLE_PARALLELISM") == "1"
    pending_player_options: Dict[
        Tuple[int, int], Tuple[threading.Event, List[ResponseOption] | None]
    ] = {}
    pending_npc_responses: Dict[
        Tuple[int, int, str], Tuple[threading.Event, List[ResponseOption] | None]
    ] = {}
    pending_player_choices: Dict[int, Tuple[int, str, ResponseOption]] = {}
    player_option_signatures: Dict[Tuple[int, int], tuple] = {}
    last_history_signature: Tuple[Tuple[str, str], ...] | None = None
    assessment_threads: List[threading.Thread] = []
    assessment_lock = threading.Lock()
    state_lock = threading.Lock()
    attribute_tooltips, credibility_tooltip_text, config_field_help_texts = _load_tooltip_texts()

    asset_root = Path(__file__).resolve().parent / "assets"

    @app.route("/assets/<path:filename>")
    def serve_asset(filename: str) -> Response:
        return send_from_directory(asset_root, filename)

    @app.route("/web/style.css")
    def serve_web_style() -> Response:
        return send_from_directory(WEB_RESOURCES_DIR, STYLE_BASENAME)

    tooltip_asset_path = "/assets/tooltip.jpg"
    roll_asset_path = "/assets/rolling.gif"
    footer_html = _load_snippet("global_footer.html").format(github_url=GITHUB_URL)
    roll_indicator_markup = _load_snippet("roll_indicator.html").format(
        roll_asset_path=roll_asset_path
    )
    roll_indicator_script = _load_snippet("roll_indicator.js")
    state_refresh_script = _load_snippet("state_refresh.js")
    css_link_tag = "<link rel='stylesheet' href='/web/style.css'>"

    def _render_page(
        content: str,
        *,
        body_class: str = "page-default",
        include_footer: bool = True,
        extra_scripts: Sequence[str] | None = None,
    ) -> str:
        parts: List[str] = [css_link_tag, f"<body class='{body_class}'>", content]
        if include_footer:
            parts.append(footer_html)
        if extra_scripts:
            parts.extend(extra_scripts)
        parts.append("</body>")
        return "".join(parts)

    def _tooltip_icon(description: str) -> str:
        content = escape(description, False)
        return (
            "<span class='tooltip-icon'>"
            + f"<img src='{tooltip_asset_path}' alt='Info tooltip icon'>"
            + f"<span class='tooltip-text'>{content}</span>"
            + "</span>"
        )

    def _config_label(title: str, description: str) -> str:
        return (
            f"<span class='config-label-text'>{escape(title, False)}"
            + _tooltip_icon(description)
            + "</span>"
        )

    def _loading_markup(text: str) -> str:
        safe_text = escape(text, False)
        return (
            "<div class='inline-loading'>"
            + f"<img src='{roll_asset_path}' alt='Loading indicator'>"
            + f"<span>{safe_text}</span>"
            + "</div>"
        )

    def _scenario_display_name(name: str) -> str:
        base = (name or "").replace("-", " ").replace("_", " ").strip()
        return base.title() if base else "Unknown Scenario"

    def _scenario_summary_text(name: str) -> str:
        for root in scenario_roots:
            candidate = root / f"{name}.yaml"
            if not candidate.exists():
                continue
            try:
                with open(candidate, "r", encoding="utf-8") as fh:
                    payload = yaml.safe_load(fh) or {}
            except (FileNotFoundError, yaml.YAMLError):
                continue
            if isinstance(payload, dict):
                summary = payload.get("ScenarioSummary")
                if isinstance(summary, str):
                    return summary
        return ""

    def _format_faction(name: str) -> str:
        cleaned = (name or "").strip()
        if not cleaned:
            return "Unknown"
        result: list[str] = []
        for idx, char in enumerate(cleaned):
            if idx > 0 and char.isupper() and not cleaned[idx - 1].isupper():
                result.append(" ")
            result.append(char)
        return "".join(result)

    def _campaign_config(level_index: int, sector: str) -> GameConfig:
        baseline = load_game_config()
        scenario = CAMPAIGN_SCENARIOS[level_index]
        if sector == "private":
            enabled = PRIVATE_SECTOR_FACTIONS
            player_faction = "CivilSociety"
        else:
            enabled = PUBLIC_SECTOR_FACTIONS
            player_faction = "ScientificCommunity"
        return GameConfig(
            scenario=scenario,
            win_threshold=baseline.win_threshold,
            max_rounds=baseline.max_rounds,
            roll_success_threshold=baseline.roll_success_threshold,
            action_time_cost_years=baseline.action_time_cost_years,
            format_prompt_character_limit=baseline.format_prompt_character_limit,
            conversation_force_action_after=baseline.conversation_force_action_after,
            enabled_factions=tuple(enabled),
            player_faction=player_faction,
        )

    def _reload_state(config: GameConfig) -> None:
        nonlocal config_in_use, initial_characters, game_state, last_history_signature
        global current_config
        logger.info("Reloading game state with config %s", config)
        config_in_use = config
        current_config = config
        try:
            new_characters = load_characters(config=config)
        except StopIteration:
            logger.warning(
                "Character loading interrupted by exhausted mock; reusing existing roster"
            )
            new_characters = list(initial_characters)
        except RuntimeError as exc:
            logger.error(
                "Failed to load characters with config %s: %s; reusing existing roster",
                config,
                exc,
            )
            new_characters = list(initial_characters)
        initial_characters = list(new_characters)
        existing_player = getattr(game_state, "player_character", None)
        try:
            new_state = GameState(list(initial_characters), config_override=config)
        except StopIteration:
            logger.warning(
                "Player character reset interrupted by exhausted mock; reusing existing persona"
            )
            new_state = GameState(
                list(initial_characters),
                config_override=config,
                player_override=existing_player,
            )
        except RuntimeError as exc:
            logger.error(
                "Failed to reset game state with config %s: %s; keeping previous state",
                config,
                exc,
            )
            return
        game_state = new_state
        pending_player_options.clear()
        player_option_signatures.clear()
        pending_npc_responses.clear()
        pending_player_choices.clear()
        with assessment_lock:
            assessment_threads.clear()
        last_history_signature = None

    def _format_summary_html(text: str) -> str:
        stripped = (text or "").strip()
        if not stripped:
            return ""
        paragraphs: List[str] = []
        blocks = [block.strip() for block in stripped.split("\n\n")]
        for block in blocks:
            if not block:
                continue
            safe_block = escape(block, quote=False).replace("\n", "<br>")
            paragraphs.append(f"<p>{safe_block}</p>")
        if not paragraphs:
            safe_text = escape(stripped, quote=False).replace("\n", "<br>")
            return f"<p>{safe_text}</p>"
        return "".join(paragraphs)

    def _scenario_summary_section(text: str) -> str:
        formatted = _format_summary_html(text)
        if not formatted:
            return ""
        return (
            "<section class='scenario-summary'><h2>Scenario Overview</h2>"
            + formatted
            + "</section>"
        )

    def _credibility_matrix_block(
        *,
        player_faction: str | None,
        partner_faction: str | None,
        credibility_value: int | None,
    ) -> str:
        if not partner_faction:
            return ""
        player_label = _format_faction_label(player_faction)
        partner_label = _format_faction_label(partner_faction)
        value_text = (
            "—" if credibility_value is None else f"{int(credibility_value)} / 100"
        )
        return (
            "<section class='credibility-matrix'>"
            + "<h2>Credibility Snapshot</h2>"
            + "<table><thead><tr><th>Player faction</th><th>Partner faction</th><th>Credibility"
            + _tooltip_icon(credibility_tooltip_text)
            + "</th></tr></thead><tbody>"
            + "<tr>"
            + f"<td>{escape(player_label or 'Unknown', False)}</td>"
            + f"<td>{escape(partner_label or 'Unknown', False)}</td>"
            + f"<td>{escape(value_text, False)}</td>"
            + "</tr></tbody></table>"
            + "<p class='credibility-note'>Values come from the credibility matrix that seeds each player–faction pairing. Higher credibility both reduces reroll costs and keeps triplet collaborations possible; falling below the required triplet cost forces partners to act only in their own interest.</p>"
            + "</section>"
        )

    def _conversation_log(conversation: Sequence[ConversationEntry]) -> str:
        if not conversation:
            return "<p class='empty-conversation'>No conversation yet. Start by greeting the character.</p>"
        items = []
        for entry in conversation:
            raw_type = str(entry.type or "dialogue").strip().lower()
            type_label = raw_type.replace("_", " ").title()
            type_hint = f"<em>({escape(raw_type, False)})</em>" if raw_type else ""
            items.append(
                "<li>"
                + "<div class='message-header'>"
                + f"<span class='speaker'>{escape(entry.speaker, False)}</span>"
                + f"<span class='message-type'>{escape(type_label, False)}</span>"
                + type_hint
                + "</div>"
                + f"<p>{escape(entry.text, False)}</p>"
                + "</li>"
            )
        return "<ul class='conversation-log'>" + "".join(items) + "</ul>"

    def _conversation_frame(
        player_panel_html: str,
        middle_panel_html: str,
        partner_panel_html: str,
        state_html: str,
    ) -> str:
        return (
            roll_indicator_markup
            + "<main class='conversation-page'>"
            + "<div class='layout-container'>"
            + f"<div class='panel player-panel'>{player_panel_html}</div>"
            + f"<div class='panel conversation-panel'>{middle_panel_html}</div>"
            + f"<div class='panel partner-panel'>{partner_panel_html}</div>"
            + "</div>"
            + f"<div class='state-container'>{state_html}</div>"
            + "</main>"
        )

    def _profile_panel(
        character: Character,
        *,
        credibility: int | None = None,
        profile_url: str | None = None,
    ) -> str:
        attribute_items: List[str] = []
        for label in ("Leadership", "Technology", "Policy", "Network"):
            key = label.lower()
            tooltip_text = attribute_tooltips.get(key)
            label_html = (
                f"<span class='attribute-label'>{escape(label, False)}"
                + (_tooltip_icon(tooltip_text) if tooltip_text else "")
                + "</span>"
            )
            value_html = f"<span class='attribute-value'>{int(character.attribute_score(key))}</span>"
            attribute_items.append(f"<li>{label_html}{value_html}</li>")
        attributes = "".join(attribute_items)
        credibility_block = ""
        if credibility is not None:
            label_html = (
                "<span class='attribute-label'>Credibility"
                + _tooltip_icon(credibility_tooltip_text)
                + "</span>"
            )
            credibility_block = (
                "<div class='credibility-box'>"
                f"{label_html}"
                f"<strong>{int(credibility)}</strong>"
                "</div>"
            )
        photo_block = _profile_image_html(
            getattr(character, "name", ""),
            css_class="profile-photo",
            alt_label=f"Portrait of {getattr(character, 'display_name', character.name)}",
        )
        footer_block = ""
        if profile_url:
            footer_block = (
                "<div class='profile-footer'>"
                + f"<a href='{escape(profile_url, quote=True)}'>View profile</a>"
                + "</div>"
            )
        return (
            "<div class='profile-card'>"
            f"{photo_block}"
            f"<h2 class='profile-name'>{escape(character.display_name, quote=False)}</h2>"
            f"<ul class='attribute-list'>{attributes}</ul>"
            f"{credibility_block}"
            f"{footer_block}"
            "</div>"
        )

    @app.before_request
    def log_request() -> None:
        logger.info("%s %s", request.method, request.path)

    @app.route("/", methods=["GET"])
    def main_page() -> str:
        current_summary = ""
        if campaign_state.active:
            level_index = min(campaign_state.current_level, len(CAMPAIGN_SCENARIOS) - 1)
            level_name = _scenario_display_name(CAMPAIGN_SCENARIOS[level_index])
            sector_note = (
                "Choose your next sector to begin."
                if campaign_state.sector_choice is None
                else (
                    "Working with the Public Sector."
                    if campaign_state.sector_choice == "public"
                    else "Working with the Private Sector."
                )
            )
            current_summary = f"<p>Current campaign run: Level {level_index + 1} – {escape(level_name, False)}. {sector_note}</p>"
        free_play_actions = (
            "<div class='mode-actions'>"
            "<a href='/free-play'>Configure Free Play</a>"
            "<a class='secondary' href='/start'>Resume Current Run</a>"
            "</div>"
        )
        if campaign_state.active:
            campaign_actions = (
                "<div class='mode-actions'>"
                "<a href='/campaign/level'>Resume Campaign</a>"
                "<form method='post' action='/campaign/start'>"
                "<button type='submit' class='secondary'>Restart Campaign</button>"
                "</form>"
                "</div>"
            )
        else:
            campaign_actions = (
                "<div class='mode-actions'>"
                "<form method='post' action='/campaign/start'>"
                "<button type='submit'>Begin Campaign</button>"
                "</form>"
                "</div>"
            )
        campaign_description = "<p>Tackle a guided three-level journey through scenarios 01, 02, and 03. At the start of each level you choose to coordinate with the Public or Private sector, which reshapes your faction alignment and the coalitions available to you.</p>"
        free_play_description = "<p>Experiment freely with every system in the negotiation sandbox. Tune the active scenario, scoring thresholds, and pacing rules before diving straight into a single open-ended session.</p>"
        body = (
            "<main class='landing-page'>"
            + "<h1>Keep the Future Human Survival RPG</h1>"
            + "<p class='landing-tagline'>AI Safety Negotiation Game</p>"
            + "<div class='mode-container'>"
            + "<section class='mode-panel'><div class='mode-content'>"
            + "<span class='mode-tag'>Sandbox Mode</span>"
            + "<h2>Free Play</h2>"
            + free_play_description
            + free_play_actions
            + "</div></section>"
            + "<section class='mode-panel'><div class='mode-content'>"
            + "<span class='mode-tag'>Story Mode</span>"
            + "<h2>Campaign</h2>"
            + campaign_description
            + current_summary
            + campaign_actions
            + "</div></section>"
            + "</div></main>"
        )
        return _render_page(body, body_class="page-landing")

    @app.route("/free-play", methods=["GET", "POST"])
    def free_play() -> Response | str:
        nonlocal config_in_use, current_mode
        current_mode = "free_play"
        campaign_state.active = False
        config_snapshot = config_in_use
        selectable_scenarios = [
            name
            for name in available_scenarios
            if name not in FREE_PLAY_HIDDEN_SCENARIOS
        ]
        if not selectable_scenarios:
            selectable_scenarios = list(available_scenarios)
        preferred_players = ("ScientificCommunity", "CivilSociety")
        allowed_player_factions = [
            faction for faction in preferred_players if faction in known_factions
        ]
        if not allowed_player_factions:
            allowed_player_factions = list(preferred_players)
        form_config = config_snapshot
        validation_errors: list[str] = []

        def _parse_int(field: str, fallback: int) -> int:
            try:
                return int(request.form.get(field, fallback))
            except (TypeError, ValueError):
                return fallback

        def _parse_float(field: str, fallback: float) -> float:
            try:
                return float(request.form.get(field, fallback))
            except (TypeError, ValueError):
                return fallback

        if request.method == "POST":
            scenario = request.form.get("scenario", config_snapshot.scenario)
            scenario = (scenario or config_snapshot.scenario).strip().lower()
            if scenario not in selectable_scenarios:
                if config_snapshot.scenario in selectable_scenarios:
                    scenario = config_snapshot.scenario
                elif selectable_scenarios:
                    scenario = selectable_scenarios[0]
            win_threshold = max(
                0, _parse_int("win_threshold", config_snapshot.win_threshold)
            )
            max_rounds = max(1, _parse_int("max_rounds", config_snapshot.max_rounds))
            roll_threshold = max(
                1,
                _parse_int(
                    "roll_success_threshold", config_snapshot.roll_success_threshold
                ),
            )
            char_limit = max(
                1,
                _parse_int(
                    "format_prompt_character_limit",
                    config_snapshot.format_prompt_character_limit,
                ),
            )
            force_after = max(
                0,
                _parse_int(
                    "conversation_force_action_after",
                    config_snapshot.conversation_force_action_after,
                ),
            )
            action_cost = max(
                0.0,
                _parse_float(
                    "action_time_cost_years", config_snapshot.action_time_cost_years
                ),
            )
            selected_factions = [
                faction.strip()
                for faction in request.form.getlist("enabled_factions")
                if faction.strip()
            ]
            filtered_factions: list[str] = []
            for faction in selected_factions:
                if faction in known_factions and faction not in filtered_factions:
                    filtered_factions.append(faction)
            selected_factions = filtered_factions
            if not selected_factions:
                selected_factions = list(config_snapshot.enabled_factions)
            if len(selected_factions) < 3:
                validation_errors.append(
                    "Select at least three factions to enable free play."
                )
            player_faction = (
                request.form.get("player_faction", config_snapshot.player_faction)
                or config_snapshot.player_faction
            )
            player_faction = player_faction.strip()
            if player_faction not in allowed_player_factions:
                if config_snapshot.player_faction in allowed_player_factions:
                    player_faction = config_snapshot.player_faction
                else:
                    player_faction = allowed_player_factions[0]
            form_config = GameConfig(
                scenario=scenario,
                win_threshold=win_threshold,
                max_rounds=max_rounds,
                roll_success_threshold=roll_threshold,
                action_time_cost_years=action_cost,
                format_prompt_character_limit=char_limit,
                conversation_force_action_after=force_after,
                enabled_factions=tuple(selected_factions),
                player_faction=player_faction,
            )
            if not validation_errors:
                with state_lock:
                    _reload_state(form_config)
                return redirect("/start")

        scenario_options = []
        for name in selectable_scenarios:
            selected = " selected" if name == form_config.scenario else ""
            scenario_options.append(
                "<option value='{value}'{selected}>{label}</option>".format(
                    value=escape(name, False),
                    selected=selected,
                    label=escape(_scenario_display_name(name), False),
                )
            )
        faction_options = []
        active_factions = set(form_config.enabled_factions)
        for faction in known_factions:
            selected = " selected" if faction in active_factions else ""
            faction_options.append(
                "<option value='{value}'{selected}>{label}</option>".format(
                    value=escape(faction, False),
                    selected=selected,
                    label=escape(_format_faction(faction), False),
                )
            )
        player_options = []
        for faction in allowed_player_factions:
            selected = " selected" if faction == form_config.player_faction else ""
            player_options.append(
                "<option value='{value}'{selected}>{label}</option>".format(
                    value=escape(faction, False),
                    selected=selected,
                    label=escape(_format_faction(faction), False),
                )
            )
        if not player_options and config_snapshot.player_faction:
            label = escape(_format_faction(config_snapshot.player_faction), False)
            player_options.append(
                "<option value='{value}' selected>{label}</option>".format(
                    value=escape(config_snapshot.player_faction, False),
                    label=label,
                )
            )
        error_html = ""
        if validation_errors:
            error_items = "".join(
                f"<li>{escape(message, False)}</li>" for message in validation_errors
            )
            error_html = f"<div class='config-error'><ul>{error_items}</ul></div>"
        field_help = config_field_help_texts
        form_body = (
            "<main class='config-page'>"
            + "<h1>Configure Free Play</h1>"
            + "<section class='config-settings'>"
            + "<form method='post'>"
            + error_html
            + "<label>"
            + _config_label("Scenario", field_help["scenario"])
            + "<select name='scenario'>"
            + "".join(scenario_options)
            + "</select></label>"
            + "<label>"
            + _config_label("Win threshold", field_help["win_threshold"])
            + f"<input type='number' name='win_threshold' min='0' value='{form_config.win_threshold}'></label>"
            + "<label>"
            + _config_label("Max rounds", field_help["max_rounds"])
            + f"<input type='number' name='max_rounds' min='1' value='{form_config.max_rounds}'></label>"
            + "<label>"
            + _config_label(
                "Roll success threshold", field_help["roll_success_threshold"]
            )
            + f"<input type='number' name='roll_success_threshold' min='1' value='{form_config.roll_success_threshold}'></label>"
            + "<label>"
            + _config_label(
                "Action time cost (years)", field_help["action_time_cost_years"]
            )
            + f"<input type='number' step='0.1' min='0' name='action_time_cost_years' value='{form_config.action_time_cost_years}'></label>"
            + "<label>"
            + _config_label(
                "Prompt character limit", field_help["format_prompt_character_limit"]
            )
            + f"<input type='number' min='1' name='format_prompt_character_limit' value='{form_config.format_prompt_character_limit}'></label>"
            + "<label>"
            + _config_label(
                "Conversation force action after",
                field_help["conversation_force_action_after"],
            )
            + f"<input type='number' min='0' name='conversation_force_action_after' value='{form_config.conversation_force_action_after}'></label>"
            + "<label>"
            + _config_label("Enabled factions", field_help["enabled_factions"])
            + "<select name='enabled_factions' multiple size='6'>"
            + "".join(faction_options)
            + "</select></label>"
            + "<label>"
            + _config_label("Player faction", field_help["player_faction"])
            + "<select name='player_faction'>"
            + "".join(player_options)
            + "</select></label>"
            + "<p class='config-note'>Applying new settings resets the current game immediately.</p>"
            + "<button type='submit'>Apply &amp; Start Free Play</button>"
            + "</form>"
            + "</section>"
            + "</main>"
        )
        return _render_page(form_body)

    @app.route("/campaign/start", methods=["POST"])
    def campaign_start() -> Response:
        nonlocal current_mode, config_in_use
        current_mode = "campaign"
        campaign_state.reset()
        with state_lock:
            _reload_state(load_game_config())
            config_snapshot = config_in_use
        logger.info("Campaign started with baseline config %s", config_snapshot)
        return redirect("/campaign/level")

    @app.route("/campaign/level", methods=["GET", "POST"])
    def campaign_level() -> Response | str:
        nonlocal current_mode, config_in_use
        if not campaign_state.active:
            return redirect("/")
        current_mode = "campaign"
        level_index = min(campaign_state.current_level, len(CAMPAIGN_SCENARIOS) - 1)
        scenario_key = CAMPAIGN_SCENARIOS[level_index]
        scenario_name = _scenario_display_name(scenario_key)
        summary_text = _scenario_summary_text(scenario_key)
        summary_html = _format_summary_html(summary_text)
        if request.method == "POST":
            selected_sector = request.form.get("sector", "").strip().lower()
            if selected_sector not in {"public", "private"}:
                return redirect("/campaign/level")
            campaign_state.sector_choice = selected_sector
            new_config = _campaign_config(level_index, selected_sector)
            logger.info(
                "Starting campaign level %s with %s sector",
                level_index + 1,
                selected_sector,
            )
            with state_lock:
                _reload_state(new_config)
            return redirect("/start")

        sector_cards = []
        public_blurb = (
            "<p>Build coalitions with governments, regulators, and civil society advocates.\n"
            "Shape law, oversight, and democratic pressure from inside the public sphere.</p>"
        )
        private_blurb = (
            "<p>Negotiate with corporate and hardware power-brokers alongside the scientific community.\n"
            "Influence incentives and technical safeguards from within industry.</p>"
        )
        public_details = (
            "<ul>"
            + "".join(
                f"<li>{escape(_format_faction(faction), False)}</li>"
                for faction in PUBLIC_SECTOR_FACTIONS
            )
            + "</ul>"
            + "<p><strong>Player faction:</strong> Scientific Community</p>"
        )
        public_persona_section = ""
        public_profile = _player_profile_by_faction("ScientificCommunity")
        if public_profile:
            return_target = quote("/campaign/level", safe="")
            profile_url = f"{_player_persona_path('ScientificCommunity')}?return={return_target}"
            public_persona_section = (
                "<div class='sector-player-preview-wrapper'><h3>Your Persona</h3>"
                + _sector_preview_block(
                    public_profile, label="Scientific Community", profile_url=profile_url
                )
                + "</div>"
            )
        private_details = (
            "<ul>"
            + "".join(
                f"<li>{escape(_format_faction(faction), False)}</li>"
                for faction in PRIVATE_SECTOR_FACTIONS
            )
            + "</ul>"
            + "<p><strong>Player faction:</strong> Civil Society</p>"
        )
        private_persona_section = ""
        private_profile = _player_profile_by_faction("CivilSociety")
        if private_profile:
            return_target = quote("/campaign/level", safe="")
            profile_url = f"{_player_persona_path('CivilSociety')}?return={return_target}"
            private_persona_section = (
                "<div class='sector-player-preview-wrapper'><h3>Your Persona</h3>"
                + _sector_preview_block(
                    private_profile, label="Civil Society", profile_url=profile_url
                )
                + "</div>"
            )
        sector_cards.append(
            "<article class='sector-card'>"
            "<h2>Partner with the Public Sector</h2>"
            + public_blurb
            + public_details
            + public_persona_section
            + "<form method='post'><input type='hidden' name='sector' value='public'>"
            + "<button type='submit'>Engage Public Sector</button></form>"
            + "</article>"
        )
        sector_cards.append(
            "<article class='sector-card'>"
            "<h2>Partner with the Private Sector</h2>"
            + private_blurb
            + private_details
            + private_persona_section
            + "<form method='post'><input type='hidden' name='sector' value='private'>"
            + "<button type='submit'>Engage Private Sector</button></form>"
            + "</article>"
        )
        active_sector_note = ""
        if campaign_state.sector_choice:
            label = (
                "Public Sector"
                if campaign_state.sector_choice == "public"
                else "Private Sector"
            )
            active_sector_note = f"<p><strong>Previous selection:</strong> {escape(label, False)}. Choosing a sector again will restart this level.</p>"
        body = (
            "<section class='campaign-container'>"
            + "<div class='campaign-header'>"
            + f"<h1>Level {level_index + 1}: {escape(scenario_name, False)}</h1>"
            + "<p>Select who you will coordinate with before launching the next negotiation.</p>"
            + (
                f"<div class='campaign-summary'>{summary_html}</div>"
                if summary_html
                else ""
            )
            + active_sector_note
            + "</div>"
            + "<div class='sector-grid'>"
            + "".join(sector_cards)
            + "</div>"
            + "</section>"
        )
        return _render_page(body)

    @app.route("/campaign/next", methods=["POST"])
    def campaign_next() -> Response:
        nonlocal current_mode, config_in_use
        if not campaign_state.active:
            return redirect("/")
        if campaign_state.current_level >= len(CAMPAIGN_SCENARIOS) - 1:
            return redirect("/campaign/complete")
        campaign_state.current_level += 1
        campaign_state.sector_choice = None
        current_mode = "campaign"
        with state_lock:
            _reload_state(load_game_config())
        logger.info("Advanced to campaign level %s", campaign_state.current_level + 1)
        return redirect("/campaign/level")

    @app.route("/campaign/complete", methods=["GET"])
    def campaign_complete() -> str:
        nonlocal current_mode
        current_mode = "free_play"
        campaign_state.active = False
        summaries = []
        for idx, scenario in enumerate(CAMPAIGN_SCENARIOS):
            record = campaign_state.level_outcomes.get(idx)
            sector = record.get("sector") if record else None
            sector_label = {
                "public": "Public Sector",
                "private": "Private Sector",
            }.get(sector, "Not played")
            score_text = "Not completed"
            result_text = ""
            if record:
                score_text = f"Score: {record['score']:.0f} / {record['threshold']:.0f}"
                result_text = (
                    "Victory" if record["score"] >= record["threshold"] else "Defeat"
                )
            summaries.append(
                "<li>"
                + f"<h3>Level {idx + 1}: {escape(_scenario_display_name(scenario), False)}</h3>"
                + f"<p><strong>Sector:</strong> {escape(sector_label, False)}</p>"
                + f"<p><strong>{result_text or 'Pending'}:</strong> {score_text}</p>"
                + "</li>"
            )
        body = (
            "<section class='campaign-container'>"
            + "<div class='campaign-header'>"
            + "<h1>Campaign Summary</h1>"
            + "<p>Review how each level unfolded. You can return to the home screen to start a fresh run or dive into free play with your preferred settings.</p>"
            + "</div>"
            + "<ol class='campaign-summary-list'>"
            + "".join(summaries)
            + "</ol>"
            + "<div class='campaign-actions'>"
            + "<a href='/' class='secondary'>Back to Home</a>"
            + "<a href='/free-play'>Configure Free Play</a>"
            + "</div>"
            + "</section>"
        )
        return _render_page(body)

    @app.route("/start", methods=["GET"])
    def list_characters() -> Response:
        logger.info("Listing characters")
        nonlocal last_history_signature
        if (
            current_mode == "campaign"
            and campaign_state.active
            and campaign_state.sector_choice is None
        ):
            return redirect("/campaign/level")
        with state_lock:
            score = game_state.final_weighted_score()
            hist_len = len(game_state.history)
            time_status = game_state.formatted_time_status()
            state_html = game_state.render_state()
            characters = list(game_state.characters)
            history_snapshot = list(game_state.history)
            player = game_state.player_character
            conversation_snapshots = [
                game_state.conversation_history(char) for char in characters
            ]
            action_snapshots = [
                list(game_state.available_npc_actions(char)) for char in characters
            ]
            credibility_values = [
                game_state.current_credibility(getattr(char, "faction", None))
                for char in characters
            ]
        if (
            score >= game_state.config.win_threshold
            or hist_len >= game_state.config.max_rounds
        ):
            return redirect("/result")
        if enable_parallel:
            history_signature = tuple(
                (str(label), str(action)) for label, action in history_snapshot
            )
            if history_signature != last_history_signature:
                pending_player_options.clear()
                player_option_signatures.clear()
                pending_npc_responses.clear()
            pending_player_choices.clear()
            last_history_signature = history_signature

            def launch(
                idx: int,
                char: Character,
                convo: Sequence[ConversationEntry],
                actions: Sequence[ResponseOption],
            ) -> None:
                if convo:
                    return
                key = _player_pending_key(idx, len(convo))
                signature = _player_context_signature(history_snapshot, convo, actions)
                existing_signature = player_option_signatures.get(key)
                entry = pending_player_options.get(key)
                if existing_signature == signature and entry is not None:
                    current_event, value = entry
                    if value is not None or not current_event.is_set():
                        return
                pending_player_options.pop(key, None)
                player_option_signatures.pop(key, None)
                event = threading.Event()
                pending_player_options[key] = (event, None)
                player_option_signatures[key] = signature

                def worker(
                    hist: Sequence[Tuple[str, str]],
                    partner: Character,
                    snapshots: Sequence[ConversationEntry],
                    cache_snapshot: Mapping[str, Sequence[ConversationEntry]],
                    pending_key: Tuple[int, int],
                    pending_event: threading.Event,
                    expected_signature: tuple,
                ) -> None:
                    try:
                        credibility = game_state.current_credibility(
                            getattr(partner, "faction", None)
                        )
                        options = player.generate_responses(
                            hist,
                            snapshots,
                            partner,
                            partner_credibility=credibility,
                            conversation_cache=cache_snapshot,
                        )
                    except Exception:  # pragma: no cover - defensive logging
                        logger.exception(
                            "Failed to generate player responses in background"
                        )
                        options = []
                    current = pending_player_options.get(pending_key)
                    if current is None:
                        pending_event.set()
                        return
                    current_event, _ = current
                    if current_event is not pending_event:
                        pending_event.set()
                        return
                    if player_option_signatures.get(pending_key) != expected_signature:
                        pending_event.set()
                        return
                    pending_player_options[pending_key] = (
                        pending_event,
                        list(options),
                    )
                    pending_event.set()

                cache_snapshot = {
                    faction: tuple(entries)
                    for faction, entries in game_state.conversation_cache_for_player(
                        char
                    ).items()
                }
                threading.Thread(
                    target=worker,
                    args=(
                        tuple(history_snapshot),
                        char,
                        tuple(convo),
                        cache_snapshot,
                        key,
                        event,
                        signature,
                    ),
                    daemon=True,
                ).start()

            for idx, (char, convo, actions) in enumerate(
                zip(characters, conversation_snapshots, action_snapshots)
            ):
                launch(idx, char, convo, actions)
        option_items = []
        for idx, (char, credibility) in enumerate(zip(characters, credibility_values)):
            if credibility is None:
                credibility_text = "Credibility: N/A"
            else:
                credibility_text = f"Credibility: {int(credibility)}"
            option_items.append(
                "<div class='character-option'>"
                + "<div class='character-input'>"
                + f"<input type='radio' name='character' value='{idx}' id='char{idx}'>"
                + f"<label for='char{idx}'>"
                + f"{escape(char.display_name, quote=False)}"
                + f"<span class='character-credibility'>{escape(credibility_text, False)}</span>"
                + "</label></div>"
                + f"<a href='/characters/{idx}/profile'>View profile</a>"
                + "</div>"
            )
        options = "<div class='character-options'>" + "".join(option_items) + "</div>"
        summary_section = _scenario_summary_section(
            getattr(game_state, "scenario_summary", "")
        )
        time_block = f"<p class='character-timing' id='time-status'>{escape(time_status, False)}</p>"
        body = (
            "<main class='character-select-container'>"
            + "<h1>Keep the Future Human Survival RPG</h1>"
            + summary_section
            + time_block
            + "<form method='get' action='/actions' class='character-select-form'>"
            + options
            + "<div class='character-select-actions'><button type='submit'>Talk</button></div>"
            + "</form>"
            + "<div class='character-select-actions'>"
            + "<form method='post' action='/reset'>"
            + "<button type='submit' class='secondary'>Reset</button>"
            + "</form>"
            + "</div>"
            + f"<div class='state-container'>{state_html}</div>"
            + "</main>"
        )
        return _render_page(body, extra_scripts=[state_refresh_script])

    @app.route("/player/personas/<string:key>", methods=["GET"])
    def show_campaign_player_persona(key: str) -> Response:
        return_target = request.args.get("return", "")
        safe_return = return_target if return_target.startswith("/") else None
        normalized = _normalize_key(key)
        profile = None
        if normalized:
            profile = player_personas_by_faction.get(normalized) or player_personas_by_name.get(
                normalized
            )
        if profile is None:
            return redirect("/campaign/level")
        persona_html = _persona_card_from_profile(profile)
        name = str(profile.get("name", "") or "Player Persona")
        back_href = safe_return or "/campaign/level"
        actions = [
            "<div class='profile-actions'>",
            f"<a class='secondary' href='{escape(back_href, quote=True)}'>Back to sector selection</a>",
            "<a href='/'>Back to home</a>",
            "</div>",
        ]
        body = (
            "<main class='profile-page'>"
            + f"<h1>{escape(name, False)}</h1>"
            + persona_html
            + "".join(actions)
            + "</main>"
        )
        return _render_page(body, body_class="page-profile", extra_scripts=[state_refresh_script])

    @app.route("/characters/<int:char_id>/profile", methods=["GET"])
    def show_character_profile(char_id: int) -> Response:
        return_target = request.args.get("return", "")
        safe_return = return_target if return_target.startswith("/") else None
        with state_lock:
            if char_id < 0 or char_id >= len(game_state.characters):
                return redirect("/start")
            character = game_state.characters[char_id]
            player_faction = getattr(game_state.player_character, "faction", None)
            credibility_value = game_state.current_credibility(
                getattr(character, "faction", None)
            )
        persona_html = _persona_card_for_character(character)
        matrix_html = _credibility_matrix_block(
            player_faction=player_faction,
            partner_faction=getattr(character, "faction", None),
            credibility_value=credibility_value,
        )
        actions = ["<div class='profile-actions'>"]
        if safe_return:
            actions.append(
                f"<a class='secondary' href='{escape(safe_return, quote=True)}'>Return to conversation</a>"
            )
        actions.append("<a href='/start'>Back to character selection</a>")
        actions.append("</div>")
        body = (
            "<main class='profile-page'>"
            + f"<h1>{escape(character.display_name, False)}</h1>"
            + persona_html
            + matrix_html
            + "".join(actions)
            + "</main>"
        )
        return _render_page(body, body_class="page-profile", extra_scripts=[state_refresh_script])

    @app.route("/player/profile", methods=["GET"])
    def show_player_profile() -> Response:
        return_target = request.args.get("return", "")
        safe_return = return_target if return_target.startswith("/") else None
        with state_lock:
            player_character = game_state.player_character
        persona_html = _persona_card_for_character(player_character)
        actions = ["<div class='profile-actions'>"]
        if safe_return:
            actions.append(
                f"<a class='secondary' href='{escape(safe_return, quote=True)}'>Return to conversation</a>"
            )
        actions.append("<a href='/start'>Back to character selection</a>")
        actions.append("</div>")
        body = (
            "<main class='profile-page'>"
            + f"<h1>{escape(player_character.display_name, False)}</h1>"
            + persona_html
            + "".join(actions)
            + "</main>"
        )
        return _render_page(body, body_class="page-profile", extra_scripts=[state_refresh_script])

    def _character_snapshot(
        char_id: int,
    ) -> Tuple[
        Character,
        List[Tuple[str, str]],
        List[ConversationEntry],
        List[ResponseOption],
        str,
        Character,
        Dict[str, str],
        int | None,
    ]:
        with state_lock:
            character = game_state.characters[char_id]
            history = list(game_state.history)
            conversation = game_state.conversation_history(character)
            available_actions = list(game_state.available_npc_actions(character))
            state_html = game_state.render_state()
            player = game_state.player_character
            action_labels = game_state.action_label_map(character)
            credibility_value = game_state.current_credibility(
                getattr(character, "faction", None)
            )
        return (
            character,
            history,
            conversation,
            available_actions,
            state_html,
            player,
            action_labels,
            credibility_value,
        )

    def _resolve_player_options(
        char_id: int,
        history: Sequence[Tuple[str, str]],
        conversation: Sequence[ConversationEntry],
        character: Character,
        player: Character,
        *,
        available_actions: Sequence[ResponseOption] | None = None,
    ) -> Tuple[List[ResponseOption] | None, bool]:
        conversation_length = len(conversation)
        key = _player_pending_key(char_id, conversation_length)
        signature = _player_context_signature(history, conversation, available_actions)
        existing_signature = player_option_signatures.get(key)
        entry = pending_player_options.get(key)
        if existing_signature is not None and existing_signature != signature:
            pending_player_options.pop(key, None)
            player_option_signatures.pop(key, None)
            entry = None
        if entry is not None:
            event, value = entry
            if value is not None:
                _clear_player_option_entries(char_id, keep_length=conversation_length)
                return list(value), False
            if not event.is_set():
                return None, True
            _clear_player_option_entries(char_id, keep_length=conversation_length)
            return [], False

        if not enable_parallel:
            credibility = game_state.current_credibility(
                getattr(character, "faction", None)
            )
            cache_snapshot = game_state.conversation_cache_for_player(character)
            options = player.generate_responses(
                history,
                conversation,
                character,
                partner_credibility=credibility,
                conversation_cache=cache_snapshot,
            )
            event = threading.Event()
            event.set()
            pending_player_options[key] = (event, list(options))
            player_option_signatures[key] = signature
            _clear_player_option_entries(char_id, keep_length=conversation_length)
            return list(options), False

        event = threading.Event()
        pending_player_options[key] = (event, None)
        player_option_signatures[key] = signature

        def worker(
            hist: Sequence[Tuple[str, str]],
            convo: Sequence[ConversationEntry],
            cache_snapshot: Mapping[str, Sequence[ConversationEntry]],
            partner: Character,
            pending_key: Tuple[int, int],
            expected_signature: tuple,
        ) -> None:
            try:
                credibility = game_state.current_credibility(
                    getattr(partner, "faction", None)
                )
                options = player.generate_responses(
                    hist,
                    convo,
                    partner,
                    partner_credibility=credibility,
                    conversation_cache=cache_snapshot,
                )
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Failed to generate player responses in background")
                options = []
            current = pending_player_options.get(pending_key)
            if current is None:
                event.set()
                return
            current_event, _ = current
            if current_event is not event:
                event.set()
                return
            if player_option_signatures.get(pending_key) != expected_signature:
                event.set()
                return
            pending_player_options[pending_key] = (event, list(options))
            event.set()

        cache_snapshot = {
            faction: tuple(entries)
            for faction, entries in game_state.conversation_cache_for_player(
                character
            ).items()
        }
        threading.Thread(
            target=worker,
            args=(
                tuple(history),
                tuple(conversation),
                cache_snapshot,
                character,
                key,
                signature,
            ),
            daemon=True,
        ).start()
        _clear_player_option_entries(char_id, keep_length=conversation_length)
        return None, True

    def _option_signature(option: ResponseOption) -> str:
        payload = option.to_payload()
        return json.dumps(payload, sort_keys=True, default=str)

    def _player_context_signature(
        history: Sequence[Tuple[str, str]],
        conversation: Sequence[ConversationEntry],
        actions: Sequence[ResponseOption] | None = None,
    ) -> tuple:
        history_sig = tuple((str(label), str(action)) for label, action in history)
        conversation_sig = tuple(
            (entry.speaker, entry.text, entry.type) for entry in conversation
        )
        if actions:
            action_sig = tuple(
                (
                    option.text,
                    option.type,
                    option.related_triplet,
                    option.related_attribute,
                )
                for option in actions
            )
        else:
            action_sig = ()
        return history_sig, conversation_sig, action_sig

    def _player_pending_key(char_id: int, conversation_length: int) -> Tuple[int, int]:
        return (char_id, conversation_length)

    def _clear_player_option_entries(
        char_id: int, keep_length: int | None = None
    ) -> None:
        removable = [
            key
            for key in pending_player_options
            if key[0] == char_id and (keep_length is None or key[1] != keep_length)
        ]
        for key in removable:
            pending_player_options.pop(key, None)
            player_option_signatures.pop(key, None)

    def _npc_pending_key(
        char_id: int, conversation_length: int, signature: str
    ) -> Tuple[int, int, str]:
        return (char_id, conversation_length, signature)

    def _clear_pending_npc_entries(char_id: int, signature: str | None = None) -> None:
        removable = [
            key
            for key in pending_npc_responses
            if key[0] == char_id and (signature is None or key[2] == signature)
        ]
        for key in removable:
            pending_npc_responses.pop(key, None)

    def _preload_npc_responses(
        char_id: int,
        history: Sequence[Tuple[str, str]],
        conversation: Sequence[ConversationEntry],
        options: Sequence[ResponseOption],
        character: Character,
        player: Character,
    ) -> None:
        if not enable_parallel:
            return
        base_length = len(conversation)
        for option in options:
            if option.type != "chat":
                continue
            signature = _option_signature(option)
            key = _npc_pending_key(char_id, base_length + 1, signature)
            if key in pending_npc_responses:
                continue
            done_event = threading.Event()
            pending_npc_responses[key] = (done_event, None)

            def worker(
                hist: Sequence[Tuple[str, str]],
                convo: Sequence[ConversationEntry],
                response_option: ResponseOption,
                partner: Character,
                expected_length: int,
                pending_key: Tuple[int, int, str],
            ) -> None:
                try:
                    simulated = list(convo) + [
                        ConversationEntry(
                            speaker=partner.display_name,
                            text=response_option.text,
                            type=response_option.type,
                        )
                    ]
                    credibility = game_state.current_credibility(
                        getattr(character, "faction", None)
                    )
                    limit = getattr(
                        game_state.config, "conversation_force_action_after", 0
                    )
                    force_action_required = limit > 0 and len(simulated) >= limit
                    replies = character.generate_responses(
                        hist,
                        simulated,
                        partner,
                        partner_credibility=credibility,
                        force_action=force_action_required,
                    )
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception("Failed to generate NPC responses in background")
                    replies = []
                if len(simulated) != expected_length:
                    # Conversation changed while computing; drop result.
                    pending_npc_responses.pop(pending_key, None)
                    return
                pending_npc_responses[pending_key] = (done_event, list(replies))
                done_event.set()

            threading.Thread(
                target=worker,
                args=(
                    tuple(history),
                    tuple(conversation),
                    option,
                    player,
                    base_length + 1,
                    key,
                ),
                daemon=True,
            ).start()

    def _resolve_npc_responses(
        char_id: int,
        conversation_length: int,
        option: ResponseOption,
        history: Sequence[Tuple[str, str]],
        conversation: Sequence[ConversationEntry],
        character: Character,
        player: Character,
    ) -> Tuple[List[ResponseOption] | None, bool]:
        signature = _option_signature(option)
        key = _npc_pending_key(char_id, conversation_length, signature)
        entry = pending_npc_responses.get(key)
        if enable_parallel and entry is not None:
            event, value = entry
            if value is not None:
                return list(value), False
            if not event.is_set():
                return None, True
            refreshed = pending_npc_responses.get(key)
            if refreshed is not None and refreshed is not entry:
                refreshed_event, refreshed_value = refreshed
                if refreshed_value is not None:
                    return list(refreshed_value), False
                if not refreshed_event.is_set():
                    return None, True
            return None, True
        if enable_parallel and entry is None:
            pending_choice = pending_player_choices.get(char_id)
            if (
                pending_choice
                and pending_choice[0] == conversation_length
                and pending_choice[1] == signature
            ):
                return None, True
        credibility = game_state.current_credibility(
            getattr(character, "faction", None)
        )
        force_action_required = game_state.should_force_action(character)
        replies = character.generate_responses(
            history,
            conversation,
            player,
            partner_credibility=credibility,
            force_action=force_action_required,
        )
        if enable_parallel:
            event = threading.Event()
            event.set()
            pending_npc_responses[key] = (event, list(replies))
        return list(replies), False

    def _render_conversation(
        char_id: int,
        character: Character,
        conversation: Sequence[ConversationEntry],
        chat_options: Sequence[ResponseOption],
        action_options: Sequence[ResponseOption],
        state_html: str,
        player: Character,
        action_labels: Dict[str, str],
        partner_credibility: int | None,
        *,
        loading_chat: bool = False,
    ) -> str:
        logger.debug(
            "Rendering conversation for %s with %d history items",
            character.name,
            len(conversation),
        )
        option_items: List[str] = []
        option_counter = 0
        for option in chat_options:
            payload = escape(json.dumps(option.to_payload()), quote=True)
            label_html = escape(option.text, quote=False)
            option_items.append(
                "<li class='response-option chat-option'>"
                + f"<input type='radio' name='response' value='{payload}' id='opt{option_counter}' data-kind='chat'>"
                + f"<label for='opt{option_counter}'><span class='option-title'>{label_html}</span></label>"
                + "</li>"
            )
            option_counter += 1
        for action_index, option in enumerate(action_options, 1):
            payload = escape(json.dumps(option.to_payload()), quote=True)
            label_text = action_labels.get(option.text)
            if not label_text:
                attribute = (
                    option.related_attribute.title()
                    if option.related_attribute
                    else "None"
                )
                label_text = f"Action {action_index} [{attribute}]"
            option_items.append(
                "<li class='response-option action-option'>"
                + f"<input type='radio' name='response' value='{payload}' id='opt{option_counter}' data-kind='action'>"
                + f"<label for='opt{option_counter}'>"
                + f"<span class='option-title'>{escape(label_text, quote=False)}</span>"
                + (
                    "<span class='option-description' title='{desc}'>{text}</span>".format(
                        desc=escape(option.text, quote=True),
                        text=escape(option.text, quote=False),
                    )
                )
                + "</label></li>"
            )
            option_counter += 1
        options_html_parts: List[str] = []
        if loading_chat:
            options_html_parts.append(_loading_markup("Loading chat responses..."))
        if option_items:
            options_html_parts.append("<ul>" + "".join(option_items) + "</ul>")
        elif not loading_chat:
            options_html_parts.append(
                "<p class='empty-conversation'>No options available.</p>"
            )
        options_html = "".join(options_html_parts)
        form_html = (
            "<form class='options-form' method='post' action='/actions'>"
            + options_html
            + f"<input type='hidden' name='character' value='{char_id}'>"
            + "<div class='options-actions'><button type='submit' class='primary-button'>Send</button></div>"
            + "</form>"
        )
        conversation_html = _conversation_log(conversation)
        conversation_panel = (
            "<div class='conversation-content'>"
            + f"<section class='conversation-thread'><h2>Conversation with {escape(character.display_name, quote=False)}</h2>{conversation_html}</section>"
            + f"<section class='conversation-responses'><h2>Responses</h2>{form_html}</section>"
            + "<section class='conversation-actions'><a class='primary-button secondary' href='/start'>Back to characters</a></section>"
            + "</div>"
        )
        player_panel = _profile_panel(
            player,
            profile_url=f"/player/profile?return=/actions?character={char_id}",
        )
        partner_panel = _profile_panel(
            character,
            credibility=partner_credibility,
            profile_url=f"/characters/{char_id}/profile?return=/actions?character={char_id}",
        )
        page = _conversation_frame(
            player_panel, conversation_panel, partner_panel, state_html
        )
        return _render_page(
            page,
            extra_scripts=[roll_indicator_script, state_refresh_script],
        )

    def _render_success_page(
        char_id: int,
        character: Character,
        attempt: ActionAttempt,
        conversation: Sequence[ConversationEntry],
        state_html: str,
        player: Character,
        partner_credibility: int | None,
        roll_threshold: int,
    ) -> str:
        total = attempt.effective_score + attempt.roll
        attribute_label = attempt.attribute or "none"
        success_text = (
            f"Succeeded {attempt.label} (attribute {attribute_label}: {attempt.effective_score}, "
            f"roll={attempt.roll:.2f}, total={total:.2f}, threshold={roll_threshold})"
        )
        credibility_delta = (
            -attempt.credibility_cost
            if attempt.option.related_triplet is not None
            else attempt.credibility_gain
        )
        credibility_text = f"Credibility change: {credibility_delta:+d}."
        keep_talking_form = (
            "<form method='get' action='/actions'>"
            + f"<input type='hidden' name='character' value='{char_id}'>"
            + f"<button type='submit' class='primary-button'>Keep talking to {escape(character.display_name, quote=False)}</button>"
            + "</form>"
        )
        back_form = (
            "<form method='get' action='/start'>"
            + "<button type='submit' class='primary-button secondary'>Back to character selection</button>"
            + "</form>"
        )
        outcome_section = (
            "<section class='action-outcome'>"
            + "<h2>Action Outcome</h2>"
            + f"<p>{escape(success_text, quote=False)}</p>"
            + f"<p>{escape(credibility_text, quote=False)}</p>"
            + "<div class='action-outcome-actions'>"
            + keep_talking_form
            + back_form
            + "</div></section>"
        )
        conversation_section = (
            "<section class='conversation-thread'><h2>Conversation So Far</h2>"
            + _conversation_log(conversation)
            + "</section>"
        )
        conversation_panel = (
            "<div class='conversation-content'>"
            + outcome_section
            + conversation_section
            + "</div>"
        )
        player_panel = _profile_panel(
            player,
            profile_url=f"/player/profile?return=/actions?character={char_id}",
        )
        partner_panel = _profile_panel(
            character,
            credibility=partner_credibility,
            profile_url=f"/characters/{char_id}/profile?return=/actions?character={char_id}",
        )
        page = _conversation_frame(
            player_panel, conversation_panel, partner_panel, state_html
        )
        return _render_page(
            page,
            extra_scripts=[roll_indicator_script, state_refresh_script],
        )

    def _render_failure_page(
        char_id: int,
        character: Character,
        attempt: ActionAttempt,
        conversation: Sequence[ConversationEntry],
        state_html: str,
        player: Character,
        next_cost: int,
        partner_credibility: int | None,
        *,
        can_reroll: bool,
        credibility_notes: Sequence[str],
    ) -> str:
        failure_text = attempt.failure_text or (
            f"Failed {attempt.label} (attribute {attempt.attribute or 'none'}: {attempt.effective_score}, roll={attempt.roll:.2f})"
        )
        if next_cost > 0:
            reroll_note = f"Reroll will cost {next_cost} credibility."
            reroll_label = f"Reroll (-{next_cost} credibility)"
        else:
            reroll_note = "Reroll will not cost additional credibility."
            reroll_label = "Reroll (no cost)"
        payload = escape(json.dumps(attempt.option.to_payload()), quote=True)
        shortage_text = ""
        if not can_reroll:
            base = "Insufficient credibility to reroll."
            if credibility_notes:
                details = "; ".join(credibility_notes)
                shortage_text = f"<p class='warning-text'>{escape(base + ' ' + details, quote=False)}</p>"
            else:
                shortage_text = (
                    f"<p class='warning-text'>{escape(base, quote=False)}</p>"
                )
        reroll_forms = "<div class='reroll-actions'>"
        if can_reroll:
            reroll_forms += (
                "<form method='post' action='/reroll' class='roll-trigger'>"
                + f"<input type='hidden' name='character' value='{char_id}'>"
                + f"<input type='hidden' name='action' value='{payload}'>"
                + f"<button type='submit' class='primary-button'>{escape(reroll_label, False)}</button>"
                + "</form>"
            )
        reroll_forms += (
            "<form method='post' action='/finalize_failure'>"
            + f"<input type='hidden' name='character' value='{char_id}'>"
            + f"<input type='hidden' name='action' value='{payload}'>"
            + "<button type='submit' class='primary-button secondary'>Accept Failure</button>"
            + "</form>"
        )
        reroll_forms += "</div>"
        outcome_section = (
            "<section class='action-outcome'>"
            + "<h2>Action Outcome</h2>"
            + f"<p>{escape(failure_text, quote=False)}</p>"
            + f"<p>{escape(reroll_note, quote=False)}</p>"
            + shortage_text
            + reroll_forms
            + "</section>"
        )
        conversation_section = (
            "<section class='conversation-thread'><h2>Conversation So Far</h2>"
            + _conversation_log(conversation)
            + "</section>"
        )
        conversation_panel = (
            "<div class='conversation-content'>"
            + outcome_section
            + conversation_section
            + "</div>"
        )
        player_panel = _profile_panel(
            player,
            profile_url=f"/player/profile?return=/actions?character={char_id}",
        )
        partner_panel = _profile_panel(
            character,
            credibility=partner_credibility,
            profile_url=f"/characters/{char_id}/profile?return=/actions?character={char_id}",
        )
        page = _conversation_frame(
            player_panel, conversation_panel, partner_panel, state_html
        )
        return _render_page(
            page,
            extra_scripts=[roll_indicator_script, state_refresh_script],
        )

    @app.route("/actions", methods=["GET", "POST"])
    def character_actions() -> Response:
        char_id = int(request.values["character"])
        if request.method == "POST" and "response" in request.form:
            option = _option_from_payload(request.form["response"])
            with state_lock:
                character = game_state.characters[char_id]
            logger.info(
                "Player selected %s (%s) for %s",
                option.text,
                option.type,
                character.name,
            )
            if option.is_action:
                with state_lock:
                    game_state.log_player_response(character, option)
                    attempt = game_state.attempt_action(character, option)
                    chars_snapshot = list(game_state.characters)
                    history_snapshot = list(game_state.history)
                    conversation_snapshot = game_state.conversation_history(character)
                    player_snapshot = game_state.player_character
                    state_html = game_state.render_state()
                    next_cost = game_state.next_reroll_cost(character, option)
                    can_reroll, reroll_shortages = game_state.reroll_affordability(
                        character, option
                    )
                    partner_credibility = game_state.current_credibility(
                        getattr(character, "faction", None)
                    )
                    roll_threshold_snapshot = game_state.config.roll_success_threshold
                    game_state.clear_available_actions(character)
                shortage_messages = [
                    f"{target}: have {available}, need {needed}"
                    for target, available, needed in reroll_shortages
                ]
                _clear_player_option_entries(char_id)
                pending_player_choices.pop(char_id, None)
                _clear_pending_npc_entries(char_id)

                if attempt.success:
                    partner_view = partner_credibility
                    roll_threshold = roll_threshold_snapshot
                    if enable_parallel:
                        with state_lock:
                            game_state.start_assessment()
                            interim_state_html = game_state.render_state()
                            partner_view = game_state.current_credibility(
                                getattr(character, "faction", None)
                            )
                            roll_threshold = game_state.config.roll_success_threshold

                        def run_assessment(
                            chars: List[Character],
                            hist: List[Tuple[str, str]],
                        ) -> None:
                            try:
                                scores = assessor.assess(
                                    chars,
                                    hist,
                                    parallel=True,
                                )
                                with state_lock:
                                    game_state.update_progress(scores)
                            finally:
                                with assessment_lock:
                                    try:
                                        assessment_threads.remove(
                                            threading.current_thread()
                                        )
                                    except ValueError:
                                        pass

                        t = threading.Thread(
                            target=run_assessment,
                            args=(chars_snapshot, history_snapshot),
                            daemon=True,
                        )
                        with assessment_lock:
                            assessment_threads.append(t)
                        t.start()
                        latest_state_html = interim_state_html
                    else:
                        with state_lock:
                            game_state.start_assessment()
                        scores = assessor.assess(chars_snapshot, history_snapshot)
                        with state_lock:
                            game_state.update_progress(scores)
                            latest_state_html = game_state.render_state()
                            partner_view = game_state.current_credibility(
                                getattr(character, "faction", None)
                            )
                            roll_threshold = game_state.config.roll_success_threshold
                    success_page = _render_success_page(
                        char_id,
                        character,
                        attempt,
                        conversation_snapshot,
                        latest_state_html,
                        player_snapshot,
                        partner_view,
                        roll_threshold,
                    )
                    return Response(success_page)

                failure_page = _render_failure_page(
                    char_id,
                    character,
                    attempt,
                    conversation_snapshot,
                    state_html,
                    player_snapshot,
                    next_cost,
                    partner_credibility,
                    can_reroll=can_reroll,
                    credibility_notes=shortage_messages,
                )
                return Response(failure_page)

            with state_lock:
                history_snapshot = list(game_state.history)
                game_state.log_player_response(character, option)
                conversation = game_state.conversation_history(character)
                player = game_state.player_character
            conversation_length = len(conversation)
            signature = _option_signature(option)
            pending_player_choices[char_id] = (
                conversation_length,
                signature,
                option,
            )
            replies, waiting = _resolve_npc_responses(
                char_id,
                conversation_length,
                option,
                history_snapshot,
                conversation,
                character,
                player,
            )
            if waiting:
                body = (
                    "<main class='loading-page'>"
                    + _loading_markup("Loading...")
                    + "</main>"
                    + f"<meta http-equiv='refresh' content='1;url=/actions?character={char_id}'>"
                )
                return _render_page(body)
            replies = replies or []
            with state_lock:
                game_state.log_npc_responses(character, replies)
            pending_player_choices.pop(char_id, None)
            _clear_pending_npc_entries(char_id, signature)
            _clear_player_option_entries(char_id)
            return redirect(f"/actions?character={char_id}")

        (
            character,
            history,
            conversation,
            npc_actions,
            state_html,
            player,
            action_labels,
            partner_credibility,
        ) = _character_snapshot(char_id)
        pending_choice = pending_player_choices.get(char_id)
        if pending_choice:
            expected_length, signature, chosen_option = pending_choice
            if (
                len(conversation) == expected_length
                and conversation
                and conversation[-1].speaker == player.display_name
            ):
                replies, waiting = _resolve_npc_responses(
                    char_id,
                    expected_length,
                    chosen_option,
                    history,
                    conversation,
                    character,
                    player,
                )
                if waiting:
                    body = (
                        "<main class='loading-page'>"
                        + _loading_markup("Loading...")
                        + "</main>"
                        + f"<meta http-equiv='refresh' content='1;url=/actions?character={char_id}'>"
                    )
                    return _render_page(body)
                replies = replies or []
                with state_lock:
                    game_state.log_npc_responses(character, replies)
                    history = list(game_state.history)
                    conversation = game_state.conversation_history(character)
                    npc_actions = list(game_state.available_npc_actions(character))
                    state_html = game_state.render_state()
                    action_labels = game_state.action_label_map(character)
                    partner_credibility = game_state.current_credibility(
                        getattr(character, "faction", None)
                    )
                pending_player_choices.pop(char_id, None)
                _clear_pending_npc_entries(char_id, signature)
                _clear_player_option_entries(char_id)
            else:
                pending_player_choices.pop(char_id, None)
                _clear_pending_npc_entries(char_id, signature)
                _clear_player_option_entries(char_id)
        else:
            pending_player_choices.pop(char_id, None)
            _clear_pending_npc_entries(char_id)
            _clear_player_option_entries(char_id, keep_length=len(conversation))

        options, loading = _resolve_player_options(
            char_id,
            history,
            conversation,
            character,
            player,
            available_actions=npc_actions,
        )
        resolved_options = options or []
        chat_options = [opt for opt in resolved_options if not opt.is_action]
        action_bucket: Dict[str, ResponseOption] = {
            action.text: action for action in npc_actions
        }
        for opt in resolved_options:
            if opt.is_action and opt.text not in action_bucket:
                action_bucket[opt.text] = opt
        action_options = list(action_bucket.values())
        if not loading:
            _preload_npc_responses(
                char_id,
                history,
                conversation,
                list(chat_options) + action_options,
                character,
                player,
            )
        page = _render_conversation(
            char_id,
            character,
            conversation,
            list(chat_options),
            action_options,
            state_html,
            player,
            action_labels,
            partner_credibility,
            loading_chat=loading,
        )
        if loading:
            page += f"<meta http-equiv='refresh' content='1;url=/actions?character={char_id}'>"
        return Response(page)

    @app.route("/reroll", methods=["POST"])
    def reroll_action_route() -> Response:
        char_id = int(request.form["character"])
        option = _option_from_payload(request.form["action"])
        with state_lock:
            character = game_state.characters[char_id]
            can_reroll, reroll_shortages = game_state.reroll_affordability(
                character, option
            )
            if not can_reroll:
                attempt = game_state.pending_failures.get((character.name, option.text))
                if attempt is None:
                    raise ValueError("No pending failed action to reroll")
                next_can_reroll = can_reroll
                next_shortages = reroll_shortages
            else:
                attempt = game_state.reroll_action(character, option)
                next_can_reroll, next_shortages = game_state.reroll_affordability(
                    character, option
                )
            chars_snapshot = list(game_state.characters)
            history_snapshot = list(game_state.history)
            conversation_snapshot = game_state.conversation_history(character)
            player_snapshot = game_state.player_character
            state_html = game_state.render_state()
            next_cost = game_state.next_reroll_cost(character, option)
            partner_credibility = game_state.current_credibility(
                getattr(character, "faction", None)
            )
            roll_threshold_snapshot = game_state.config.roll_success_threshold
            if can_reroll:
                game_state.clear_available_actions(character)
        shortage_messages = [
            f"{target}: have {available}, need {needed}"
            for target, available, needed in next_shortages
        ]
        if not can_reroll:
            failure_page = _render_failure_page(
                char_id,
                character,
                attempt,
                conversation_snapshot,
                state_html,
                player_snapshot,
                next_cost,
                partner_credibility,
                can_reroll=next_can_reroll,
                credibility_notes=shortage_messages,
            )
            return Response(failure_page)
        if attempt.success:
            partner_view = partner_credibility
            roll_threshold = roll_threshold_snapshot
            if enable_parallel:
                with state_lock:
                    game_state.start_assessment()
                    interim_state_html = game_state.render_state()
                    partner_view = game_state.current_credibility(
                        getattr(character, "faction", None)
                    )
                    roll_threshold = game_state.config.roll_success_threshold

                def run_assessment(
                    chars: List[Character],
                    hist: List[Tuple[str, str]],
                ) -> None:
                    try:
                        scores = assessor.assess(
                            chars,
                            hist,
                            parallel=True,
                        )
                        with state_lock:
                            game_state.update_progress(scores)
                    finally:
                        with assessment_lock:
                            try:
                                assessment_threads.remove(threading.current_thread())
                            except ValueError:
                                pass

                t = threading.Thread(
                    target=run_assessment,
                    args=(chars_snapshot, history_snapshot),
                    daemon=True,
                )
                with assessment_lock:
                    assessment_threads.append(t)
                t.start()
                latest_state_html = interim_state_html
            else:
                with state_lock:
                    game_state.start_assessment()
                scores = assessor.assess(chars_snapshot, history_snapshot)
                with state_lock:
                    game_state.update_progress(scores)
                    latest_state_html = game_state.render_state()
                    partner_view = game_state.current_credibility(
                        getattr(character, "faction", None)
                    )
                    roll_threshold = game_state.config.roll_success_threshold
            success_page = _render_success_page(
                char_id,
                character,
                attempt,
                conversation_snapshot,
                latest_state_html,
                player_snapshot,
                partner_view,
                roll_threshold,
            )
            return Response(success_page)

        failure_page = _render_failure_page(
            char_id,
            character,
            attempt,
            conversation_snapshot,
            state_html,
            player_snapshot,
            next_cost,
            partner_credibility,
            can_reroll=next_can_reroll,
            credibility_notes=shortage_messages,
        )
        return Response(failure_page)

    @app.route("/finalize_failure", methods=["POST"])
    def finalize_failure_route() -> Response:
        char_id = int(request.form["character"])
        option = _option_from_payload(request.form["action"])
        with state_lock:
            character = game_state.characters[char_id]
            game_state.finalize_failed_action(character, option)
            game_state.start_assessment()
            chars_snapshot = list(game_state.characters)
            history_snapshot = list(game_state.history)
        if enable_parallel:

            def run_assessment(
                chars: List[Character],
                hist: List[Tuple[str, str]],
            ) -> None:
                try:
                    scores = assessor.assess(
                        chars,
                        hist,
                        parallel=True,
                    )
                    with state_lock:
                        game_state.update_progress(scores)
                finally:
                    with assessment_lock:
                        try:
                            assessment_threads.remove(threading.current_thread())
                        except ValueError:
                            pass

            t = threading.Thread(
                target=run_assessment,
                args=(chars_snapshot, history_snapshot),
                daemon=True,
            )
            with assessment_lock:
                assessment_threads.append(t)
            t.start()
        else:
            scores = assessor.assess(chars_snapshot, history_snapshot)
            with state_lock:
                game_state.update_progress(scores)
        return redirect("/start")

    @app.route("/reset", methods=["POST"])
    def reset() -> Response:
        logger.info("Resetting game state")
        with state_lock:
            _reload_state(config_in_use)
        return redirect("/")


    @app.route("/factions", methods=["GET"])
    def list_faction_details() -> str:
        with state_lock:
            details = [dict(entry) for entry in game_state.all_faction_details()]
        articles = [
            _render_faction_article(
                detail,
                heading_tag="h2",
                include_link=True,
                include_quotes=False,
            )
            for detail in details
        ]
        content = "".join(articles)
        body = (
            "<section class='faction-detail-page'>"
            + "<h1>Faction detail directory</h1>"
            + scoring_note
            + "<div class='faction-grid'>"
            + content
            + "</div>"
            + "<div class='faction-actions'>"
            + "<a class='secondary' href='/start'>Back to game</a>"
            + "</div>"
            + "</section>"
        )
        return _render_page(body)

    @app.route("/factions/<string:slug>", methods=["GET"])
    def show_faction_detail(slug: str) -> str:
        with state_lock:
            detail = game_state.faction_detail(slug)
        if detail is None:
            return redirect('/factions')
        article = _render_faction_article(
            detail, heading_tag="h1", include_link=False, include_quotes=True
        )
        body = (
            "<section class='faction-detail-page single'>"
            + article
            + scoring_note
            + "<div class='faction-actions'>"
            + "<a class='secondary' href='/factions'>Browse all factions</a>"
            + "<a href='/start'>Back to game</a>"
            + "</div>"
            + "</section>"
        )
        return _render_page(body)

    @app.route("/state", methods=["GET"])
    def state_snapshot() -> Response:
        with state_lock:
            snapshot_html = game_state.render_state()
            time_status = game_state.formatted_time_status()
            version = game_state.progress_version
            pending = game_state.assessment_pending
        payload = {
            "state_html": snapshot_html,
            "time_status": time_status,
            "progress_version": version,
            "assessment_pending": pending,
        }
        return Response(json.dumps(payload), mimetype="application/json")

    @app.route("/instructions", methods=["GET"])
    def instructions() -> str:
        steps = """<ol>
            <li>Speak with the representatives of each faction shaping how humanity navigates advanced AI.
            Build rapport and understand their perspectives.</li>
            <li>Convince partners to propose actions that close their outstanding gaps.
            <a href='/factions'>Browse faction details</a> to review the current gaps, desired end states, and reference material.</li>
            <li>Stay alert for proposals that further a faction's misaligned interests.
            You can accept them, but expect trade-offs later.</li>
            <li>After every action all factions are reassessed on a 0–100 scale, judging how well recent moves reduced their gaps.</li>
            <li>Manage two core resources:
                <ul>
                    <li><strong>Time</strong> – every action advances the in-game calendar, limiting how many interventions you can pursue.</li>
                    <li><strong>Credibility</strong> – aligned actions consume credibility with partner factions, while indulging misaligned requests may boost it.
                    Credibility costs differ for each player–faction pairing.</li>
                </ul>
            </li>
            <li>Win by pushing the final weighted score above the campaign's victory threshold.
            Lose if the in-game calendar expires before you secure that mandate.</li>
        </ol>"""
        body = (
            "<section class='instructions-page'>"
            + "<h1>How to keep the future human</h1>"
            + steps
            + "<h2>Reference material</h2>"
            + "<p>Review the <a href='/factions'>faction dossiers</a> for the current reference material, including gaps, desired end states, and quotes.</p>"
            + "<p>The Keep the Future Human coalition evaluates every conversation through the lens of its documented gaps and referenced quotes.</p>"
            + "<div class='instructions-actions'>"
            + "<a href='/start'>Return to the campaign</a>"
            + "<a class='secondary' href='/factions'>Browse faction details</a>"
            + "</div>"
            + "</section>"
        )
        return _render_page(body)

    @app.route("/result", methods=["GET"])
    def result() -> str:
        with assessment_lock:
            running = any(t.is_alive() for t in assessment_threads)
        if running:
            body = (
                "<main class='loading-page'>"
                + _loading_markup("Waiting for assessments...")
                + "</main>"
                + "<meta http-equiv='refresh' content='1'>"
            )
            return _render_page(body)
        campaign_context: Tuple[int, str, str | None] | None = None
        with state_lock:
            final = game_state.final_weighted_score()
            state_html = game_state.render_state()
            threshold = game_state.config.win_threshold
            scenario_key = game_state.config.scenario
            if current_mode == "campaign":
                level_index = min(
                    campaign_state.current_level, len(CAMPAIGN_SCENARIOS) - 1
                )
                sector_choice = campaign_state.sector_choice
                campaign_context = (
                    level_index,
                    scenario_key,
                    sector_choice,
                )
                campaign_state.level_outcomes[level_index] = {
                    "score": final,
                    "threshold": threshold,
                    "sector": sector_choice,
                    "scenario": scenario_key,
                }
        is_win = final >= threshold
        if not campaign_context:
            outcome = "You won!" if is_win else "You lost!"
            body = (
                "<main class='result-page'>"
                + f"<h1>{outcome}</h1>"
                + f"{state_html}"
                + "<form method='post' action='/reset'>"
                + "<button type='submit'>Reset</button>"
                + "</form>"
                + "</main>"
            )
            return _render_page(body)

        level_index, scenario_key, sector_choice = campaign_context
        level_number = level_index + 1
        scenario_name = _scenario_display_name(scenario_key)
        sector_label = {
            "public": "Public Sector",
            "private": "Private Sector",
        }.get(sector_choice, "Unknown coalition")
        result_title = "Victory" if is_win else "Defeat"
        score_summary = f"Final score {final:.0f} with a threshold of {threshold:.0f}."
        level_intro = (
            f"Level {level_number} – {escape(scenario_name, False)}.<br>"
            f"You partnered with the {escape(sector_label, False)}."
        )
        actions: List[str] = []
        actions.append(
            "<form method='post' action='/reset'>"
            "<button type='submit' class='secondary'>Restart Level (same sector)</button>"
            "</form>"
        )
        actions.append("<a href='/campaign/level'>Change Sector</a>")
        if is_win:
            if level_index >= len(CAMPAIGN_SCENARIOS) - 1:
                actions.append("<a href='/campaign/complete'>View Campaign Summary</a>")
            else:
                next_label = f"Advance to Level {level_number + 1}"
                actions.append(
                    "<form method='post' action='/campaign/next'>"
                    f"<button type='submit'>{next_label}</button>"
                    "</form>"
                )
        header = (
            "<section class='campaign-container'>"
            + "<div class='campaign-header'>"
            + f"<h1>{result_title}</h1>"
            + f"<p>{level_intro}<br>{score_summary}</p>"
            + "</div>"
            + f"<div class='campaign-summary'>{state_html}</div>"
            + "<div class='campaign-actions'>"
            + "".join(actions)
            + "</div>"
            + "</section>"
        )
        return _render_page(header)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
