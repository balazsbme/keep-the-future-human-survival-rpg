"""Flask web service exposing the RPG demo over HTTP."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from flask import Flask, Response, redirect, request
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


logger = logging.getLogger(__name__)

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
    footer = (
        "<p><a href='/instructions'>Instructions</a> | "
        f"<a href='{GITHUB_URL}'>GitHub</a></p>"
    )
    panel_style = (
        "<style>"
        ".layout-container{display:flex;gap:1.5rem;align-items:flex-start;}"
        ".panel{flex:1;padding:0 1rem;box-sizing:border-box;}"
        ".player-panel,.partner-panel{flex:1.2;max-width:220px;}"
        ".conversation-panel{flex:5;}"
        ".options-form ul{list-style:none;padding:0;margin:0;}"
        ".options-form li{margin-bottom:0.75rem;}"
        ".profile-card{display:flex;flex-direction:column;align-items:center;gap:0.75rem;padding:1rem;border-radius:10px;background:#fafafa;border:1px solid #e2e2e2;}"
        ".profile-photo{width:120px;height:120px;border-radius:10px;background:linear-gradient(135deg,#ececec,#d5d5d5);display:flex;align-items:center;justify-content:center;color:#666;font-weight:600;font-size:0.9rem;}"
        ".profile-name{margin:0;font-size:1.1rem;text-align:center;}"
        ".attribute-list{list-style:none;padding:0;margin:0;width:100%;}"
        ".attribute-list li{display:flex;justify-content:space-between;padding:0.25rem 0;border-bottom:1px solid #e5e5e5;font-size:0.9rem;}"
        ".attribute-list li:last-child{border-bottom:none;}"
        ".credibility-box{width:100%;padding:0.5rem;border:1px solid #d6d6d6;border-radius:8px;background:#fffdfa;text-align:center;font-size:0.9rem;}"
        ".credibility-box span{display:block;font-size:0.75rem;color:#555;margin-bottom:0.2rem;}"
        ".reroll-actions{display:flex;gap:0.75rem;flex-wrap:wrap;margin-top:0.5rem;}"
        ".reroll-actions form{margin:0;}"
        ".action-outcome-actions{display:flex;gap:0.75rem;flex-wrap:wrap;margin-top:0.75rem;}"
        ".action-outcome-actions form{margin:0;}"
        ".warning-text{color:#9c2f2f;font-weight:600;}"
        "</style>"
    )

    landing_style = (
        "<style>"
        "body{font-family:'Inter',sans-serif;margin:0;background:#f3f4f8;color:#1f2933;}"
        "h1{margin:2rem auto 1rem auto;text-align:center;font-size:2.4rem;}"
        ".mode-container{display:flex;flex-direction:column;min-height:90vh;}"
        ".mode-panel{flex:1;display:flex;align-items:center;justify-content:center;padding:3rem 1.5rem;}"
        ".mode-panel:first-child{background:linear-gradient(180deg,#eef2ff 0%,#ffffff 100%);}"
        ".mode-panel:last-child{background:linear-gradient(180deg,#fff7ed 0%,#ffffff 100%);border-top:1px solid #e0e7ff;}"
        ".mode-content{max-width:540px;text-align:center;}"
        ".mode-tag{display:inline-block;margin-bottom:0.75rem;padding:0.4rem 1rem;border-radius:999px;font-weight:600;font-size:0.9rem;background:rgba(29,78,216,0.12);color:#1d4ed8;}"
        ".mode-content h2{margin:0 0 0.75rem 0;font-size:2rem;}"
        ".mode-content p{margin:0 0 1.5rem 0;line-height:1.6;font-size:1.05rem;}"
        ".mode-actions{display:flex;justify-content:center;gap:1rem;flex-wrap:wrap;}"
        ".mode-actions a,.mode-actions button{display:inline-block;padding:0.75rem 1.75rem;font-size:1rem;border-radius:999px;border:none;background:#1d4ed8;color:#fff;text-decoration:none;cursor:pointer;box-shadow:0 10px 24px rgba(29,78,216,0.15);}"
        ".mode-actions a.secondary,.mode-actions button.secondary{background:#334155;}"
        ".mode-actions form{margin:0;}"
        "</style>"
    )

    campaign_style = (
        "<style>"
        ".campaign-container{max-width:960px;margin:2rem auto;font-family:'Inter',sans-serif;color:#1f2933;}"
        ".campaign-header{background:#ffffff;border-radius:16px;padding:2rem;box-shadow:0 14px 32px rgba(15,23,42,0.08);margin-bottom:2rem;}"
        ".campaign-header h1{margin:0 0 0.5rem 0;font-size:2.2rem;}"
        ".campaign-header p{margin:0.5rem 0 0 0;line-height:1.6;font-size:1.05rem;}"
        ".campaign-summary{margin-bottom:2rem;}"
        ".sector-grid{display:flex;flex-wrap:wrap;gap:1.5rem;}"
        ".sector-card{flex:1 1 280px;background:#ffffff;border-radius:14px;padding:1.75rem;box-shadow:0 12px 28px rgba(15,23,42,0.07);}"
        ".sector-card h2{margin:0;font-size:1.4rem;}"
        ".sector-card p{margin:0.75rem 0 0 0;line-height:1.5;}"
        ".sector-card ul{margin:0.75rem 0 1.25rem 1.25rem;line-height:1.5;}"
        ".sector-card form{margin-top:1.25rem;}"
        ".sector-card button{padding:0.7rem 1.6rem;border:none;border-radius:999px;background:#1d4ed8;color:#fff;font-size:1rem;cursor:pointer;box-shadow:0 10px 24px rgba(29,78,216,0.15);}"
        ".campaign-summary-list{list-style:none;padding:0;margin:0;display:grid;gap:1rem;}"
        ".campaign-summary-list li{background:#ffffff;border-radius:12px;padding:1.25rem;box-shadow:0 12px 28px rgba(15,23,42,0.07);}"
        ".campaign-summary-list h3{margin:0 0 0.5rem 0;font-size:1.2rem;}"
        ".campaign-summary-list p{margin:0.25rem 0;line-height:1.5;}"
        ".campaign-actions{margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;}"
        ".campaign-actions a,.campaign-actions button{padding:0.75rem 1.5rem;border-radius:999px;border:none;background:#1d4ed8;color:#fff;text-decoration:none;cursor:pointer;box-shadow:0 10px 24px rgba(29,78,216,0.15);}"
        ".campaign-actions a.secondary,.campaign-actions button.secondary{background:#334155;}"
        "</style>"
    )

    intro_style = (
        "<style>"
        ".instructions,.config-settings{margin:1rem 0;padding:1rem;border:1px solid #dcdcdc;border-radius:10px;background:#f8faff;}"
        ".instructions h2,.config-settings h2{margin-top:0;margin-bottom:0.5rem;}"
        ".instructions ul{margin:0.5rem 0 0 1.25rem;}"
        ".instructions li{margin-bottom:0.4rem;}"
        ".config-settings form{display:flex;flex-direction:column;gap:0.75rem;max-width:360px;}"
        ".config-settings label{display:flex;flex-direction:column;font-weight:600;font-size:0.95rem;}"
        ".config-settings input,.config-settings select{margin-top:0.25rem;padding:0.45rem;border:1px solid #c6c6c6;border-radius:6px;font-size:0.95rem;}"
        ".config-settings button{align-self:flex-start;padding:0.45rem 0.9rem;font-size:0.95rem;}"
        ".config-error{margin:0 0 0.75rem 0;padding:0.75rem;border:1px solid #d93025;background:#fdecea;color:#a50e0e;border-radius:8px;font-size:0.9rem;}"
        ".config-note{margin:0;font-size:0.85rem;color:#555;}"
        "</style>"
    )

    summary_style = (
        "<style>"
        ".scenario-summary{margin:1.5rem 0;padding:1rem;border:1px solid #dcdcdc;"
        "border-radius:10px;background:#f8f8f8;}"
        ".scenario-summary h2{margin:0 0 0.5rem 0;font-size:1.2rem;}"
        ".scenario-summary p{margin:0.5rem 0;line-height:1.5;}"
        "</style>"
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
            summary_style
            + "<section class='scenario-summary'><h2>Scenario Overview</h2>"
            + formatted
            + "</section>"
        )

    def _profile_panel(character: Character, *, credibility: int | None = None) -> str:
        attributes = "".join(
            f"<li><span>{label}</span><span>{int(character.attribute_score(label.lower()))}</span></li>"
            for label in ("Leadership", "Technology", "Policy", "Network")
        )
        credibility_block = ""
        if credibility is not None:
            credibility_block = (
                "<div class='credibility-box'>"
                "<span>Credibility</span>"
                f"<strong>{int(credibility)}</strong>"
                "</div>"
            )
        return (
            "<div class='profile-card'>"
            "<div class='profile-photo' role='img' aria-label='Portrait placeholder'>Portrait</div>"
            f"<h2 class='profile-name'>{escape(character.display_name, quote=False)}</h2>"
            f"<ul class='attribute-list'>{attributes}</ul>"
            f"{credibility_block}"
            "</div>"
        )

    @app.before_request
    def log_request() -> None:
        logger.info("%s %s", request.method, request.path)

    @app.route("/", methods=["GET"])
    def main_page() -> str:
        current_summary = ""
        if campaign_state.active:
            level_index = min(
                campaign_state.current_level, len(CAMPAIGN_SCENARIOS) - 1
            )
            level_name = _scenario_display_name(CAMPAIGN_SCENARIOS[level_index])
            sector_note = "Choose your next sector to begin." if campaign_state.sector_choice is None else (
                "Working with the Public Sector." if campaign_state.sector_choice == "public" else "Working with the Private Sector."
            )
            current_summary = (
                f"<p>Current campaign run: Level {level_index + 1} – {escape(level_name, False)}. {sector_note}</p>"
            )
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
        campaign_description = (
            "<p>Tackle a guided three-level journey through scenarios 01, 02, and 03. At the start of each level you choose to coordinate with the Public or Private sector, which reshapes your faction alignment and the coalitions available to you.</p>"
        )
        free_play_description = (
            "<p>Experiment freely with every system in the negotiation sandbox. Tune the active scenario, scoring thresholds, and pacing rules before diving straight into a single open-ended session.</p>"
        )
        body = (
            landing_style
            + "<main>"
            + "<h1>AI Safety Negotiation Game</h1>"
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
            + footer
        )
        return body

    @app.route("/free-play", methods=["GET", "POST"])
    def free_play() -> Response | str:
        nonlocal config_in_use, current_mode
        current_mode = "free_play"
        campaign_state.active = False
        config_snapshot = config_in_use
        selectable_scenarios = [
            name for name in available_scenarios if name not in FREE_PLAY_HIDDEN_SCENARIOS
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
            win_threshold = max(0, _parse_int("win_threshold", config_snapshot.win_threshold))
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
        form_body = (
            intro_style
            + "<h1>Configure Free Play</h1>"
            + "<section class='config-settings'>"
            + "<form method='post'>"
            + error_html
            + "<label>Scenario<select name='scenario'>"
            + "".join(scenario_options)
            + "</select></label>"
            + f"<label>Win threshold<input type='number' name='win_threshold' min='0' value='{form_config.win_threshold}'></label>"
            + f"<label>Max rounds<input type='number' name='max_rounds' min='1' value='{form_config.max_rounds}'></label>"
            + f"<label>Roll success threshold<input type='number' name='roll_success_threshold' min='1' value='{form_config.roll_success_threshold}'></label>"
            + f"<label>Action time cost (years)<input type='number' step='0.1' min='0' name='action_time_cost_years' value='{form_config.action_time_cost_years}'></label>"
            + f"<label>Prompt character limit<input type='number' min='1' name='format_prompt_character_limit' value='{form_config.format_prompt_character_limit}'></label>"
            + f"<label>Conversation force action after<input type='number' min='0' name='conversation_force_action_after' value='{form_config.conversation_force_action_after}'></label>"
            + "<label>Enabled factions<select name='enabled_factions' multiple size='6'>"
            + "".join(faction_options)
            + "</select></label>"
            + "<label>Player faction<select name='player_faction'>"
            + "".join(player_options)
            + "</select></label>"
            + "<p class='config-note'>Applying new settings resets the current game immediately.</p>"
            + "<button type='submit'>Apply &amp; Start Free Play</button>"
            + "</form>"
            + "</section>"
            + footer
        )
        return form_body

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
                "Starting campaign level %s with %s sector", level_index + 1, selected_sector
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
        private_details = (
            "<ul>"
            + "".join(
                f"<li>{escape(_format_faction(faction), False)}</li>"
                for faction in PRIVATE_SECTOR_FACTIONS
            )
            + "</ul>"
            + "<p><strong>Player faction:</strong> Civil Society</p>"
        )
        sector_cards.append(
            "<article class='sector-card'>"
            "<h2>Partner with the Public Sector</h2>"
            + public_blurb
            + public_details
            + "<form method='post'><input type='hidden' name='sector' value='public'>"
            + "<button type='submit'>Engage Public Sector</button></form>"
            + "</article>"
        )
        sector_cards.append(
            "<article class='sector-card'>"
            "<h2>Partner with the Private Sector</h2>"
            + private_blurb
            + private_details
            + "<form method='post'><input type='hidden' name='sector' value='private'>"
            + "<button type='submit'>Engage Private Sector</button></form>"
            + "</article>"
        )
        active_sector_note = ""
        if campaign_state.sector_choice:
            label = (
                "Public Sector" if campaign_state.sector_choice == "public" else "Private Sector"
            )
            active_sector_note = (
                f"<p><strong>Previous selection:</strong> {escape(label, False)}. Choosing a sector again will restart this level.</p>"
            )
        body = (
            campaign_style
            + "<section class='campaign-container'>"
            + "<div class='campaign-header'>"
            + f"<h1>Level {level_index + 1}: {escape(scenario_name, False)}</h1>"
            + "<p>Select who you will coordinate with before launching the next negotiation.</p>"
            + (f"<div class='campaign-summary'>{summary_html}</div>" if summary_html else "")
            + active_sector_note
            + "</div>"
            + "<div class='sector-grid'>"
            + "".join(sector_cards)
            + "</div>"
            + "</section>"
            + footer
        )
        return body

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
                score_text = (
                    f"Score: {record['score']:.0f} / {record['threshold']:.0f}"
                )
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
            campaign_style
            + "<section class='campaign-container'>"
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
            + footer
        )
        return body

    @app.route("/start", methods=["GET"])
    def list_characters() -> Response:
        logger.info("Listing characters")
        nonlocal last_history_signature
        if current_mode == "campaign" and campaign_state.active and campaign_state.sector_choice is None:
            return redirect("/campaign/level")
        with state_lock:
            score = game_state.final_weighted_score()
            hist_len = len(game_state.history)
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
        if score >= game_state.config.win_threshold or hist_len >= game_state.config.max_rounds:
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
                signature = _player_context_signature(
                    history_snapshot, convo, actions
                )
                existing_signature = player_option_signatures.get(key)
                entry = pending_player_options.get(key)
                if (
                    existing_signature == signature
                    and entry is not None
                ):
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
                        logger.exception("Failed to generate player responses in background")
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
                    for faction, entries in game_state.conversation_cache_for_player(char).items()
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
                "<div>"
                + f'<input type="radio" name="character" value="{idx}" id="char{idx}">'  # noqa: E501
                + f'<label for="char{idx}">{escape(char.display_name, quote=False)}'
                + f"<span style='display:block;font-size:0.85rem;color:#555;'>{credibility_text}</span>"  # noqa: E501
                + "</label></div>"
            )
        options = "".join(option_items)
        summary_section = _scenario_summary_section(
            getattr(game_state, "scenario_summary", "")
        )
        time_block = (
            f"<p style='font-weight:600;'>Time passed since November 2025: {game_state.time_elapsed_years:.1f} years</p>"
        )
        body = (
            "<h1>Keep the Future Human Survival RPG</h1>"
            + summary_section
            + time_block
            + "<form method='get' action='/actions'>"
            f"{options}"
            "<button type='submit'>Talk</button>"
            "</form>"
            "<form method='post' action='/reset'>"
            "<button type='submit'>Reset</button>"
            "</form>"
            f"{state_html}" + footer
        )
        return Response(body)

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
        signature = _player_context_signature(
            history, conversation, available_actions
        )
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
            for faction, entries in game_state.conversation_cache_for_player(character).items()
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
                    force_action_required = (
                        limit > 0 and len(simulated) >= limit
                    )
                    replies = character.generate_responses(
                        hist,
                        simulated,
                        partner,
                        partner_credibility=credibility,
                        force_action=force_action_required,
                    )
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception(
                        "Failed to generate NPC responses in background"
                    )
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
            if pending_choice and pending_choice[0] == conversation_length and pending_choice[1] == signature:
                return None, True
        credibility = game_state.current_credibility(getattr(character, "faction", None))
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
                f"<li><input type='radio' name='response' value='{payload}' id='opt{option_counter}'>"
                f"<label for='opt{option_counter}'>{label_html}</label></li>"
            )
            option_counter += 1
        for action_index, option in enumerate(action_options, 1):
            payload = escape(json.dumps(option.to_payload()), quote=True)
            label_text = action_labels.get(option.text)
            if not label_text:
                attribute = option.related_attribute.title() if option.related_attribute else "None"
                label_text = f"Action {action_index} [{attribute}]"
            option_items.append(
                f"<li><input type='radio' name='response' value='{payload}' id='opt{option_counter}'>"
                f"<label for='opt{option_counter}' title='{escape(option.text, quote=True)}'>"
                f"<strong>{escape(label_text, quote=False)}</strong></label></li>"
            )
            option_counter += 1
        options_html_parts: List[str] = []
        if loading_chat:
            options_html_parts.append("<p><em>Loading chat responses...</em></p>")
        if option_items:
            options_html_parts.append("<ul>" + "".join(option_items) + "</ul>")
        elif not loading_chat:
            options_html_parts.append("<p>No options available.</p>")
        options_html = "".join(options_html_parts)
        form_html = (
            "<form class='options-form' method='post' action='/actions'>"
            + options_html
            + f"<input type='hidden' name='character' value='{char_id}'>"
            + "<button type='submit'>Send</button>"
            + "</form>"
        )
        if conversation:
            convo_items = "".join(
                f"<li><strong>{escape(entry.speaker, quote=False)}</strong>: {escape(entry.text, quote=False)} "
                f"<em>({entry.type})</em></li>"
                for entry in conversation
            )
            convo_block = f"<ol>{convo_items}</ol>"
        else:
            convo_block = "<p>No conversation yet. Start by greeting the character.</p>"
        conversation_panel = (
            f"<section><h2>Conversation with {escape(character.display_name, quote=False)}</h2>{convo_block}</section>"
            + f"<section><h2>Responses</h2>{form_html}</section>"
            + "<section><a href='/start'>Back to characters</a></section>"
        )
        player_panel = _profile_panel(player)
        partner_panel = _profile_panel(character, credibility=partner_credibility)
        layout = (
            panel_style
            + "<div class='layout-container'>"
            + f"<div class='panel player-panel'>{player_panel}</div>"
            + f"<div class='panel conversation-panel'>{conversation_panel}</div>"
            + f"<div class='panel partner-panel'>{partner_panel}</div>"
            + "</div>"
        )
        return layout + state_html + footer

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
            + f"<button type='submit'>Keep talking to {escape(character.display_name, quote=False)}</button>"
            + "</form>"
        )
        back_form = (
            "<form method='get' action='/start'>"
            + "<button type='submit'>Back to character selection</button>"
            + "</form>"
        )
        outcome_section = (
            "<section><h2>Action Outcome</h2>"
            f"<p>{escape(success_text, quote=False)}</p>"
            f"<p>{escape(credibility_text, quote=False)}</p>"
            + "<div class='action-outcome-actions'>"
            + keep_talking_form
            + back_form
            + "</div></section>"
        )
        if conversation:
            convo_items = "".join(
                f"<li><strong>{escape(entry.speaker, quote=False)}</strong>: {escape(entry.text, quote=False)} <em>({entry.type})</em></li>"
                for entry in conversation
            )
            convo_block = (
                "<section><h2>Conversation So Far</h2><ol>" + convo_items + "</ol></section>"
            )
        else:
            convo_block = (
                "<section><h2>Conversation So Far</h2><p>No conversation yet.</p></section>"
            )
        conversation_panel = outcome_section + convo_block
        player_panel = _profile_panel(player)
        partner_panel = _profile_panel(character, credibility=partner_credibility)
        layout = (
            panel_style
            + "<div class='layout-container'>"
            + f"<div class='panel player-panel'>{player_panel}</div>"
            + f"<div class='panel conversation-panel'>{conversation_panel}</div>"
            + f"<div class='panel partner-panel'>{partner_panel}</div>"
            + "</div>"
        )
        return layout + state_html + footer

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
                shortage_text = (
                    f"<p class='warning-text'>{escape(base + ' ' + details, quote=False)}</p>"
                )
            else:
                shortage_text = f"<p class='warning-text'>{escape(base, quote=False)}</p>"
        outcome_section = (
            "<section><h2>Action Outcome</h2>"
            f"<p>{escape(failure_text, quote=False)}</p>"
            f"<p>{reroll_note}</p>"
        )
        if shortage_text:
            outcome_section += shortage_text
        outcome_section += "<div class='reroll-actions'>"
        if can_reroll:
            outcome_section += (
                "<form method='post' action='/reroll'>"
                + f"<input type='hidden' name='character' value='{char_id}'>"
                + f"<input type='hidden' name='action' value='{payload}'>"
                + f"<button type='submit'>{reroll_label}</button>"
                + "</form>"
            )
        outcome_section += (
            "<form method='post' action='/finalize_failure'>"
            + f"<input type='hidden' name='character' value='{char_id}'>"
            + f"<input type='hidden' name='action' value='{payload}'>"
            + "<button type='submit'>Accept Failure</button>"
            + "</form></div></section>"
        )
        if conversation:
            convo_items = "".join(
                f"<li><strong>{escape(entry.speaker, quote=False)}</strong>: {escape(entry.text, quote=False)} <em>({entry.type})</em></li>"
                for entry in conversation
            )
            convo_block = f"<section><h2>Conversation So Far</h2><ol>{convo_items}</ol></section>"
        else:
            convo_block = (
                "<section><h2>Conversation So Far</h2><p>No conversation yet.</p></section>"
            )
        conversation_panel = outcome_section + convo_block
        player_panel = _profile_panel(player)
        partner_panel = _profile_panel(character, credibility=partner_credibility)
        layout = (
            panel_style
            + "<div class='layout-container'>"
            + f"<div class='panel player-panel'>{player_panel}</div>"
            + f"<div class='panel conversation-panel'>{conversation_panel}</div>"
            + f"<div class='panel partner-panel'>{partner_panel}</div>"
            + "</div>"
        )
        return layout + state_html + footer
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
                    latest_state_html = state_html
                    partner_view = partner_credibility
                    roll_threshold = roll_threshold_snapshot
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
                        with state_lock:
                            latest_state_html = game_state.render_state()
                            partner_view = game_state.current_credibility(
                                getattr(character, "faction", None)
                            )
                            roll_threshold = game_state.config.roll_success_threshold
                    else:
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
                    "<p>Loading...</p>"
                    f"<meta http-equiv='refresh' content='1;url=/actions?character={char_id}'>"
                    f"{footer}"
                )
                return Response(body)
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
                        "<p>Loading...</p>"
                        f"<meta http-equiv='refresh' content='1;url=/actions?character={char_id}'>"
                        f"{footer}"
                    )
                    return Response(body)
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
            page += (
                f"<meta http-equiv='refresh' content='1;url=/actions?character={char_id}'>"
            )
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
                attempt = game_state.pending_failures.get(
                    (character.name, option.text)
                )
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
            latest_state_html = state_html
            partner_view = partner_credibility
            roll_threshold = roll_threshold_snapshot
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
                with state_lock:
                    latest_state_html = game_state.render_state()
                    partner_view = game_state.current_credibility(
                        getattr(character, "faction", None)
                    )
                    roll_threshold = game_state.config.roll_success_threshold
            else:
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

    @app.route("/instructions", methods=["GET"])
    def instructions() -> str:
        content = game_state.reference_material or "No reference material configured."
        return (
            "<h1>Instructions</h1>"
            "<h2>Reference material</h2>"
            f"<pre>{content}</pre>"
            "<a href='/start'>Back to game</a>"
            f"{footer}"
        )

    @app.route("/result", methods=["GET"])
    def result() -> str:
        nonlocal current_mode
        with assessment_lock:
            running = any(t.is_alive() for t in assessment_threads)
        if running:
            return (
                "<p>Waiting for assessments...</p>"
                "<meta http-equiv='refresh' content='1'>"
            )
        campaign_context: Tuple[int, str, str | None] | None = None
        with state_lock:
            final = game_state.final_weighted_score()
            state_html = game_state.render_state()
            threshold = game_state.config.win_threshold
            scenario_key = game_state.config.scenario
            if current_mode == "campaign":
                level_index = min(campaign_state.current_level, len(CAMPAIGN_SCENARIOS) - 1)
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
            return (
                f"<h1>{outcome}</h1>"
                f"{state_html}"
                "<form method='post' action='/reset'>"
                "<button type='submit'>Reset</button>"
                "</form>"
                f"{footer}"
            )

        level_index, scenario_key, sector_choice = campaign_context
        level_number = level_index + 1
        scenario_name = _scenario_display_name(scenario_key)
        sector_label = {
            "public": "Public Sector",
            "private": "Private Sector",
        }.get(sector_choice, "Unknown coalition")
        result_title = "Victory" if is_win else "Defeat"
        score_summary = (
            f"Final score {final:.0f} with a threshold of {threshold:.0f}."
        )
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
        actions.append(
            "<a href='/campaign/level'>Change Sector</a>"
        )
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
            campaign_style
            + "<section class='campaign-container'>"
            + "<div class='campaign-header'>"
            + f"<h1>{result_title}</h1>"
            + f"<p>{level_intro}<br>{score_summary}</p>"
            + "</div>"
            + f"<div class='campaign-summary'>{state_html}</div>"
            + "<div class='campaign-actions'>"
            + "".join(actions)
            + "</div>"
            + "</section>"
            + footer
        )
        return header

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
