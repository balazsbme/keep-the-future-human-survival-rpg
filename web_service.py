"""Flask web service exposing the RPG demo over HTTP."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
import logging
import os
import threading
from html import escape
from typing import Dict, List, Sequence, Tuple

from flask import Flask, Response, redirect, request

from cli_game import load_characters
from rpg.assessment_agent import AssessmentAgent
from rpg.character import Character, ResponseOption
from rpg.conversation import ConversationEntry
from rpg.game_state import ActionAttempt, GameState, WIN_THRESHOLD


GITHUB_URL = "https://github.com/balazsbme/keep-the-future-human-survival-rpg"


logger = logging.getLogger(__name__)


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
    initial_characters = load_characters()
    game_state = GameState(list(initial_characters))
    assessor = AssessmentAgent()
    enable_parallel = os.environ.get("ENABLE_PARALLELISM") == "1"
    pending_player_options: Dict[
        Tuple[int, int], Tuple[threading.Event, List[ResponseOption] | None]
    ] = {}
    pending_npc_responses: Dict[
        Tuple[int, int, str], Tuple[threading.Event, List[ResponseOption] | None]
    ] = {}
    pending_player_choices: Dict[int, Tuple[int, str, ResponseOption]] = {}
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
        turns = game_state.config.max_rounds
        summary_section = _scenario_summary_section(
            getattr(game_state, "scenario_summary", "")
        )
        return (
            "<h1>AI Safety Negotiation Game</h1>"
            + summary_section
            + "<p>You are an expert negotiator with connections to every major faction shaping AI governance. You can persuade their key representatives to propose and take actions. Your objective is to ensure AI is developed in humanity's best interest and keep the future human.</p>"
            f"<p>You have {turns} turns to reach a final weighted score of {WIN_THRESHOLD} or higher to win.</p>"
            "<a href='/start'>Start</a>"
            f"{footer}"
        )

    @app.route("/start", methods=["GET"])
    def list_characters() -> Response:
        logger.info("Listing characters")
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
            credibility_values = [
                game_state.current_credibility(getattr(char, "faction", None))
                for char in characters
            ]
        if score >= WIN_THRESHOLD or hist_len >= game_state.config.max_rounds:
            return redirect("/result")
        if enable_parallel:
            pending_player_options.clear()
            pending_npc_responses.clear()
            pending_player_choices.clear()

            def launch(idx: int, char: Character, convo: Sequence[ConversationEntry]) -> None:
                if convo:
                    return
                key = _player_pending_key(idx, len(convo))
                event = threading.Event()
                pending_player_options[key] = (event, None)

                def worker(
                    hist: Sequence[Tuple[str, str]],
                    partner: Character,
                    snapshots: Sequence[ConversationEntry],
                    pending_key: Tuple[int, int],
                    pending_event: threading.Event,
                ) -> None:
                    try:
                        options = player.generate_responses(hist, snapshots, partner)
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
                    pending_player_options[pending_key] = (
                        pending_event,
                        list(options),
                    )
                    pending_event.set()

                threading.Thread(
                    target=worker,
                    args=(
                        tuple(history_snapshot),
                        char,
                        tuple(convo),
                        key,
                        event,
                    ),
                    daemon=True,
                ).start()

            for idx, (char, convo) in enumerate(
                zip(characters, conversation_snapshots)
            ):
                launch(idx, char, convo)
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
        body = (
            "<h1>Keep the Future Human Survival RPG</h1>"
            + summary_section
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
    ) -> Tuple[List[ResponseOption] | None, bool]:
        conversation_length = len(conversation)
        key = _player_pending_key(char_id, conversation_length)
        entry = pending_player_options.get(key)
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
            options = player.generate_responses(history, conversation, character)
            event = threading.Event()
            event.set()
            pending_player_options[key] = (event, list(options))
            _clear_player_option_entries(char_id, keep_length=conversation_length)
            return list(options), False

        event = threading.Event()
        pending_player_options[key] = (event, None)

        def worker(
            hist: Sequence[Tuple[str, str]],
            convo: Sequence[ConversationEntry],
            partner: Character,
            pending_key: Tuple[int, int],
        ) -> None:
            try:
                options = player.generate_responses(hist, convo, partner)
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
            pending_player_options[pending_key] = (event, list(options))
            event.set()

        threading.Thread(
            target=worker,
            args=(tuple(history), tuple(conversation), character, key),
            daemon=True,
        ).start()
        _clear_player_option_entries(char_id, keep_length=conversation_length)
        return None, True

    def _option_signature(option: ResponseOption) -> str:
        payload = option.to_payload()
        return json.dumps(payload, sort_keys=True, default=str)

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
                    replies = character.generate_responses(hist, simulated, partner)
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
        replies = character.generate_responses(history, conversation, player)
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

    def _render_failure_page(
        char_id: int,
        character: Character,
        attempt: ActionAttempt,
        conversation: Sequence[ConversationEntry],
        state_html: str,
        player: Character,
        next_cost: int,
        partner_credibility: int | None,
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
        outcome_section = (
            "<section><h2>Action Outcome</h2>"
            f"<p>{escape(failure_text, quote=False)}</p>"
            f"<p>{reroll_note}</p>"
            + "<div class='reroll-actions'>"
            + "<form method='post' action='/reroll'>"
            + f"<input type='hidden' name='character' value='{char_id}'>"
            + f"<input type='hidden' name='action' value='{payload}'>"
            + f"<button type='submit'>{reroll_label}</button>"
            + "</form>"
            + "<form method='post' action='/finalize_failure'>"
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
                    how_to_win = game_state.how_to_win
                    conversation_snapshot = game_state.conversation_history(character)
                    player_snapshot = game_state.player_character
                    state_html = game_state.render_state()
                    next_cost = game_state.next_reroll_cost(character, option)
                    partner_credibility = game_state.current_credibility(
                        getattr(character, "faction", None)
                    )
                    game_state.clear_available_actions(character)
                _clear_player_option_entries(char_id)
                pending_player_choices.pop(char_id, None)
                _clear_pending_npc_entries(char_id)

                if attempt.success:
                    if enable_parallel:

                        def run_assessment(
                            chars: List[Character],
                            instructions: str,
                            hist: List[Tuple[str, str]],
                        ) -> None:
                            try:
                                scores = assessor.assess(
                                    chars,
                                    instructions,
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
                            args=(chars_snapshot, how_to_win, history_snapshot),
                            daemon=True,
                        )
                        with assessment_lock:
                            assessment_threads.append(t)
                        t.start()
                    else:
                        scores = assessor.assess(
                            chars_snapshot, how_to_win, history_snapshot
                        )
                        with state_lock:
                            game_state.update_progress(scores)
                    return redirect("/start")

                failure_page = _render_failure_page(
                    char_id,
                    character,
                    attempt,
                    conversation_snapshot,
                    state_html,
                    player_snapshot,
                    next_cost,
                    partner_credibility,
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
            char_id, history, conversation, character, player
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
            attempt = game_state.reroll_action(character, option)
            chars_snapshot = list(game_state.characters)
            history_snapshot = list(game_state.history)
            how_to_win = game_state.how_to_win
            conversation_snapshot = game_state.conversation_history(character)
            player_snapshot = game_state.player_character
            state_html = game_state.render_state()
            next_cost = game_state.next_reroll_cost(character, option)
            partner_credibility = game_state.current_credibility(
                getattr(character, "faction", None)
            )
            game_state.clear_available_actions(character)
        if attempt.success:
            if enable_parallel:

                def run_assessment(
                    chars: List[Character],
                    instructions: str,
                    hist: List[Tuple[str, str]],
                ) -> None:
                    try:
                        scores = assessor.assess(
                            chars,
                            instructions,
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
                    args=(chars_snapshot, how_to_win, history_snapshot),
                    daemon=True,
                )
                with assessment_lock:
                    assessment_threads.append(t)
                t.start()
            else:
                scores = assessor.assess(
                    chars_snapshot, how_to_win, history_snapshot
                )
                with state_lock:
                    game_state.update_progress(scores)
            return redirect("/start")

        failure_page = _render_failure_page(
            char_id,
            character,
            attempt,
            conversation_snapshot,
            state_html,
            player_snapshot,
            next_cost,
            partner_credibility,
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
            how_to_win = game_state.how_to_win
        if enable_parallel:

            def run_assessment(
                chars: List[Character],
                instructions: str,
                hist: List[Tuple[str, str]],
            ) -> None:
                try:
                    scores = assessor.assess(
                        chars,
                        instructions,
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
                args=(chars_snapshot, how_to_win, history_snapshot),
                daemon=True,
            )
            with assessment_lock:
                assessment_threads.append(t)
            t.start()
        else:
            scores = assessor.assess(
                chars_snapshot, how_to_win, history_snapshot
            )
            with state_lock:
                game_state.update_progress(scores)
        return redirect("/start")

    @app.route("/reset", methods=["POST"])
    def reset() -> Response:
        nonlocal game_state
        logger.info("Resetting game state")
        with state_lock:
            game_state = GameState(list(initial_characters))
        pending_player_options.clear()
        pending_npc_responses.clear()
        pending_player_choices.clear()
        with assessment_lock:
            assessment_threads.clear()
        return redirect("/start")

    @app.route("/instructions", methods=["GET"])
    def instructions() -> str:
        content = game_state.how_to_win
        return (
            "<h1>Instructions</h1>"
            f"<pre>{content}</pre>"
            "<a href='/start'>Back to game</a>"
            f"{footer}"
        )

    @app.route("/result", methods=["GET"])
    def result() -> str:
        with assessment_lock:
            running = any(t.is_alive() for t in assessment_threads)
        if running:
            return (
                "<p>Waiting for assessments...</p>"
                "<meta http-equiv='refresh' content='1'>"
            )
        with state_lock:
            final = game_state.final_weighted_score()
            state_html = game_state.render_state()
        outcome = "You won!" if final >= WIN_THRESHOLD else "You lost!"
        return (
            f"<h1>{outcome}</h1>"
            f"{state_html}"
            "<form method='post' action='/reset'>"
            "<button type='submit'>Reset</button>"
            "</form>"
            f"{footer}"
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
