"""Flask web service exposing the RPG demo over HTTP."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
import logging
import os
import queue
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
        int, queue.Queue[List[ResponseOption]] | List[ResponseOption]
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
        ".player-panel{flex:3;}"
        ".conversation-panel{flex:4;}"
        ".partner-panel{flex:3;}"
        ".panel section{margin-bottom:1.5rem;}"
        ".panel h2{margin-top:0;}"
        ".context-block{white-space:pre-line;background:#f7f7f7;padding:0.75rem;border-radius:6px;}"
        ".options-form ul{list-style:none;padding:0;margin:0;}"
        ".options-form li{margin-bottom:0.75rem;}"
        "</style>"
    )

    def _profile_sections(character: Character, include_context: bool = False) -> str:
        attributes = "".join(
            f"<li><strong>{label}</strong>: {int(character.attribute_score(label.lower()))}</li>"
            for label in ("Leadership", "Technology", "Policy", "Network")
        )
        sections = [
            f"<section><h3>Attributes</h3><ul>{attributes}</ul></section>"
        ]
        detail_fields = [
            ("Faction", getattr(character, "faction", "")),
            ("Perks", getattr(character, "perks", "")),
            ("Weaknesses", getattr(character, "weaknesses", "")),
            ("Motivations", getattr(character, "motivations", "")),
            ("Background", getattr(character, "background", "")),
        ]
        detail_items = [
            f"<dt>{label}</dt><dd>{escape(str(value), quote=False)}</dd>"
            for label, value in detail_fields
            if value
        ]
        if detail_items:
            sections.append(
                "<section><h3>Profile</h3><dl>" + "".join(detail_items) + "</dl></section>"
            )
        if include_context:
            context = getattr(character, "base_context", "")
            if context:
                sections.append(
                    "<section><h3>Context</h3><div class='context-block'>"
                    + escape(str(context), quote=False).replace("\n", "<br>")
                    + "</div></section>"
                )
        return "".join(sections)

    @app.before_request
    def log_request() -> None:
        logger.info("%s %s", request.method, request.path)

    @app.route("/", methods=["GET"])
    def main_page() -> str:
        turns = game_state.config.max_rounds
        return (
            "<h1>AI Safety Negotiation Game</h1>"
            "<p>You are an expert negotiator with connections to every major faction shaping AI governance. You can persuade their key representatives to propose and take actions. Your objective is to ensure AI is developed in humanity's best interest and keep the future human.</p>"
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
        if score >= WIN_THRESHOLD or hist_len >= game_state.config.max_rounds:
            return redirect("/result")
        if enable_parallel:
            pending_player_options.clear()
            pending_npc_responses.clear()
            pending_player_choices.clear()

            def launch(idx: int, char: Character, convo: Sequence[ConversationEntry]) -> None:
                if convo:
                    return
                q: queue.Queue[List[ResponseOption]] = queue.Queue()
                pending_player_options[idx] = q

                def worker(
                    hist: Sequence[Tuple[str, str]],
                    partner: Character,
                    snapshots: Sequence[ConversationEntry],
                ) -> None:
                    try:
                        options = player.generate_responses(hist, snapshots, partner)
                    except Exception:  # pragma: no cover - defensive logging
                        logger.exception("Failed to generate player responses in background")
                        options = []
                    q.put(list(options))

                threading.Thread(
                    target=worker,
                    args=(tuple(history_snapshot), char, tuple(convo)),
                    daemon=True,
                ).start()

            for idx, (char, convo) in enumerate(
                zip(characters, conversation_snapshots)
            ):
                launch(idx, char, convo)
        options = "".join(
            f'<input type="radio" name="character" value="{idx}" id="char{idx}">'  # noqa: E501
            f'<label for="char{idx}">{escape(char.display_name, quote=False)}</label><br>'
            for idx, char in enumerate(characters)
        )
        body = (
            "<h1>Keep the Future Human Survival RPG</h1>"
            "<form method='get' action='/actions'>"
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
    ) -> Tuple[Character, List[Tuple[str, str]], List[ConversationEntry], List[ResponseOption], str, Character]:
        with state_lock:
            character = game_state.characters[char_id]
            history = list(game_state.history)
            conversation = game_state.conversation_history(character)
            available_actions = list(game_state.available_npc_actions(character))
            state_html = game_state.render_state()
            player = game_state.player_character
        return (
            character,
            history,
            conversation,
            available_actions,
            state_html,
            player,
        )

    def _resolve_player_options(
        char_id: int,
        history: Sequence[Tuple[str, str]],
        conversation: Sequence[ConversationEntry],
        character: Character,
        player: Character,
    ) -> Tuple[List[ResponseOption] | None, bool]:
        if conversation and enable_parallel:
            pending_player_options.pop(char_id, None)
        if enable_parallel and not conversation:
            entry = pending_player_options.get(char_id)
            if isinstance(entry, list):
                return list(entry), False
            if isinstance(entry, queue.Queue):
                if entry.empty():
                    return None, True
                options = entry.get()
                pending_player_options[char_id] = list(options)
                return list(options), False
        options = player.generate_responses(history, conversation, character)
        if enable_parallel and not conversation:
            pending_player_options[char_id] = list(options)
        return list(options), False

    def _option_signature(option: ResponseOption) -> str:
        payload = option.to_payload()
        return json.dumps(payload, sort_keys=True, default=str)

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
        options: Sequence[ResponseOption],
        state_html: str,
        player: Character,
    ) -> str:
        logger.debug(
            "Rendering conversation for %s with %d history items",
            character.name,
            len(conversation),
        )
        option_items = []
        for idx, option in enumerate(options):
            payload = escape(json.dumps(option.to_payload()), quote=True)
            label_text = escape(option.text, quote=False)
            if option.is_action:
                label_html = f"<strong>Action:</strong> {label_text}"
            else:
                label_html = label_text
            option_items.append(
                f"<li><input type='radio' name='response' value='{payload}' id='opt{idx}'>"
                f"<label for='opt{idx}'>{label_html}</label></li>"
            )
        if option_items:
            options_html = "<ul>" + "".join(option_items) + "</ul>"
        else:
            options_html = "<p>No options available.</p>"
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
        player_panel = (
            f"<h2>{escape(player.display_name, quote=False)}</h2>"
            + _profile_sections(player, include_context=True)
        )
        partner_panel = (
            f"<h2>{escape(character.display_name, quote=False)}</h2>"
            + _profile_sections(character, include_context=True)
        )
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
    ) -> str:
        failure_text = attempt.failure_text or (
            f"Failed '{attempt.option.text}' (attribute {attempt.attribute or 'none'}: {attempt.effective_score}, roll={attempt.roll:.2f})"
        )
        outcome_section = (
            f"<section><h2>Action Outcome</h2><p>{escape(failure_text, quote=False)}</p></section>"
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
        payload = escape(json.dumps(attempt.option.to_payload()), quote=True)
        if next_cost > 0:
            reroll_note = f"Reroll will cost {next_cost} credibility."
            reroll_label = f"Reroll (-{next_cost} credibility)"
        else:
            reroll_note = "Reroll will not cost additional credibility."
            reroll_label = "Reroll (no cost)"
        forms_section = (
            f"<section><p>{reroll_note}</p>"
            + "<form method='post' action='/reroll'>"
            + f"<input type='hidden' name='character' value='{char_id}'>"
            + f"<input type='hidden' name='action' value='{payload}'>"
            + f"<button type='submit'>{reroll_label}</button>"
            + "</form>"
            + "<form method='post' action='/finalize_failure'>"
            + f"<input type='hidden' name='character' value='{char_id}'>"
            + f"<input type='hidden' name='action' value='{payload}'>"
            + "<button type='submit'>Accept Failure</button>"
            + "</form></section>"
        )
        conversation_panel = outcome_section + convo_block + forms_section
        player_panel = (
            f"<h2>{escape(player.display_name, quote=False)}</h2>"
            + _profile_sections(player, include_context=True)
        )
        partner_panel = (
            f"<h2>{escape(character.display_name, quote=False)}</h2>"
            + _profile_sections(character, include_context=True)
        )
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
                pending_player_options.pop(char_id, None)
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
            pending_player_options.pop(char_id, None)
            return redirect(f"/actions?character={char_id}")

        character, history, conversation, npc_actions, state_html, player = (
            _character_snapshot(char_id)
        )
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
                pending_player_choices.pop(char_id, None)
                _clear_pending_npc_entries(char_id, signature)
            else:
                pending_player_choices.pop(char_id, None)
                _clear_pending_npc_entries(char_id, signature)

        options, loading = _resolve_player_options(
            char_id, history, conversation, character, player
        )
        if loading:
            body = (
                "<p>Loading...</p>"
                f"<meta http-equiv='refresh' content='1;url=/actions?character={char_id}'>"
                f"{footer}"
            )
            return Response(body)
        options = options or []
        _preload_npc_responses(
            char_id, history, conversation, options, character, player
        )
        action_texts = {opt.text for opt in options if opt.is_action}
        for action in npc_actions:
            if action.text not in action_texts:
                options.append(action)
        page = _render_conversation(
            char_id, character, conversation, options, state_html, player
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
