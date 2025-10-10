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
from rpg.game_state import GameState, WIN_THRESHOLD


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
    pending_responses: Dict[
        int, queue.Queue[List[ResponseOption]] | List[ResponseOption]
    ] = {}
    assessment_threads: List[threading.Thread] = []
    assessment_lock = threading.Lock()
    state_lock = threading.Lock()
    footer = (
        "<p><a href='/instructions'>Instructions</a> | "
        f"<a href='{GITHUB_URL}'>GitHub</a></p>"
    )

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
            pending_responses.clear()

            def launch(idx: int, char: Character, convo: Sequence[ConversationEntry]) -> None:
                if convo:
                    return
                q: queue.Queue[List[ResponseOption]] = queue.Queue()
                pending_responses[idx] = q

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
            pending_responses.pop(char_id, None)
        if enable_parallel and not conversation:
            entry = pending_responses.get(char_id)
            if isinstance(entry, list):
                return list(entry), False
            if isinstance(entry, queue.Queue):
                if entry.empty():
                    return None, True
                options = entry.get()
                pending_responses[char_id] = list(options)
                return list(options), False
        options = player.generate_responses(history, conversation, character)
        if enable_parallel and not conversation:
            pending_responses[char_id] = list(options)
        return list(options), False

    def _render_conversation(
        char_id: int,
        character: Character,
        conversation: Sequence[ConversationEntry],
        options: Sequence[ResponseOption],
        state_html: str,
    ) -> str:
        logger.debug(
            "Rendering conversation for %s with %d history items",
            character.name,
            len(conversation),
        )
        radios = "".join(
            f'<input type="radio" name="response" value="{escape(json.dumps(option.to_payload()), quote=True)}" id="opt{idx}">'  # noqa: E501
            + (
                f"<label for='opt{idx}'><strong>Action:</strong> {escape(option.text, quote=False)}</label><br>"
                if option.is_action
                else f"<label for='opt{idx}'>{escape(option.text, quote=False)}</label><br>"
            )
            for idx, option in enumerate(options)
        )
        options_html = radios if radios else "<p>No options available.</p>"
        if conversation:
            convo_items = "".join(
                f"<li><strong>{escape(entry.speaker, quote=False)}</strong>: {escape(entry.text, quote=False)}"
                f" <em>({entry.type})</em></li>"
                for entry in conversation
            )
            convo_block = f"<h2>Conversation</h2><ol>{convo_items}</ol>"
        else:
            convo_block = "<p>No conversation yet. Start by greeting the character.</p>"
        return (
            f"<h1>{escape(character.display_name, quote=False)}</h1>"
            f"{convo_block}"
            "<form method='post' action='/actions'>"
            f"{options_html}"
            f"<input type='hidden' name='character' value='{char_id}'>"
            "<button type='submit'>Send</button>"
            "</form>"
            "<a href='/start'>Back to characters</a>"
            f"{state_html}"
            f"{footer}"
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
                    game_state.record_action(character, option)
                    chars_snapshot = list(game_state.characters)
                    history_snapshot = list(game_state.history)
                    how_to_win = game_state.how_to_win
                pending_responses.pop(char_id, None)

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

            with state_lock:
                history_snapshot = list(game_state.history)
                game_state.log_player_response(character, option)
                conversation = game_state.conversation_history(character)
                player = game_state.player_character
            responses = character.generate_responses(
                history_snapshot, conversation, player
            )
            with state_lock:
                game_state.log_npc_responses(character, responses)
            pending_responses.pop(char_id, None)
            return redirect(f"/actions?character={char_id}")

        character, history, conversation, npc_actions, state_html, player = (
            _character_snapshot(char_id)
        )
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
        action_texts = {opt.text for opt in options if opt.is_action}
        for action in npc_actions:
            if action.text not in action_texts:
                options.append(action)
        page = _render_conversation(
            char_id, character, conversation, options, state_html
        )
        return Response(page)

    @app.route("/reset", methods=["POST"])
    def reset() -> Response:
        nonlocal game_state
        logger.info("Resetting game state")
        with state_lock:
            game_state = GameState(list(initial_characters))
        pending_responses.clear()
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
