"""Flask web service exposing the RPG demo over HTTP."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
from typing import Dict, List
import logging
import os
import threading
import queue
from html import escape

from flask import Flask, request, redirect, Response

from cli_game import load_characters
from rpg.game_state import GameState, WIN_THRESHOLD
from rpg.character import ActionOption
from rpg.assessment_agent import AssessmentAgent

# Link to this project's source code repository for inclusion in the web UI footer
GITHUB_URL = "https://github.com/balazsbme/keep-the-future-human-survival-rpg"


logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Return a configured Flask application ready to serve the game.

    Returns:
        The configured Flask application.
    """
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    app = Flask(__name__)
    # Load characters once at startup and reuse for resets to avoid external
    # dependency calls during tests.
    initial_characters = load_characters()
    game_state = GameState(list(initial_characters))
    assessor = AssessmentAgent()
    enable_parallel = os.environ.get("ENABLE_PARALLELISM") == "1"
    pending_actions: Dict[
        int, queue.Queue[List[ActionOption]] | List[ActionOption]
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
        """Log the HTTP method and path for incoming requests.

        Returns:
            None.
        """
        logger.info("%s %s", request.method, request.path)

    @app.route("/", methods=["GET"])
    def main_page() -> str:
        """Display the landing page with game introduction."""
        turns = game_state.config.max_rounds
        return (
            "<h1>AI Safety Negotiation Game</h1>"
            "<p>You are an expert negotiator with connections to every major faction shaping AI governance. You can persuade their key representatives to propose and take actions. Your objective is to ensure AI is developed in humanity's best interest and keep the future human.</p>"
            f"<p>You have {turns} turns to reach a final weighted score of {WIN_THRESHOLD} or higher to win.</p>"
            "<a href='/start'>Start</a>"
            f"{footer}"
        )

    @app.route("/start", methods=["GET"])
    def list_characters() -> str:
        """Display available characters and current game state."""
        logger.info("Listing characters")
        with state_lock:
            score = game_state.final_weighted_score()
            hist_snapshot = list(game_state.history)
            characters = list(game_state.characters)
            state_html = game_state.render_state()
        if score >= WIN_THRESHOLD or len(hist_snapshot) >= game_state.config.max_rounds:
            return redirect("/result")

        if enable_parallel:
            pending_actions.clear()

            def launch(idx: int, char) -> None:
                q: queue.Queue[List[ActionOption]] = queue.Queue()
                pending_actions[idx] = q

                def worker() -> None:
                    q.put(char.generate_actions(hist_snapshot))

                threading.Thread(target=worker, daemon=True).start()

            for idx, char in enumerate(characters):
                launch(idx, char)

        options = "".join(
            f'<input type="radio" name="character" value="{idx}" id="char{idx}">'
            f'<label for="char{idx}">{escape(char.display_name, quote=False)}</label><br>'
            for idx, char in enumerate(characters)
        )
        return (
            "<h1>Keep the Future Human Survival RPG</h1>"
            "<form method='post' action='/actions'>"
            f"{options}"
            "<button type='submit'>Choose</button>"
            "</form>"
            "<form method='post' action='/reset'>"
            "<button type='submit'>Reset</button>"
            "</form>"
            f"{state_html}"
            f"{footer}"
        )
    @app.route("/actions", methods=["GET", "POST"])
    def character_actions() -> str:
        """Show actions for the selected character.

        Returns:
            HTML string listing possible actions.
        """
        char_id = int(request.values["character"])
        with state_lock:
            char = game_state.characters[char_id]
            hist_snapshot = list(game_state.history)
            state_html = game_state.render_state()
        logger.info("Generating actions for %s (%d)", char.name, char_id)
        actions: List[ActionOption]
        if enable_parallel:
            entry = pending_actions.get(char_id)
            if isinstance(entry, list):
                actions = entry
            elif isinstance(entry, queue.Queue):
                if entry.empty():
                    return (
                        "<p>Loading...</p>"
                        f"<meta http-equiv='refresh' content='1;url=/actions?character={char_id}'>"
                        f"{footer}"
                    )
                actions = entry.get()
                pending_actions[char_id] = actions
            else:
                actions = char.generate_actions(hist_snapshot)
        else:
            actions = char.generate_actions(hist_snapshot)
        logger.debug("Actions: %s", [action.text for action in actions])
        radios = "".join(
            f'<input type="radio" name="action" value="{escape(json.dumps(action.to_payload()), quote=True)}" id="a{idx}">'
            f'<label for="a{idx}">{escape(action.text, quote=False)}</label><br>'
            for idx, action in enumerate(actions)
        )
        display_name = escape(char.display_name, quote=False)
        return (
            f"<h1>{display_name}</h1>"
            f"<p>Which action do you want {display_name} to perform?</p>"
            "<form method='post' action='/perform'>"
            f"{radios}"
            f"<input type='hidden' name='character' value='{char_id}'>"
            "<button type='submit'>Send</button>"
            "</form>"
            "<a href='/start'>Back to characters</a>"
            "<form method='post' action='/reset'>"
            "<button type='submit'>Reset</button>"
            "</form>"
            f"{state_html}"
            f"{footer}"
        )

    @app.route("/perform", methods=["POST"])
    def character_perform() -> Response:
        """Carry out a character action and redirect to the start page.

        Returns:
            A redirect response to the start page.
        """
        char_id = int(request.form["character"])
        action_raw = request.form["action"]
        try:
            payload = json.loads(action_raw)
        except json.JSONDecodeError:
            selected_action = ActionOption(text=action_raw)
        else:
            if isinstance(payload, dict):
                selected_action = ActionOption.from_payload(payload)
            else:
                selected_action = ActionOption(text=str(payload))
        with state_lock:
            char = game_state.characters[char_id]
        logger.info(
            "Performing action '%s' for %s (%d)",
            selected_action.text,
            char.name,
            char_id,
        )
        with state_lock:
            game_state.record_action(char, selected_action)
            chars_snapshot = list(game_state.characters)
            history_snapshot = list(game_state.history)
            how_to_win = game_state.how_to_win

        if enable_parallel:
            def run_assessment(chars: List, htw: str, hist: List) -> None:
                try:
                    scores = assessor.assess(
                        chars,
                        htw,
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
            logger.debug("Scores: %s", scores)
            with state_lock:
                game_state.update_progress(scores)

        with state_lock:
            final_score = game_state.final_weighted_score()
            hist_len = len(game_state.history)
            max_rounds = game_state.config.max_rounds
        if final_score >= WIN_THRESHOLD or hist_len >= max_rounds:
            return redirect("/result")
        return redirect("/start")

    @app.route("/reset", methods=["POST"])
    def reset() -> Response:
        """Reset the game to its initial state."""
        nonlocal game_state
        with state_lock:
            # Recreate game state from the initially loaded characters
            game_state = GameState(list(initial_characters))
            pending_actions.clear()
        with assessment_lock:
            assessment_threads.clear()
        return redirect("/start")

    @app.route("/instructions", methods=["GET"])
    def instructions() -> str:
        """Display the how-to-win instructions."""
        with state_lock:
            content = game_state.how_to_win
        return (
            "<h1>Instructions</h1>"
            f"<pre>{content}</pre>"
            "<a href='/start'>Back to game</a>"
            f"{footer}"
        )

    @app.route("/result", methods=["GET"])
    def result() -> str:
        """Display the final game outcome."""
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
