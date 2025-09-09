"""Flask web service exposing the RPG demo over HTTP."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from typing import List
import logging
import os

from flask import Flask, request, redirect, Response

from cli_game import load_characters
from rpg.game_state import GameState
from rpg.assessment_agent import AssessmentAgent


logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Return a configured Flask application ready to serve the game.

    Returns:
        The configured Flask application.
    """
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    app = Flask(__name__)
    game_state = GameState(load_characters())
    assessor = AssessmentAgent()

    @app.before_request
    def log_request() -> None:
        """Log the HTTP method and path for incoming requests.

        Returns:
            None.
        """
        logger.info("%s %s", request.method, request.path)

    @app.route("/", methods=["GET"])
    def list_characters() -> str:
        """Display available characters and current game state.

        Returns:
            HTML string listing characters and state.
        """
        logger.info("Listing characters")
        options = "".join(
            f'<input type="radio" name="character" value="{idx}" id="char{idx}">'\
            f'<label for="char{idx}">{char.name}</label><br>'
            for idx, char in enumerate(game_state.characters)
        )
        return (
            "<form method='post' action='/actions'>"
            f"{options}"
            "<button type='submit'>Choose</button>"
            "</form>"
            f"{game_state.render_state()}"
        )

    @app.route("/actions", methods=["POST"])
    def character_actions() -> str:
        """Show actions for the selected character.

        Returns:
            HTML string listing possible actions.
        """
        char_id = int(request.form["character"])
        logger.info("Generating actions for character %d", char_id)
        char = game_state.characters[char_id]
        actions: List[str] = char.generate_actions(game_state.history)
        logger.debug("Actions: %s", actions)
        radios = "".join(
            f'<input type="radio" name="action" value="{a}" id="a{idx}">'\
            f'<label for="a{idx}">{a}</label><br>'
            for idx, a in enumerate(actions)
        )
        return (
            "<form method='post' action='/perform'>"
            f"{radios}"
            f"<input type='hidden' name='character' value='{char_id}'>"
            "<button type='submit'>Send</button>"
            "</form>"
            "<a href='/'>Back to characters</a>"
            f"{game_state.render_state()}"
        )

    @app.route("/perform", methods=["POST"])
    def character_perform() -> Response:
        """Carry out a character action and redirect to the main page.

        Returns:
            A redirect response to the root page.
        """
        char_id = int(request.form["character"])
        action = request.form["action"]
        logger.info("Performing action '%s' for character %d", action, char_id)
        char = game_state.characters[char_id]
        game_state.record_action(char, action)
        scores = assessor.assess(game_state.characters, game_state.how_to_win, game_state.history)
        logger.debug("Scores: %s", scores)
        game_state.update_progress(scores)
        return redirect("/")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
