"""Flask web service exposing the RPG demo over HTTP."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from typing import List
import os

from flask import Flask, request

from example_game import load_characters
from rpg.game_state import GameState


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    game_state = GameState(load_characters())

    @app.route("/", methods=["GET"])
    def list_characters() -> str:
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
        )

    @app.route("/actions", methods=["POST"])
    def character_actions() -> str:
        char_id = int(request.form["character"])
        char = game_state.characters[char_id]
        actions: List[str] = char.generate_actions()
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
        )

    @app.route("/perform", methods=["POST"])
    def character_perform() -> str:
        char_id = int(request.form["character"])
        action = request.form["action"]
        char = game_state.characters[char_id]
        result = char.perform_action(action)
        game_state.record_action(char, action)
        return f"<p>{result}</p>"

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
