from __future__ import annotations

from typing import List
import os

from flask import Flask, request

from example_game import load_characters


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    characters = load_characters()

    @app.route("/", methods=["GET"])
    def list_characters() -> str:
        options = "".join(
            f'<input type="radio" name="character" value="{idx}" id="char{idx}">'
            f'<label for="char{idx}">{char.name}</label><br>'
            for idx, char in enumerate(characters)
        )
        return (
            "<form method='post' action='/questions'>"
            f"{options}"
            "<button type='submit'>Choose</button>"
            "</form>"
        )

    @app.route("/questions", methods=["POST"])
    def character_questions() -> str:
        char_id = int(request.form["character"])
        char = characters[char_id]
        questions: List[str] = char.generate_questions()
        radios = "".join(
            f'<input type="radio" name="question" value="{q}" id="q{idx}">'
            f'<label for="q{idx}">{q}</label><br>'
            for idx, q in enumerate(questions)
        )
        return (
            "<form method='post' action='/answer'>"
            f"{radios}"
            f"<input type='hidden' name='character' value='{char_id}'>"
            "<button type='submit'>Send</button>"
            "</form>"
        )

    @app.route("/answer", methods=["POST"])
    def character_answer() -> str:
        char_id = int(request.form["character"])
        question = request.form["question"]
        char = characters[char_id]
        answer = char.answer_question(question)
        return f"<p>{answer}</p>"

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
