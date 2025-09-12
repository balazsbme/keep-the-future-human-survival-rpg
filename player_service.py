"""Flask web service to configure and launch automated players."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from typing import Dict, List
import logging
import os

from flask import Flask, request, redirect

from cli_game import load_characters
from rpg.game_state import GameState
from rpg.assessment_agent import AssessmentAgent
from players import RandomPlayer, GeminiWinPlayer, GeminiGovCorpPlayer, Player
from evaluations.assessment_baseline import run_baseline_assessment
from evaluations.assessment_consistency import run_consistency_assessment


logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Return a configured Flask app exposing automated players."""
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    app = Flask(__name__)
    characters = load_characters()
    state = GameState(list(characters))
    assessor = AssessmentAgent()
    players: Dict[str, Player] = {
        "random": RandomPlayer(),
        "gemini-win": GeminiWinPlayer(),
        "gemini-govcorp": GeminiGovCorpPlayer(
            next((c.base_context for c in characters if c.name == "Governments"), ""),
            next((c.base_context for c in characters if c.name == "Corporations"), ""),
        ),
    }
    progress: List[Dict[str, object]] = []

    @app.route("/", methods=["GET", "POST"])
    def index():
        nonlocal state
        if request.method == "POST":
            player_key = request.form["player"]
            rounds = int(request.form.get("rounds", "10"))
            logger.info("Starting new game with player %s for %d rounds", player_key, rounds)
            state = GameState(list(characters))
            progress.clear()
            player = players[player_key]
            for round_num in range(1, rounds + 1):
                logger.info("Beginning round %d", round_num)
                player.take_turn(state, assessor)
                progress.append(
                    {
                        "round": round_num,
                        "actor": state.history[-1][0] if state.history else "",
                        "score": state.final_weighted_score(),
                        "actors": {
                            name: state._actor_weighted_score(name)
                            for name in state.progress
                        },
                    }
                )
                logger.info(
                    "Round %d result: actor=%s score=%s",
                    round_num,
                    progress[-1]["actor"],
                    progress[-1]["score"],
                )
                if state.final_weighted_score() >= 80:
                    logger.info("Final score threshold reached; ending game")
                    break
            logger.info(
                "Game finished after %d rounds with final score %s",
                len(progress),
                state.final_weighted_score(),
            )
            return redirect("/progress")
        logger.info("Showing main player selection page")
        options = "".join(
            f'<option value="{key}">{key}</option>' for key in players
        )
        return (
            "<h1>Automated Players</h1>"
            "<form method='post'>"
            f"<label>Player: <select name='player'>{options}</select></label><br>"
            "<label>Rounds: <input name='rounds' value='10'></label><br>"
            "<button type='submit'>Start</button>"
            "</form>"
            "<h2>Evaluations</h2>"
            "<form action='/evaluation/baseline' method='post'><button type='submit'>Baseline Assessment</button></form>"
            "<form action='/evaluation/consistency' method='post'><button type='submit'>Consistency Assessment</button></form>"
        )

    @app.route("/progress", methods=["GET"])
    def show_progress():
        logger.info("Showing progress page")
        rows = "".join(
            "<tr><td>{round}</td><td>{actor}</td><td>{score}</td><td>{actors}</td></tr>".format(
                round=entry["round"],
                actor=entry["actor"],
                score=entry["score"],
                actors=", ".join(
                    f"{name}: {score}" for name, score in entry["actors"].items()
                ),
            )
            for entry in progress
        )
        final_score = state.final_weighted_score()
        result = "Win" if final_score >= 80 else "Lose"
        summary = (
            "<h1>Game Status</h1>"
            f"<div>Iterations: {len(progress)}</div>"
            f"<div>Actions: {len(state.history)}</div>"
            f"<div>Current weighted final score: {final_score}</div>"
            f"<div>Result: {result}</div>"
            "<h2>Game Progress</h2>"
        )
        return (
            summary
            + "<table><tr><th>Round</th><th>Actor</th><th>Weighted Score</th><th>Actor Scores</th></tr>"
            + rows
            + "</table>"
            + f"<div>Final weighted score: {final_score}</div>"
            + "<a href='/'>Back</a>"
        )

    @app.route("/evaluation/baseline", methods=["POST"])
    def baseline_evaluation():
        result = run_baseline_assessment()
        return (
            "<h1>Baseline Assessment</h1>"
            f"<pre>{result}</pre>"
            "<a href='/'>Back</a>"
        )

    @app.route("/evaluation/consistency", methods=["POST"])
    def consistency_evaluation():
        result = run_consistency_assessment()
        return (
            "<h1>Consistency Assessment</h1>"
            f"<pre>{result}</pre>"
            "<a href='/'>Back</a>"
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
