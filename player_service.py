"""Flask web service to configure and launch automated players."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from typing import Dict, List, Set
import logging
import os
import tempfile

from flask import Flask, abort, redirect, request, send_from_directory

from cli_game import load_characters
from evaluations.assessment_baseline import run_baseline_assessment
from evaluations.assessment_consistency import run_consistency_assessment
from player_manager import PlayerManager
from players import GeminiGovCorpPlayer, GeminiWinPlayer, RandomPlayer
from rpg.assessment_agent import AssessmentAgent


logger = logging.getLogger(__name__)


def create_app(log_dir: str | None = None) -> Flask:
    """Return a configured Flask app exposing automated players."""

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    app = Flask(__name__)
    characters = load_characters()
    assessor = AssessmentAgent()
    players: Dict[str, Player] = {
        "random": RandomPlayer(),
        "gemini-win": GeminiWinPlayer(),
        "gemini-govcorp": GeminiGovCorpPlayer(
            next((c.base_context for c in characters if c.name == "Governments"), ""),
            next((c.base_context for c in characters if c.name == "Corporations"), ""),
        ),
    }
    if log_dir is None:
        try:
            os.makedirs(app.instance_path, exist_ok=True)
            log_dir = os.path.join(app.instance_path, "player_logs")
        except PermissionError:
            fallback_instance = os.path.join(
                tempfile.gettempdir(), "keep-the-future-human-instance"
            )
            os.makedirs(fallback_instance, exist_ok=True)
            log_dir = os.path.join(fallback_instance, "player_logs")
            logger.warning(
                "Falling back to temporary instance directory at %s due to permission error",
                fallback_instance,
            )
    os.makedirs(log_dir, exist_ok=True)
    app.config["PLAYER_LOG_DIR"] = log_dir
    manager = PlayerManager(characters, assessor, log_dir)
    game_runs: List[Dict[str, object]] = []
    known_logs: Set[str] = set()
    last_run: Dict[str, object] = {}

    @app.route("/", methods=["GET", "POST"])
    def index():
        nonlocal game_runs, known_logs, last_run
        if request.method == "POST":
            player_key = request.form["player"]
            try:
                rounds = int(request.form.get("rounds", "10"))
            except ValueError:
                rounds = 10
            try:
                games = int(request.form.get("games", "1"))
            except ValueError:
                games = 1
            rounds = max(1, rounds)
            games = max(1, games)
            logger.info(
                "Starting sequence for player %s: %d games with %d rounds each",
                player_key,
                games,
                rounds,
            )
            player = players[player_key]
            game_runs = manager.run_sequence(player_key, player, games, rounds)
            known_logs = {entry["log_filename"] for entry in game_runs}
            last_run = {"player": player_key, "rounds": rounds, "games": games}
            return redirect("/progress")
        logger.info("Showing main player selection page")
        options = "".join(
            f'<option value="{key}">{key}</option>' for key in players
        )
        return (
            "<h1>Automated Player Manager</h1>"
            "<form method='post'>"
            f"<label>Player: <select name='player'>{options}</select></label><br>"
            "<label>Games: <input type='number' min='1' name='games' value='1'></label><br>"
            "<label>Rounds per game: <input type='number' min='1' name='rounds' value='10'></label><br>"
            "<button type='submit'>Run Sequence</button>"
            "</form>"
            "<h2>Evaluations</h2>"
            "<form action='/evaluation/baseline' method='post'><button type='submit'>Baseline Assessment</button></form>"
            "<form action='/evaluation/consistency' method='post'><button type='submit'>Consistency Assessment</button></form>"
        )

    @app.route("/progress", methods=["GET"])
    def show_progress():
        logger.info("Showing progress page")
        if not game_runs:
            return "<h1>No games have been run yet.</h1><a href='/'>Back</a>"
        summary = (
            "<h1>Player Manager Progress</h1>"
            f"<div>Selected player: {last_run.get('player', '')}</div>"
            f"<div>Games requested: {last_run.get('games', 0)}</div>"
            f"<div>Rounds per game: {last_run.get('rounds', 0)}</div>"
        )

        def actor_score_lines(actor_data: Dict[str, Dict[str, object]]) -> str:
            return "<br>".join(
                "{}: {} (weighted {})".format(
                    name,
                    ", ".join(str(score) for score in details["scores"]),
                    details["weighted"],
                )
                for name, details in actor_data.items()
            )

        sections = []
        for entry in game_runs:
            rows = "".join(
                "<tr><td>{round}</td><td>{actor}</td><td>{score}</td><td>{actors}</td></tr>".format(
                    round=round_info["round"],
                    actor=round_info["actor"],
                    score=round_info["score"],
                    actors=actor_score_lines(round_info["actors"]),
                )
                for round_info in entry["rounds"]
            )
            final_actors = actor_score_lines(entry["final_actors"])
            sections.append(
                (
                    f"<section><h2>Game {entry['game_number']}</h2>"
                    f"<div>Iterations: {entry['iterations']}</div>"
                    f"<div>Actions: {entry['actions']}</div>"
                    f"<div>Final weighted score: {entry['final_score']}</div>"
                    f"<div>Result: {entry['result']}</div>"
                    f"<div>Final actor scores:<br>{final_actors}</div>"
                    f"<div><a href='/logs/{entry['log_filename']}'>Download log</a></div>"
                    "<h3>Round-by-round progress</h3>"
                    "<table><tr><th>Round</th><th>Actor</th><th>Weighted Score</th><th>Actor Scores</th></tr>"
                    + rows
                    + "</table></section>"
                )
            )

        return summary + "".join(sections) + "<a href='/'>Back</a>"

    @app.route("/logs/<path:filename>", methods=["GET"])
    def download_log(filename: str):
        if filename not in known_logs:
            logger.info("Unknown log requested: %s", filename)
            abort(404)
        log_dir = app.config["PLAYER_LOG_DIR"]
        file_path = os.path.join(log_dir, filename)
        if not os.path.isfile(file_path):
            abort(404)
        return send_from_directory(log_dir, filename, as_attachment=True)

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
