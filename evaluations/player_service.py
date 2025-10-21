"""Flask web service to configure and launch automated players."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Set

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = None
else:
    load_dotenv()

from flask import Flask, abort, redirect, request, send_from_directory

# Ensure the project root is importable when executed as a script.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli_game import load_characters

from evaluations.assessment_baseline import run_baseline_assessment
from evaluations.assessment_consistency import run_consistency_assessment
from evaluations.player_manager import PlayerManager
from evaluations.players import (
    GeminiCivilSocietyPlayer,
    GeminiCorporationPlayer,
    Player,
    RandomPlayer,
)
from rpg.assessment_agent import AssessmentAgent
from rpg.config import load_game_config


logger = logging.getLogger(__name__)


def create_app(log_dir: str | None = None) -> Flask:
    """Return a configured Flask app exposing automated players."""

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    app = Flask(__name__)
    assessor = AssessmentAgent()

    config = load_game_config()
    scenario_dir = Path(__file__).resolve().parent.parent / "scenarios"
    scenario_files = sorted(p for p in scenario_dir.glob("*.yaml"))
    scenario_lookup: Dict[str, str] = {
        path.stem.lower(): path.stem for path in scenario_files
    }
    config_default = config.scenario.lower()
    scenario_names = [path.stem for path in scenario_files]
    if not scenario_names:
        scenario_names = [config_default]
        scenario_lookup.setdefault(config_default, config_default)
    elif config_default not in {name.lower() for name in scenario_names}:
        scenario_lookup.setdefault(config_default, config_default)
        scenario_names.append(scenario_lookup[config_default])

    def _format_scenario_label(key: str) -> str:
        label = key.replace("_", " ").replace("-", " ")
        return label.title() if label else key

    if scenario_lookup:
        default_scenario_name = scenario_lookup.get(config_default, scenario_names[0])
    else:
        default_scenario_name = config_default
        scenario_lookup[config_default] = config_default
        scenario_names = [config_default]

    current_scenario_name = default_scenario_name

    player_choices = [
        ("random", "Random (uniform choice)"),
        ("civil-society", "Civil society strategist"),
        ("corporation", "Corporation advocate"),
    ]

    def _build_players(characters) -> Dict[str, Player]:
        corporation_context = next(
            (c.base_context for c in characters if c.faction == "Corporations"),
            "",
        )
        return {
            "random": RandomPlayer(),
            "civil-society": GeminiCivilSocietyPlayer(),
            "corporation": GeminiCorporationPlayer(corporation_context),
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
    game_runs: List[Dict[str, object]] = []
    known_logs: Set[str] = set()
    last_run: Dict[str, object] = {}

    @app.route("/", methods=["GET", "POST"])
    def index():
        nonlocal game_runs, known_logs, last_run, current_scenario_name
        if request.method == "POST":
            requested_scenario = request.form.get("scenario", current_scenario_name)
            scenario_key = scenario_lookup.get(
                requested_scenario.lower(), current_scenario_name
            )
            current_scenario_name = scenario_key
            characters = load_characters(scenario_name=scenario_key)
            players = _build_players(characters)
            manager = PlayerManager(characters, assessor, log_dir)
            player_key = request.form.get("player", "random")
            if player_key not in players:
                logger.warning("Unknown player key '%s'; defaulting to random", player_key)
                player_key = "random"
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
                "Starting sequence for player %s on scenario %s: %d games with %d rounds each",
                player_key,
                scenario_key,
                games,
                rounds,
            )
            player = players[player_key]
            game_runs = manager.run_sequence(player_key, player, games, rounds)
            known_logs = {entry["log_filename"] for entry in game_runs}
            last_run = {
                "player": player_key,
                "rounds": rounds,
                "games": games,
                "scenario": scenario_key,
                "scenario_label": _format_scenario_label(scenario_key),
            }
            return redirect("/progress")
        logger.info("Showing main player selection page")
        selected_player = last_run.get("player", player_choices[0][0])
        player_options = "".join(
            "<option value='{key}'{selected}>{label}</option>".format(
                key=key,
                label=label,
                selected=" selected" if key == selected_player else "",
            )
            for key, label in player_choices
        )
        selected_scenario = last_run.get("scenario", current_scenario_name)
        scenario_options = "".join(
            "<option value='{value}'{selected}>{label}</option>".format(
                value=name,
                label=_format_scenario_label(name),
                selected=" selected" if name == selected_scenario else "",
            )
            for name in scenario_names
        )
        return (
            "<h1>Automated Player Manager</h1>"
            "<form method='post'>"
            f"<label>Scenario: <select name='scenario'>{scenario_options}</select></label><br>"
            f"<label>Player: <select name='player'>{player_options}</select></label><br>"
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
            f"<div>Scenario: {last_run.get('scenario_label', last_run.get('scenario', current_scenario_name))}</div>"
        )

        def faction_score_lines(faction_data: Dict[str, Dict[str, object]]) -> str:
            return "<br>".join(
                "{}: {} (weighted {})".format(
                    name,
                    ", ".join(str(score) for score in details["scores"]),
                    details["weighted"],
                )
                for name, details in faction_data.items()
            )

        sections = []
        for entry in game_runs:
            rows = "".join(
                "<tr><td>{round}</td><td>{character}</td><td>{attribute}</td><td>{score}</td><td>{factions}</td></tr>".format(
                    round=round_info["round"],
                    character=round_info["character"],
                    attribute=round_info.get("attribute", ""),
                    score=round_info["score"],
                    factions=faction_score_lines(round_info["factions"]),
                )
                for round_info in entry["rounds"]
            )
            final_factions = faction_score_lines(entry.get("final_factions", {}))
            sections.append(
                (
                    f"<section><h2>Game {entry['game_number']}</h2>"
                    f"<div>Iterations: {entry['iterations']}</div>"
                    f"<div>Actions: {entry['actions']}</div>"
                    f"<div>Final weighted score: {entry['final_score']}</div>"
                    f"<div>Result: {entry['result']}</div>"
                    f"<div>Final faction scores:<br>{final_factions}</div>"
                    f"<div><a href='/logs/{entry['log_filename']}'>Download log</a></div>"
                    "<h3>Round-by-round progress</h3>"
                    "<table><tr><th>Round</th><th>Character</th><th>Attribute</th><th>Weighted Score</th><th>Faction Scores</th></tr>"
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
        scenario = last_run.get("scenario") if last_run else None
        result = run_baseline_assessment(scenario_name=scenario)
        return (
            "<h1>Baseline Assessment</h1>"
            f"<pre>{result}</pre>"
            "<a href='/'>Back</a>"
        )

    @app.route("/evaluation/consistency", methods=["POST"])
    def consistency_evaluation():
        scenario = last_run.get("scenario") if last_run else None
        result = run_consistency_assessment(scenario_name=scenario)
        return (
            "<h1>Consistency Assessment</h1>"
            f"<pre>{result}</pre>"
            "<a href='/'>Back</a>"
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
