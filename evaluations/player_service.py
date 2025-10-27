"""Flask web service to configure and launch automated players."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import html
import json
import logging
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

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
from evaluations.game_database import GameDatabaseRecorder
from evaluations.player_manager import PlayerManager
from evaluations.players import (
    ActionFirstRandomPlayer,
    GeminiCivilSocietyPlayer,
    GeminiCorporationPlayer,
    Player,
    RandomPlayer,
)
from evaluations.sqlite3_connector import SQLiteConnector
from rpg.assessment_agent import AssessmentAgent
from rpg.config import load_game_config


logger = logging.getLogger(__name__)


class _ClosingGameDatabaseRecorder(GameDatabaseRecorder):
    """Game recorder that closes its connector after each run."""

    def _close_connector(self) -> None:
        connector = getattr(self, "_connector", None)
        if connector is not None:
            try:
                connector.close()
            finally:
                self._connector = None  # type: ignore[assignment]

    def on_game_end(
        self,
        state,
        *,
        result,
        successful,
        error=None,
        log_warning_count=0,
        log_error_count=0,
    ):
        try:
            super().on_game_end(
                state,
                result=result,
                successful=successful,
                error=error,
                log_warning_count=log_warning_count,
                log_error_count=log_error_count,
            )
        finally:
            self._close_connector()

    def on_game_error(
        self,
        state,
        error,
        *,
        log_warning_count=0,
        log_error_count=0,
    ):
        try:
            super().on_game_error(
                state,
                error,
                log_warning_count=log_warning_count,
                log_error_count=log_error_count,
            )
        finally:
            self._close_connector()


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
        ("action-first", "Action-first opportunist"),
        ("civil-society", "Civil society strategist"),
        ("corporation", "Corporation advocate"),
    ]
    player_label_lookup = dict(player_choices)

    def _build_player(
        characters: Iterable,
        player_key: str,
        player_config: Any,
    ) -> Player:
        corporation_context = next(
            (c.base_context for c in characters if getattr(c, "faction", None) == "Corporations"),
            "",
        )
        if player_key == "random":
            return RandomPlayer()
        if player_key == "action-first":
            return ActionFirstRandomPlayer()
        if player_key == "civil-society":
            model = None
            if isinstance(player_config, dict):
                model = player_config.get("model")
            elif isinstance(player_config, str) and player_config:
                model = player_config
            if model:
                return GeminiCivilSocietyPlayer(model=str(model))
            return GeminiCivilSocietyPlayer()
        if player_key == "corporation":
            model = None
            if isinstance(player_config, dict):
                model = player_config.get("model")
            elif isinstance(player_config, str) and player_config:
                model = player_config
            if model:
                return GeminiCorporationPlayer(corporation_context, model=str(model))
            return GeminiCorporationPlayer(corporation_context)
        raise KeyError(f"Unknown player key '{player_key}'")

    def _normalise_player_config(value: Any) -> tuple[Any | None, str, str]:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None, "Default", ""
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return text, text, text
            else:
                if isinstance(parsed, dict):
                    label = json.dumps(parsed, sort_keys=True)
                elif isinstance(parsed, list):
                    label = json.dumps(parsed)
                else:
                    label = str(parsed)
                return parsed, label, text
        if value is None:
            return None, "Default", ""
        if isinstance(value, dict):
            label = json.dumps(value, sort_keys=True)
            text = json.dumps(value)
            return value, label, text
        if isinstance(value, list):
            label = json.dumps(value)
            text = json.dumps(value)
            return value, label, text
        label = str(value)
        return value, label, label

    def _parse_positive_int(value: Any, default: int) -> int:
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            candidate = default
        return max(1, candidate)

    def _resolve_scenario(requested: str | None, fallback: str) -> str:
        if not requested:
            return fallback
        key = requested.strip().lower()
        if not key:
            return fallback
        if key not in scenario_lookup:
            message = f"Unknown scenario '{requested}'"
            logger.error(message)
            raise ValueError(message)
        return scenario_lookup[key]
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
    completed_sequences: List[Dict[str, Any]] = []
    known_logs: Set[str] = set()
    last_form: Dict[str, Any] = {
        "player": player_choices[0][0],
        "rounds": 10,
        "games": 1,
        "scenario": current_scenario_name,
        "log_to_db": False,
        "player_config": "",
        "parallel_runs": 1,
    }
    last_batch_config = ""
    last_scenario = current_scenario_name

    @app.route("/", methods=["GET", "POST"])
    def index():
        nonlocal completed_sequences, known_logs, last_form, current_scenario_name
        nonlocal last_batch_config, last_scenario
        if request.method == "POST":
            default_player = request.form.get(
                "player", last_form.get("player", player_choices[0][0])
            )
            if default_player not in player_label_lookup:
                logger.warning(
                    "Unknown player key '%s'; defaulting to random", default_player
                )
                default_player = "random"
            try:
                default_scenario = _resolve_scenario(
                    request.form.get("scenario"),
                    last_form.get("scenario", current_scenario_name),
                )
            except ValueError as exc:
                abort(400, description=str(exc))
            default_games = _parse_positive_int(
                request.form.get("games", last_form.get("games", 1)), 1
            )
            default_rounds = _parse_positive_int(
                request.form.get("rounds", last_form.get("rounds", 10)), 10
            )
            parallel_runs = _parse_positive_int(
                request.form.get(
                    "parallel_runs", last_form.get("parallel_runs", 1)
                ),
                1,
            )
            default_log_to_db = bool(request.form.get("log_to_db"))
            (
                default_config,
                default_config_label,
                default_config_input,
            ) = _normalise_player_config(
                request.form.get("player_config", last_form.get("player_config", ""))
            )
            batch_raw = request.form.get("batch_runs", "").strip()
            run_requests: List[Dict[str, Any]] = []
            if batch_raw:
                last_batch_config = batch_raw
                try:
                    batch_payload = json.loads(batch_raw)
                except json.JSONDecodeError:
                    logger.error("Invalid batch configuration JSON provided")
                    abort(400, description="Invalid batch configuration JSON")
                if not isinstance(batch_payload, list) or not batch_payload:
                    abort(400, description="Batch configuration must be a non-empty list")
                for entry in batch_payload:
                    if not isinstance(entry, dict):
                        abort(400, description="Each batch configuration must be an object")
                    entry_player = entry.get("player", default_player)
                    if entry_player not in player_label_lookup:
                        logger.warning(
                            "Unknown player key '%s'; defaulting to random", entry_player
                        )
                        entry_player = "random"
                    try:
                        entry_scenario = _resolve_scenario(
                            entry.get("scenario"), default_scenario
                        )
                    except ValueError as exc:
                        abort(400, description=str(exc))
                    entry_games = _parse_positive_int(
                        entry.get("games", default_games), default_games
                    )
                    entry_rounds = _parse_positive_int(
                        entry.get("rounds", default_rounds), default_rounds
                    )
                    raw_log_to_db = entry.get("log_to_db", default_log_to_db)
                    if isinstance(raw_log_to_db, str):
                        entry_log_to_db = raw_log_to_db.lower() in {
                            "1",
                            "true",
                            "yes",
                            "on",
                        }
                    else:
                        entry_log_to_db = bool(raw_log_to_db)
                    entry_config_value = entry.get("player_config", default_config)
                    (
                        entry_config,
                        entry_config_label,
                        entry_config_input,
                    ) = _normalise_player_config(entry_config_value)
                    run_requests.append(
                        {
                            "player": entry_player,
                            "player_label": player_label_lookup.get(
                                entry_player, entry_player
                            ),
                            "scenario": entry_scenario,
                            "games": entry_games,
                            "rounds": entry_rounds,
                            "log_to_db": entry_log_to_db,
                            "player_config": entry_config,
                            "player_config_label": entry_config_label,
                            "player_config_input": entry_config_input,
                        }
                    )
            else:
                last_batch_config = ""
                run_requests.append(
                    {
                        "player": default_player,
                        "player_label": player_label_lookup.get(
                            default_player, default_player
                        ),
                        "scenario": default_scenario,
                        "games": default_games,
                        "rounds": default_rounds,
                        "log_to_db": default_log_to_db,
                        "player_config": default_config,
                        "player_config_label": default_config_label,
                        "player_config_input": default_config_input,
                    }
                )

            new_known_logs: Set[str] = set()

            def _execute_run(
                index: int, run_config: Dict[str, Any]
            ) -> Tuple[int, Dict[str, Any], Set[str]]:
                scenario_key = run_config["scenario"]
                characters = load_characters(scenario_name=scenario_key)
                player_instance = _build_player(
                    characters, run_config["player"], run_config["player_config"]
                )
                game_observer_factory = None
                if run_config["log_to_db"]:

                    def game_observer_factory(
                        _state, _player_instance, selected_key, game_number
                    ):
                        notes = f"{selected_key}-{scenario_key}-game-{game_number}"
                        connector = SQLiteConnector()
                        return _ClosingGameDatabaseRecorder(connector, notes=notes)

                manager = PlayerManager(
                    characters,
                    assessor,
                    log_dir,
                    game_observer_factory=game_observer_factory,
                    scenario=scenario_key,
                )
                logger.info(
                    "Starting sequence for player %s on scenario %s: %d games with %d rounds each",
                    run_config["player"],
                    scenario_key,
                    run_config["games"],
                    run_config["rounds"],
                )
                try:
                    game_results = manager.run_sequence(
                        run_config["player"],
                        player_instance,
                        run_config["games"],
                        run_config["rounds"],
                    )
                except Exception as exc:  # pragma: no cover - defensive fallback
                    logger.exception(
                        "Run for player %s on scenario %s failed",
                        run_config["player"],
                        scenario_key,
                    )
                    summary = {
                        "player": run_config["player"],
                        "player_label": run_config["player_label"],
                        "player_config_label": run_config["player_config_label"],
                        "player_config_input": run_config["player_config_input"],
                        "rounds": run_config["rounds"],
                        "games": run_config["games"],
                        "scenario": scenario_key,
                        "scenario_label": _format_scenario_label(scenario_key),
                        "log_to_db": run_config["log_to_db"],
                        "results": [
                            {
                                "game_number": 0,
                                "rounds_completed": 0,
                                "result": "Error",
                                "log_filename": None,
                                "error": str(exc),
                            }
                        ],
                    }
                    return index, summary, set()

                condensed_results: List[Dict[str, Any]] = []
                log_names: Set[str] = set()
                for entry in game_results:
                    filename = entry.get("log_filename")
                    if filename:
                        log_names.add(filename)
                    condensed_results.append(
                        {
                            "game_number": entry.get("game_number"),
                            "rounds_completed": entry.get("iterations", 0),
                            "result": entry.get("result"),
                            "log_filename": filename,
                            "log_warning_count": entry.get("log_warning_count", 0),
                            "log_error_count": entry.get("log_error_count", 0),
                            "error": entry.get("error"),
                        }
                    )

                summary = {
                    "player": run_config["player"],
                    "player_label": run_config["player_label"],
                    "player_config_label": run_config["player_config_label"],
                    "player_config_input": run_config["player_config_input"],
                    "rounds": run_config["rounds"],
                    "games": run_config["games"],
                    "scenario": scenario_key,
                    "scenario_label": _format_scenario_label(scenario_key),
                    "log_to_db": run_config["log_to_db"],
                    "results": condensed_results,
                }
                return index, summary, log_names

            indexed_results: List[Tuple[int, Dict[str, Any], Set[str]]] = []
            with ThreadPoolExecutor(max_workers=parallel_runs) as executor:
                future_map = {
                    executor.submit(_execute_run, idx, config): idx
                    for idx, config in enumerate(run_requests)
                }
                for future in as_completed(future_map):
                    indexed_results.append(future.result())

            indexed_results.sort(key=lambda item: item[0])
            run_summaries: List[Dict[str, Any]] = []
            for _, summary, log_names in indexed_results:
                run_summaries.append(summary)
                new_known_logs.update(log_names)

            completed_sequences = run_summaries
            known_logs = new_known_logs
            if run_requests:
                first = run_requests[0]
                if run_summaries:
                    current_scenario_name = run_summaries[0]["scenario"]
                    last_scenario = run_summaries[0]["scenario"]
                last_form = {
                    "player": first["player"],
                    "rounds": first["rounds"],
                    "games": first["games"],
                    "scenario": first.get("scenario", current_scenario_name),
                    "log_to_db": first["log_to_db"],
                    "player_config": first["player_config_input"],
                    "parallel_runs": parallel_runs,
                }
            return redirect("/progress")
        logger.info("Showing main player selection page")
        selected_player = last_form.get("player", player_choices[0][0])
        player_options = "".join(
            "<option value='{key}'{selected}>{label}</option>".format(
                key=key,
                label=label,
                selected=" selected" if key == selected_player else "",
            )
            for key, label in player_choices
        )
        selected_scenario = last_form.get("scenario", current_scenario_name)
        log_to_db_checked = " checked" if last_form.get("log_to_db") else ""
        scenario_options = "".join(
            "<option value='{value}'{selected}>{label}</option>".format(
                value=name,
                label=_format_scenario_label(name),
                selected=" selected" if name == selected_scenario else "",
            )
            for name in scenario_names
        )
        player_config_value = html.escape(str(last_form.get("player_config", "")))
        games_value = html.escape(str(last_form.get("games", 1)))
        rounds_value = html.escape(str(last_form.get("rounds", 10)))
        parallel_value = html.escape(str(last_form.get("parallel_runs", 1)))
        batch_value = html.escape(last_batch_config)
        return (
            "<h1>Automated Player Manager</h1>"
            "<form method='post'>"
            f"<label>Scenario: <select name='scenario'>{scenario_options}</select></label><br>"
            f"<label>Player: <select name='player'>{player_options}</select></label><br>"
            f"<label>Player configuration: <input type='text' name='player_config' value='{player_config_value}'></label><br>"
            f"<label>Games: <input type='number' min='1' name='games' value='{games_value}'></label><br>"
            f"<label>Rounds per game: <input type='number' min='1' name='rounds' value='{rounds_value}'></label><br>"
            f"<label>Max parallel runs: <input type='number' min='1' name='parallel_runs' value='{parallel_value}'></label><br>"
            f"<label><input type='checkbox' name='log_to_db' value='1'{log_to_db_checked}> Log games to SQLite</label><br>"
            "<label>Batch runs (JSON list, optional):<br>"
            f"<textarea name='batch_runs' rows='6' cols='60'>{batch_value}</textarea></label><br>"
            "<button type='submit'>Run Sequence</button>"
            "</form>"
            "<h2>Evaluations</h2>"
            "<form action='/evaluation/baseline' method='post'><button type='submit'>Baseline Assessment</button></form>"
            "<form action='/evaluation/consistency' method='post'><button type='submit'>Consistency Assessment</button></form>"
        )

    @app.route("/progress", methods=["GET"])
    def show_progress():
        logger.info("Showing progress page")
        if not completed_sequences:
            return "<h1>No games have been run yet.</h1><a href='/'>Back</a>"
        summary_parts = [
            "<h1>Player Manager Progress</h1>",
            f"<div>Total configured runs: {len(completed_sequences)}</div>",
            "<ul>",
        ]
        for idx, run in enumerate(completed_sequences, start=1):
            log_flag = "Yes" if run.get("log_to_db") else "No"
            summary_parts.append(
                (
                    "<li><strong>Run {idx}:</strong> {player_label} on {scenario_label} "
                    "(games: {games}, rounds/game: {rounds}, log to DB: {log_flag})"
                ).format(
                    idx=idx,
                    player_label=html.escape(str(run.get("player_label", run.get("player", "")))),
                    scenario_label=html.escape(str(run.get("scenario_label", run.get("scenario", "")))),
                    games=run.get("games"),
                    rounds=run.get("rounds"),
                    log_flag=log_flag,
                )
            )
            results = run.get("results", [])
            if results:
                summary_parts.append("<ul>")
                for entry in results:
                    game_number = entry.get("game_number")
                    game_label = html.escape(str(game_number if game_number is not None else "?"))
                    rounds_completed = int(entry.get("rounds_completed", 0) or 0)
                    result = html.escape(str(entry.get("result", "Pending")))
                    error = entry.get("error")
                    error_html = (
                        f" — Error: {html.escape(str(error))}" if error else ""
                    )
                    filename = entry.get("log_filename")
                    log_link = (
                        f" — <a href='/logs/{html.escape(filename)}'>Log</a>"
                        if filename
                        else ""
                    )
                    summary_parts.append(
                        (
                            f"<li>Game {game_label}: {rounds_completed} rounds completed ({result})"
                            f"{error_html}{log_link}</li>"
                        )
                    )
                summary_parts.append("</ul>")
            summary_parts.append("</li>")
        summary_parts.append("</ul>")
        summary_parts.append("<a href='/'>Back</a>")
        return "".join(summary_parts)

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
        scenario = last_scenario if completed_sequences else current_scenario_name
        result = run_baseline_assessment(scenario_name=scenario)
        return (
            "<h1>Baseline Assessment</h1>"
            f"<pre>{result}</pre>"
            "<a href='/'>Back</a>"
        )

    @app.route("/evaluation/consistency", methods=["POST"])
    def consistency_evaluation():
        scenario = last_scenario if completed_sequences else current_scenario_name
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
