"""Manage running automated players across multiple sequential games."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import logging
import os
import threading
from typing import Callable, Dict, Iterable, List, Optional
from uuid import uuid4

from rpg.assessment_agent import AssessmentAgent
from rpg.game_state import GameState

from .game_database import GameRunObserver
from .players import Player


logger = logging.getLogger(__name__)

_LOGGING_LOCK = threading.RLock()
_LOGGING_STATE: Dict[str, object] = {
    "active_runs": 0,
    "original_level": None,
    "handler_levels": {},
}


def _configure_root_logger(file_handler: logging.Handler) -> None:
    """Attach ``file_handler`` capturing logs while preserving global state."""

    root_logger = logging.getLogger()
    with _LOGGING_LOCK:
        active_runs = int(_LOGGING_STATE["active_runs"])
        if active_runs == 0:
            _LOGGING_STATE["original_level"] = root_logger.level
            _LOGGING_STATE["handler_levels"] = {
                handler: handler.level for handler in root_logger.handlers
            }
            root_logger.setLevel(logging.DEBUG)
            previous_level = int(_LOGGING_STATE["original_level"])
            stream_level = (
                previous_level if previous_level != logging.NOTSET else logging.INFO
            )
            for handler in root_logger.handlers:
                if handler.level == logging.NOTSET and isinstance(
                    handler, logging.StreamHandler
                ):
                    handler.setLevel(stream_level)
        _LOGGING_STATE["active_runs"] = active_runs + 1
        root_logger.addHandler(file_handler)


def _restore_root_logger(file_handler: logging.Handler) -> None:
    """Detach ``file_handler`` and restore root logger state when idle."""

    root_logger = logging.getLogger()
    with _LOGGING_LOCK:
        if file_handler in root_logger.handlers:
            root_logger.removeHandler(file_handler)
        active_runs = max(0, int(_LOGGING_STATE["active_runs"]) - 1)
        _LOGGING_STATE["active_runs"] = active_runs
        if active_runs == 0:
            original_level = _LOGGING_STATE.get("original_level")
            if isinstance(original_level, int):
                root_logger.setLevel(original_level)
            handler_levels = _LOGGING_STATE.get("handler_levels", {})
            if isinstance(handler_levels, dict):
                for handler, level in handler_levels.items():
                    if handler in root_logger.handlers:
                        handler.setLevel(level)
            _LOGGING_STATE["handler_levels"] = {}
            _LOGGING_STATE["original_level"] = None

class PlayerManager:
    """Coordinate launching automated players for multiple games in sequence."""

    def __init__(
        self,
        characters: Iterable,
        assessor: AssessmentAgent,
        log_dir: str,
        game_observer_factory: Callable[[GameState, Player, str, int], GameRunObserver]
        | None = None,
    ) -> None:
        """Store the characters, assessor, and logging directory for runs."""

        self._characters = list(characters)
        self._assessor = assessor
        self._log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._game_observer_factory = game_observer_factory

    def run_sequence(
        self,
        player_key: str,
        player: Player,
        games: int,
        rounds: int,
    ) -> List[Dict[str, object]]:
        """Execute ``games`` sequential games with ``player``.

        Args:
            player_key: Identifier of the selected player for logging purposes.
            player: Player implementation to run.
            games: Number of games to execute sequentially.
            rounds: Maximum number of rounds per game.

        Returns:
            List of per-game progress dictionaries suitable for rendering.
        """

        results: List[Dict[str, object]] = []
        total_games = max(1, games)
        max_rounds = max(1, rounds)
        for game_index in range(1, total_games + 1):
            logger.info(
                "Launching game %d/%d for player %s", game_index, total_games, player_key
            )
            try:
                game_result = self._run_single_game(
                    player_key, player, max_rounds, game_index
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.exception(
                    "Game %d for player %s failed; recording error and continuing",
                    game_index,
                    player_key,
                )
                game_result = {
                    "game_number": game_index,
                    "rounds": [],
                    "final_score": 0,
                    "result": "Error",
                    "iterations": 0,
                    "actions": 0,
                    "log_filename": None,
                    "error": str(exc),
                }
            results.append(game_result)
        return results

    def _run_single_game(
        self,
        player_key: str,
        player: Player,
        rounds: int,
        game_index: int,
    ) -> Dict[str, object]:
        """Execute a single game and capture detailed progress information."""

        state = GameState(list(self._characters))
        observer: Optional[GameRunObserver] = None
        if self._game_observer_factory is not None:
            observer_candidate = self._game_observer_factory(
                state, player, player_key, game_index
            )
            if observer_candidate is not None:
                observer = observer_candidate
                observer.on_game_start(
                    state,
                    player_key=player_key,
                    player_class=state.player_character.__class__.__name__,
                    automated_player_class=player.__class__.__name__,
                    game_index=game_index,
                )
        log_filename = self._log_filename(player_key, game_index)
        log_path = os.path.join(self._log_dir, log_filename)
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        )
        _configure_root_logger(file_handler)

        rounds_progress: List[Dict[str, object]] = []
        try:
            for round_index in range(1, rounds + 1):
                logger.info("Beginning round %d", round_index)
                if observer is not None:
                    observer.before_turn(state, round_index)
                player.take_turn(state, self._assessor)
                if observer is not None:
                    observer.after_turn(state, round_index)
                character_label = state.history[-1][0] if state.history else ""
                attempt = state.last_action_attempt
                attribute_label = "None"
                action_text = ""
                if attempt is not None:
                    if attempt.attribute:
                        attribute_label = attempt.attribute.replace("_", " ").title()
                    else:
                        attribute_label = "None"
                    action_text = attempt.option.text
                final_score = state.final_weighted_score()
                faction_scores = {}
                for key, scores in state.progress.items():
                    label = state.faction_labels.get(key, key)
                    faction_scores[label] = {
                        "scores": list(scores),
                        "weighted": state._faction_weighted_score(key),
                    }
                rounds_progress.append(
                    {
                        "round": round_index,
                        "character": character_label,
                        "action": action_text,
                        "attribute": attribute_label,
                        "score": final_score,
                        "factions": faction_scores,
                    }
                )
                logger.info(
                    "Round %d result: character=%s score=%s",
                    round_index,
                    character_label,
                    final_score,
                )
                if final_score >= state.config.win_threshold:
                    logger.info(
                        "Final score threshold reached for game %d; ending early", game_index
                    )
                    break
        except Exception as exc:
            if observer is not None:
                observer.on_game_error(state, exc)
            raise
        finally:
            _restore_root_logger(file_handler)
            file_handler.close()

        final_score = state.final_weighted_score()
        game_result = {
            "game_number": game_index,
            "rounds": rounds_progress,
            "final_score": final_score,
            "result": "Win" if final_score >= state.config.win_threshold else "Lose",
            "iterations": len(rounds_progress),
            "actions": len(state.history),
            "log_filename": log_filename,
            "final_factions": {
                state.faction_labels.get(key, key): {
                    "scores": list(scores),
                    "weighted": state._faction_weighted_score(key),
                }
                for key, scores in state.progress.items()
            },
        }
        logger.info(
            "Game %d finished with final score %s; log saved to %s",
            game_index,
            final_score,
            log_filename,
        )
        if observer is not None:
            observer.on_game_end(
                state,
                result=game_result["result"],
                successful=True,
                error=None,
            )
        return game_result

    def _log_filename(self, player_key: str, game_index: int) -> str:
        """Return a unique log filename for a specific game run."""

        unique_id = uuid4().hex
        return f"{player_key}_game_{game_index}_{unique_id}.log"


__all__ = ["PlayerManager"]
