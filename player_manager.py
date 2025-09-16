"""Manage running automated players across multiple sequential games."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, List
from uuid import uuid4

from rpg.assessment_agent import AssessmentAgent
from rpg.game_state import GameState
from players import Player


logger = logging.getLogger(__name__)


class PlayerManager:
    """Coordinate launching automated players for multiple games in sequence."""

    def __init__(
        self,
        characters: Iterable,
        assessor: AssessmentAgent,
        log_dir: str,
    ) -> None:
        """Store the characters, assessor, and logging directory for runs."""

        self._characters = list(characters)
        self._assessor = assessor
        self._log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

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
            results.append(
                self._run_single_game(player_key, player, max_rounds, game_index)
            )
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
        log_filename = self._log_filename(player_key, game_index)
        log_path = os.path.join(self._log_dir, log_filename)
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        )
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.DEBUG)

        rounds_progress: List[Dict[str, object]] = []
        try:
            for round_index in range(1, rounds + 1):
                logger.info("Beginning round %d", round_index)
                player.take_turn(state, self._assessor)
                actor = state.history[-1][0] if state.history else ""
                final_score = state.final_weighted_score()
                actor_scores = {
                    name: {
                        "scores": list(scores),
                        "weighted": state._actor_weighted_score(name),
                    }
                    for name, scores in state.progress.items()
                }
                rounds_progress.append(
                    {
                        "round": round_index,
                        "actor": actor,
                        "score": final_score,
                        "actors": actor_scores,
                    }
                )
                logger.info(
                    "Round %d result: actor=%s score=%s", round_index, actor, final_score
                )
                if final_score >= 80:
                    logger.info(
                        "Final score threshold reached for game %d; ending early", game_index
                    )
                    break
        finally:
            root_logger.removeHandler(file_handler)
            root_logger.setLevel(previous_level)
            file_handler.close()

        final_score = state.final_weighted_score()
        game_result = {
            "game_number": game_index,
            "rounds": rounds_progress,
            "final_score": final_score,
            "result": "Win" if final_score >= 80 else "Lose",
            "iterations": len(rounds_progress),
            "actions": len(state.history),
            "log_filename": log_filename,
            "final_actors": {
                name: {
                    "scores": list(scores),
                    "weighted": state._actor_weighted_score(name),
                }
                for name, scores in state.progress.items()
            },
        }
        logger.info(
            "Game %d finished with final score %s; log saved to %s",
            game_index,
            final_score,
            log_filename,
        )
        return game_result

    def _log_filename(self, player_key: str, game_index: int) -> str:
        """Return a unique log filename for a specific game run."""

        unique_id = uuid4().hex
        return f"{player_key}_game_{game_index}_{unique_id}.log"


__all__ = ["PlayerManager"]
