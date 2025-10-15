"""Command-line script to run automated players."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict

from cli_game import load_characters
from rpg.game_state import GameState
from rpg.assessment_agent import AssessmentAgent
from players import (
    GeminiCivilSocietyPlayer,
    GeminiCorporationPlayer,
    RandomPlayer,
)


def create_players(characters) -> Dict[str, object]:
    """Return available player instances based on characters."""
    corp_ctx = next(
        (c.base_context for c in characters if c.faction == "Corporations"),
        "",
    )
    return {
        "random": RandomPlayer(),
        "civil-society": GeminiCivilSocietyPlayer(),
        "corporation": GeminiCorporationPlayer(corp_ctx),
    }


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--player",
        choices=["random", "civil-society", "corporation"],
        default="random",
        help="Which automated player to use",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Scenario name to load (defaults to configuration)",
    )
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds to play")
    args = parser.parse_args()

    characters = load_characters(scenario_name=args.scenario)
    state = GameState(list(characters))
    assessor = AssessmentAgent()
    players = create_players(characters)
    player = players[args.player]

    logger.info("Starting player %s for %d rounds", args.player, args.rounds)
    for round_num in range(1, args.rounds + 1):
        logger.info("Round %d", round_num)
        player.take_turn(state, assessor)
        if state.final_weighted_score() >= state.config.win_threshold:
            logger.info("Final score threshold reached; ending game")
            break

    logger.info("Game finished with final score %d", state.final_weighted_score())
    print(state.render_state())


if __name__ == "__main__":
    main()
