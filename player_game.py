"""Command-line script to run automated players."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import argparse
from typing import Dict

from cli_game import load_characters
from rpg.game_state import GameState
from rpg.assessment_agent import AssessmentAgent
from players import RandomPlayer, GeminiWinPlayer, GeminiGovCorpPlayer


def create_players(characters) -> Dict[str, object]:
    """Return available player instances based on characters."""
    gov_ctx = next((c.base_context for c in characters if c.name == "Governments"), "")
    corp_ctx = next((c.base_context for c in characters if c.name == "Corporations"), "")
    return {
        "random": RandomPlayer(),
        "gemini-win": GeminiWinPlayer(),
        "gemini-govcorp": GeminiGovCorpPlayer(gov_ctx, corp_ctx),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--player",
        choices=["random", "gemini-win", "gemini-govcorp"],
        default="random",
        help="Which automated player to use",
    )
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds to play")
    args = parser.parse_args()

    characters = load_characters()
    state = GameState(list(characters))
    assessor = AssessmentAgent()
    players = create_players(characters)
    player = players[args.player]

    for _ in range(args.rounds):
        player.take_turn(state, assessor)
        if state.final_weighted_score() >= 80:
            break

    print(state.render_state())


if __name__ == "__main__":
    main()
