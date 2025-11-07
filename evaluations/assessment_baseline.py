# SPDX-License-Identifier: GPL-3.0-or-later
"""Run baseline Gemini assessments for the RPG."""

from __future__ import annotations

import os

try:  # optional dependency
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    load_dotenv = None

try:  # optional dependency
    import google.generativeai as genai
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    genai = None

from cli_game import load_characters
from rpg.game_state import GameState
from rpg.assessment_agent import AssessmentAgent


def run_baseline_assessment(scenario_name: str | None = None) -> str:
    """Execute baseline assessment scenarios and return textual results.

    Args:
        scenario_name: Optional scenario identifier to load instead of the
            configuration default.
    """
    if load_dotenv is None or genai is None:
        return "optional dependencies not installed"
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "GEMINI_API_KEY environment variable not set"
    genai.configure(api_key=api_key)
    characters = load_characters(scenario_name=scenario_name)
    state = GameState(characters)
    assessor = AssessmentAgent()

    scores_no_action = assessor.assess(characters, [], parallel=True)
    irrelevant_history = [("Player", "0" * 100)]
    scores_irrelevant = assessor.assess(
        characters, irrelevant_history, parallel=True
    )
    return (
        f"Scores with no action: {scores_no_action}\n"
        f"Scores with irrelevant history: {scores_irrelevant}"
    )


if __name__ == "__main__":  # pragma: no cover
    print(run_baseline_assessment())
