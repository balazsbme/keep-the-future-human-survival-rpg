# SPDX-License-Identifier: GPL-3.0-or-later
"""Run repeated Gemini assessments to check consistency."""

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


def run_consistency_assessment(scenario_name: str | None = None) -> str:
    """Execute the same assessment multiple times and return results.

    Args:
        scenario_name: Optional scenario identifier to load instead of the
            configuration default.
    """
    if load_dotenv is None or genai is None:
        return "optional dependencies not installed"
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    placeholder = "AIzaSyBritn92DCiuHReGBvFl16xfCi-5gQOOgk"
    if not api_key or api_key == placeholder:
        return "GEMINI_API_KEY environment variable not set"
    genai.configure(api_key=api_key)
    characters = load_characters(scenario_name=scenario_name)
    state = GameState(characters)
    assessor = AssessmentAgent()
    history = []
    results = []
    for _ in range(10):
        scores = assessor.assess(characters, state.how_to_win, history, parallel=True)
        results.append(str(scores))
    return "\n".join(results)


if __name__ == "__main__":  # pragma: no cover
    print(run_consistency_assessment())
