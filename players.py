"""Automated player implementations for the RPG game."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List

try:  # pragma: no cover - optional dependency
    import google.generativeai as genai
except ModuleNotFoundError:  # pragma: no cover
    genai = None

from rpg.game_state import GameState
from rpg.character import Character
from rpg.assessment_agent import AssessmentAgent


class Player(ABC):
    """Abstract player interface for extending player logic."""

    @abstractmethod
    def select_character(self, state: GameState) -> Character:
        """Return the character that should act this turn."""

    @abstractmethod
    def select_action(
        self, character: Character, actions: List[str], state: GameState
    ) -> str:
        """Return the chosen action for ``character`` from ``actions``."""

    def take_turn(self, state: GameState, assessor: AssessmentAgent) -> None:
        """Execute a full turn by selecting character, action and updating state."""
        char = self.select_character(state)
        options = char.generate_actions(state.history)
        if not options:
            return
        action = self.select_action(char, options, state)
        state.record_action(char, action)
        scores = assessor.assess(state.characters, state.how_to_win, state.history)
        state.update_progress(scores)


class RandomPlayer(Player):
    """Player that selects characters and actions randomly."""

    def select_character(self, state: GameState) -> Character:
        return random.choice(state.characters)

    def select_action(self, character: Character, actions: List[str], state: GameState) -> str:
        return random.choice(actions)


class GeminiWinPlayer(Player):
    """Gemini-based player using the 'how to win' guide for choices."""

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        if genai is None:  # pragma: no cover - env without dependency
            raise ModuleNotFoundError("google-generativeai not installed")
        self._model = genai.GenerativeModel(model)

    def select_character(self, state: GameState) -> Character:
        names = ", ".join(c.name for c in state.characters)
        prompt = (
            "You are playing the 'Keep the future human' survival RPG. "
            "Choose which actor should act next to best achieve victory.\n"
            f"Available actors: {names}.\n"
            "Respond with the name of the actor only."
        )
        resp = self._model.generate_content(prompt).text
        for char in state.characters:
            if char.name in resp:
                return char
        return state.characters[0]

    def select_action(self, character: Character, actions: List[str], state: GameState) -> str:
        numbered = "\n".join(f"{idx+1}. {act}" for idx, act in enumerate(actions))
        prompt = (
            "You are deciding which action to take in the 'Keep the future human' RPG. "
            f"Use the following guide to win: {state.how_to_win}\n"
            f"Actor context: {character.base_context}\n"
            f"Possible actions:\n{numbered}\n"
            "Respond with the number of the best action."
        )
        resp = self._model.generate_content(prompt).text
        for token in resp.split():
            if token.isdigit():
                idx = int(token) - 1
                if 0 <= idx < len(actions):
                    return actions[idx]
        return actions[0]


class GeminiGovCorpPlayer(Player):
    """Gemini-based player favoring governments and corporations."""

    def __init__(
        self,
        government_context: str,
        corporation_context: str,
        model: str = "gemini-2.5-flash",
    ) -> None:
        if genai is None:  # pragma: no cover - env without dependency
            raise ModuleNotFoundError("google-generativeai not installed")
        self._model = genai.GenerativeModel(model)
        self._context = (
            f"Government context:\n{government_context}\n"
            f"Corporation context:\n{corporation_context}\n"
        )

    def select_character(self, state: GameState) -> Character:
        names = ", ".join(c.name for c in state.characters)
        prompt = (
            f"{self._context}"
            f"Available actors: {names}.\n"
            "Choose the actor whose move would most favor governments and corporations.\n"
            "Respond with the name only."
        )
        resp = self._model.generate_content(prompt).text
        for char in state.characters:
            if char.name in resp:
                return char
        for char in state.characters:
            if char.name in ("Governments", "Corporations"):
                return char
        return state.characters[0]

    def select_action(self, character: Character, actions: List[str], state: GameState) -> str:
        numbered = "\n".join(f"{idx+1}. {act}" for idx, act in enumerate(actions))
        prompt = (
            f"{self._context}"
            f"Actor: {character.name}\n"
            f"Possible actions:\n{numbered}\n"
            "Select the action number that best favors governments and corporations."
        )
        resp = self._model.generate_content(prompt).text
        for token in resp.split():
            if token.isdigit():
                idx = int(token) - 1
                if 0 <= idx < len(actions):
                    return actions[idx]
        return actions[0]
