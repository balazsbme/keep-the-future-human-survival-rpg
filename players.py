"""Automated player implementations for the RPG game."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import logging
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


logger = logging.getLogger(__name__)


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
        logger.info("Taking turn")
        char = self.select_character(state)
        logger.info("Selected character: %s", char.name)
        options = char.generate_actions(state.history)
        if not options:
            logger.info("No actions available for %s", char.name)
            return
        action = self.select_action(char, options, state)
        logger.info("Selected action for %s: %s", char.name, action)
        state.record_action(char, action)
        scores = assessor.assess(state.characters, state.how_to_win, state.history)
        logger.info("Assessment results: %s", scores)
        state.update_progress(scores)


class RandomPlayer(Player):
    """Player that selects characters and actions randomly."""

    def select_character(self, state: GameState) -> Character:
        char = random.choice(state.characters)
        logger.info("RandomPlayer chose character: %s", char.name)
        return char

    def select_action(self, character: Character, actions: List[str], state: GameState) -> str:
        action = random.choice(actions)
        logger.info("RandomPlayer chose action '%s' for %s", action, character.name)
        return action


class GeminiWinPlayer(Player):
    """Gemini-based player using the 'how to win' guide for choices."""

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        if genai is None:  # pragma: no cover - env without dependency
            raise ModuleNotFoundError("google-generativeai not installed")
        self._model = genai.GenerativeModel(model)

    def select_character(self, state: GameState) -> Character:
        names = ", ".join(
            f"{c.name} ({c.faction} faction)" if c.faction else c.name
            for c in state.characters
        )
        prompt = (
            "You are playing the 'Keep the future human' survival RPG. "
            "Choose which character should act next to best achieve victory.\n"
            f"Available characters: {names}.\n"
            "Respond with the name of the character only."
        )
        logger.debug("GeminiWinPlayer character prompt: %s", prompt)
        resp = self._model.generate_content(prompt).text
        logger.debug("GeminiWinPlayer character response: %s", resp)
        for char in state.characters:
            if char.name in resp:
                logger.info("GeminiWinPlayer chose character: %s", char.name)
                return char
        logger.info("GeminiWinPlayer defaulted to %s", state.characters[0].name)
        return state.characters[0]

    def select_action(self, character: Character, actions: List[str], state: GameState) -> str:
        numbered = "\n".join(f"{idx+1}. {act}" for idx, act in enumerate(actions))
        prompt = (
            "You are deciding which action to take in the 'Keep the future human' RPG. "
            f"Use the following guide to win: {state.how_to_win}\n"
            f"Character: {character.display_name}\n"
            f"Faction context: {character.base_context}\n"
            f"Possible actions:\n{numbered}\n"
            "Respond with the number of the best action."
        )
        logger.debug("GeminiWinPlayer action prompt: %s", prompt)
        resp = self._model.generate_content(prompt).text
        logger.debug("GeminiWinPlayer action response: %s", resp)
        for token in resp.split():
            if token.isdigit():
                idx = int(token) - 1
                if 0 <= idx < len(actions):
                    logger.info(
                        "GeminiWinPlayer chose action '%s' for %s", actions[idx], character.name
                    )
                    return actions[idx]
        logger.info("GeminiWinPlayer defaulted to action '%s'", actions[0])
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
        names = ", ".join(
            f"{c.name} ({c.faction} faction)" if c.faction else c.name
            for c in state.characters
        )
        prompt = (
            f"{self._context}"
            f"Available characters: {names}.\n"
            "Choose the character whose move would most favor governments and corporations.\n"
            "Respond with the name only."
        )
        logger.debug("GeminiGovCorpPlayer character prompt: %s", prompt)
        resp = self._model.generate_content(prompt).text
        logger.debug("GeminiGovCorpPlayer character response: %s", resp)
        for char in state.characters:
            if char.name in resp:
                logger.info("GeminiGovCorpPlayer chose character: %s", char.name)
                return char
        for char in state.characters:
            if char.faction in ("Governments", "Corporations"):
                logger.info("GeminiGovCorpPlayer defaulted to %s", char.name)
                return char
        logger.info("GeminiGovCorpPlayer defaulted to %s", state.characters[0].name)
        return state.characters[0]

    def select_action(self, character: Character, actions: List[str], state: GameState) -> str:
        numbered = "\n".join(f"{idx+1}. {act}" for idx, act in enumerate(actions))
        prompt = (
            f"{self._context}"
            f"Character: {character.display_name}\n"
            f"Possible actions:\n{numbered}\n"
            "Select the action number that best favors governments and corporations."
        )
        logger.debug("GeminiGovCorpPlayer action prompt: %s", prompt)
        resp = self._model.generate_content(prompt).text
        logger.debug("GeminiGovCorpPlayer action response: %s", resp)
        for token in resp.split():
            if token.isdigit():
                idx = int(token) - 1
                if 0 <= idx < len(actions):
                    logger.info(
                        "GeminiGovCorpPlayer chose action '%s' for %s",
                        actions[idx],
                        character.name,
                    )
                    return actions[idx]
        logger.info("GeminiGovCorpPlayer defaulted to action '%s'", actions[0])
        return actions[0]
