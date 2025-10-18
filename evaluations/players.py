"""Automated player implementations for the RPG game."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import List, Sequence

try:  # pragma: no cover - optional dependency
    import google.generativeai as genai
except ModuleNotFoundError:  # pragma: no cover
    genai = None

from rpg.game_state import GameState
from rpg.character import Character, ResponseOption
from rpg.conversation import ConversationEntry
from rpg.assessment_agent import AssessmentAgent


logger = logging.getLogger(__name__)


def _format_conversation(conversation: Sequence[ConversationEntry]) -> str:
    """Return a formatted string representation of a conversation history."""

    if not conversation:
        return "None"
    return "\n".join(
        f"{entry.speaker}: {entry.text} [{entry.type}]" for entry in conversation
    )


class Player(ABC):
    """Abstract player interface for extending player logic."""

    @abstractmethod
    def select_character(self, state: GameState) -> Character:
        """Return the character that should act this turn."""

    @abstractmethod
    def select_action(
        self,
        character: Character,
        conversation: Sequence[ConversationEntry],
        actions: List[ResponseOption],
        state: GameState,
    ) -> ResponseOption:
        """Return the chosen option for ``character`` from ``actions``."""

    def take_turn(self, state: GameState, assessor: AssessmentAgent) -> None:
        """Execute a full turn by selecting character, action and updating state."""
        logger.info("Taking turn")
        char = self.select_character(state)
        logger.info("Selected character: %s", char.name)
        partner = state.player_character
        credibility = state.current_credibility(getattr(char, "faction", None))
        action_performed = False
        max_exchanges = 8
        for exchange in range(1, max_exchanges + 1):
            conversation = state.conversation_history(char)
            logger.debug(
                "Exchange %d conversation length for %s: %d entries",
                exchange,
                char.name,
                len(conversation),
            )
            player_options = partner.generate_responses(
                state.history,
                conversation,
                char,
                partner_credibility=credibility,
            )
            stored_actions = state.available_npc_actions(char)
            options: List[ResponseOption] = []
            seen_texts = set()
            for option in player_options:
                if option.text not in seen_texts:
                    options.append(option)
                    seen_texts.add(option.text)
            for option in stored_actions:
                if option.text not in seen_texts:
                    options.append(option)
                    seen_texts.add(option.text)
            if not options:
                logger.info("No conversation options available for %s", char.name)
                state.last_action_attempt = None
                return
            try:
                selection = self.select_action(char, conversation, options, state)
            except Exception:  # pragma: no cover - defensive fallback
                logger.exception(
                    "Error selecting option for %s; defaulting to first choice",
                    char.name,
                )
                selection = options[0]
            if selection not in options:
                logger.warning(
                    "Selected option for %s not in available list; defaulting to first",
                    char.name,
                )
                selection = options[0]
            logger.info(
                "Selected option for %s: %s (%s)",
                char.name,
                selection.text,
                selection.type,
            )
            state.log_player_response(char, selection)
            if selection.is_action:
                action_performed = True
                state.record_action(char, selection)
                break
            npc_responses = char.generate_responses(
                state.history,
                state.conversation_history(char),
                partner,
                partner_credibility=credibility,
            )
            state.log_npc_responses(char, npc_responses)
        if not action_performed:
            logger.info("No action performed for %s during this turn", char.name)
            state.last_action_attempt = None
            return
        scores = assessor.assess(state.characters, state.how_to_win, state.history)
        logger.info("Assessment results: %s", scores)
        state.update_progress(scores)


class RandomPlayer(Player):
    """Player that selects characters and actions randomly."""

    def select_character(self, state: GameState) -> Character:
        char = random.choice(state.characters)
        logger.info("RandomPlayer chose character: %s", char.name)
        return char

    def select_action(
        self,
        character: Character,
        conversation: Sequence[ConversationEntry],
        actions: List[ResponseOption],
        state: GameState,
    ) -> ResponseOption:
        action = random.choice(actions)
        logger.info(
            "RandomPlayer chose action '%s' for %s", action.text, character.name
        )
        return action


class GeminiCivilSocietyPlayer(Player):
    """Gemini-based player using the civil society victory guide."""

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
        logger.debug("GeminiCivilSocietyPlayer character prompt: %s", prompt)
        resp = self._model.generate_content(prompt).text
        logger.debug("GeminiCivilSocietyPlayer character response: %s", resp)
        for char in state.characters:
            if char.name in resp:
                logger.info("GeminiCivilSocietyPlayer chose character: %s", char.name)
                return char
        logger.info(
            "GeminiCivilSocietyPlayer defaulted to %s", state.characters[0].name
        )
        return state.characters[0]

    def select_action(
        self,
        character: Character,
        conversation: Sequence[ConversationEntry],
        actions: List[ResponseOption],
        state: GameState,
    ) -> ResponseOption:
        conversation_text = _format_conversation(conversation)
        numbered = "\n".join(
            f"{idx+1}. [{'Action' if act.is_action else 'Dialogue'}] {act.text}"
            for idx, act in enumerate(actions)
        )
        prompt = (
            "You are a civil society strategist in the 'Keep the future human' RPG. "
            "Prioritise collaborative, human-centric outcomes that secure a win for the coalition. "
            f"Use the following guide to win: {state.how_to_win}\n"
            f"Character: {character.display_name}\n"
            f"Faction context: {character.base_context}\n"
            "Conversation so far:\n"
            f"{conversation_text}\n"
            f"Possible actions:\n{numbered}\n"
            "Respond with the number of the best action."
        )
        logger.debug("GeminiCivilSocietyPlayer action prompt: %s", prompt)
        resp = self._model.generate_content(prompt).text
        logger.debug("GeminiCivilSocietyPlayer action response: %s", resp)
        for token in resp.split():
            if token.isdigit():
                idx = int(token) - 1
                if 0 <= idx < len(actions):
                    logger.info(
                        "GeminiCivilSocietyPlayer chose action '%s' for %s",
                        actions[idx].text,
                        character.name,
                    )
                    return actions[idx]
        logger.info(
            "GeminiCivilSocietyPlayer defaulted to action '%s'", actions[0].text
        )
        return actions[0]


class GeminiCorporationPlayer(Player):
    """Gemini-based player focusing solely on corporation faction goals."""

    def __init__(
        self,
        corporation_context: str,
        model: str = "gemini-2.5-flash",
    ) -> None:
        if genai is None:  # pragma: no cover - env without dependency
            raise ModuleNotFoundError("google-generativeai not installed")
        self._model = genai.GenerativeModel(model)
        self._context = f"Corporation context:\n{corporation_context}\n"

    def select_character(self, state: GameState) -> Character:
        names = ", ".join(
            f"{c.name} ({c.faction} faction)" if c.faction else c.name
            for c in state.characters
        )
        prompt = (
            f"{self._context}"
            f"Available characters: {names}.\n"
            "Choose the character whose move would best advance the corporation faction objectives.\n"
            "Respond with the name only."
        )
        logger.debug("GeminiCorporationPlayer character prompt: %s", prompt)
        resp = self._model.generate_content(prompt).text
        logger.debug("GeminiCorporationPlayer character response: %s", resp)
        for char in state.characters:
            if char.name in resp:
                logger.info("GeminiCorporationPlayer chose character: %s", char.name)
                return char
        for char in state.characters:
            if char.faction == "Corporations":
                logger.info("GeminiCorporationPlayer defaulted to %s", char.name)
                return char
        logger.info(
            "GeminiCorporationPlayer defaulted to %s", state.characters[0].name
        )
        return state.characters[0]

    def select_action(
        self,
        character: Character,
        conversation: Sequence[ConversationEntry],
        actions: List[ResponseOption],
        state: GameState,
    ) -> ResponseOption:
        conversation_text = _format_conversation(conversation)
        numbered = "\n".join(
            f"{idx+1}. [{'Action' if act.is_action else 'Dialogue'}] {act.text}"
            for idx, act in enumerate(actions)
        )
        prompt = (
            f"{self._context}"
            f"Character: {character.display_name}\n"
            "Conversation so far:\n"
            f"{conversation_text}\n"
            f"Possible actions:\n{numbered}\n"
            "Select the action number that best aligns with the corporation faction objectives."
        )
        logger.debug("GeminiCorporationPlayer action prompt: %s", prompt)
        resp = self._model.generate_content(prompt).text
        logger.debug("GeminiCorporationPlayer action response: %s", resp)
        for token in resp.split():
            if token.isdigit():
                idx = int(token) - 1
                if 0 <= idx < len(actions):
                    logger.info(
                        "GeminiCorporationPlayer chose action '%s' for %s",
                        actions[idx].text,
                        character.name,
                    )
                    return actions[idx]
        logger.info(
            "GeminiCorporationPlayer defaulted to action '%s'", actions[0].text
        )
        return actions[0]


# Backwards compatibility aliases for legacy names
GeminiWinPlayer = GeminiCivilSocietyPlayer
GeminiGovCorpPlayer = GeminiCorporationPlayer
