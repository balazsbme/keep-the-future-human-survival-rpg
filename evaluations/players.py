"""Automated player implementations for the RPG game."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import logging
import os
import random
from abc import ABC, abstractmethod
from typing import List, Sequence

try:  # pragma: no cover - optional dependency
    import google.generativeai as genai
except ModuleNotFoundError:  # pragma: no cover
    genai = None

from rpg.game_state import ActionAttempt, GameState
from rpg.character import Character, ResponseOption
from rpg.conversation import ConversationEntry
from rpg.assessment_agent import AssessmentAgent


logger = logging.getLogger(__name__)

_DEFAULT_CONVERSATION_EXCHANGES = 8
_CONVERSATION_ENV_KEY = "AUTOMATED_AGENT_MAX_EXCHANGES"
_GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
_GENAI_CONFIGURED = False


def _ensure_gemini_configured() -> None:
    """Initialise the Gemini SDK using the expected environment variable."""

    if genai is None:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError("google-generativeai not installed")
    global _GENAI_CONFIGURED
    if _GENAI_CONFIGURED:
        return
    api_key = os.environ.get(_GEMINI_API_KEY_ENV)
    if not api_key:
        raise EnvironmentError(
            f"{_GEMINI_API_KEY_ENV} environment variable not set"
        )
    genai.configure(api_key=api_key)
    os.environ.setdefault("GOOGLE_API_KEY", api_key)
    _GENAI_CONFIGURED = True


def _conversation_exchange_limit() -> int:
    """Return the maximum exchanges before switching characters."""

    raw_value = os.environ.get(_CONVERSATION_ENV_KEY)
    if not raw_value:
        return _DEFAULT_CONVERSATION_EXCHANGES
    try:
        limit = int(raw_value)
    except ValueError:
        logger.warning(
            "Invalid %s value %r; falling back to %d",
            _CONVERSATION_ENV_KEY,
            raw_value,
            _DEFAULT_CONVERSATION_EXCHANGES,
        )
        return _DEFAULT_CONVERSATION_EXCHANGES
    if limit < 1:
        logger.warning(
            "%s must be at least 1; using default %d",
            _CONVERSATION_ENV_KEY,
            _DEFAULT_CONVERSATION_EXCHANGES,
        )
        return _DEFAULT_CONVERSATION_EXCHANGES
    return limit


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

    def should_reroll(
        self,
        character: Character,
        conversation: Sequence[ConversationEntry],
        attempt: ActionAttempt,
        state: GameState,
    ) -> bool:
        """Return ``True`` when the player wants to reroll ``attempt``."""

        return False

    def take_turn(self, state: GameState, assessor: AssessmentAgent) -> None:
        """Execute a full turn by selecting character, action and updating state."""
        logger.info("Taking turn")
        partner = state.player_character
        action_performed = False
        state.last_action_actor = None
        max_exchanges = _conversation_exchange_limit()
        character_attempts = 0
        max_character_attempts = max(1, len(state.characters))
        while not action_performed and character_attempts < max_character_attempts:
            char = self.select_character(state)
            character_attempts += 1
            logger.info("Selected character: %s", char.name)
            credibility = state.current_credibility(getattr(char, "faction", None))
            for exchange in range(1, max_exchanges + 1):
                conversation = state.conversation_history(char)
                logger.debug(
                    "Exchange %d conversation length for %s: %d entries",
                    exchange,
                    char.name,
                    len(conversation),
                )
                conversation_cache = state.conversation_cache_for_player(char)
                player_options = partner.generate_responses(
                    state.history,
                    conversation,
                    char,
                    partner_credibility=credibility,
                    conversation_cache=conversation_cache,
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
                    logger.info(
                        "No conversation options available for %s", char.name
                    )
                    break
                try:
                    selection = self.select_action(
                        char, conversation, options, state
                    )
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
                    attempt = state.attempt_action(char, selection)
                    if not attempt.success:
                        while True:
                            current_conversation = state.conversation_history(char)
                            if not self.should_reroll(
                                char, current_conversation, attempt, state
                            ):
                                state.finalize_failed_action(char, selection)
                                break
                            attempt = state.reroll_action(char, selection)
                            if attempt.success:
                                break
                    break
                npc_responses = char.generate_responses(
                    state.history,
                    state.conversation_history(char),
                    partner,
                    partner_credibility=credibility,
                    force_action=state.should_force_action(char),
                )
                state.log_npc_responses(char, npc_responses)
            if action_performed:
                break
            logger.info(
                "No action performed for %s; selecting another character",
                char.name,
            )
        if not action_performed:
            logger.info("No action performed during this turn")
            state.last_action_attempt = None
            state.last_action_actor = None
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

    def should_reroll(
        self,
        character: Character,
        conversation: Sequence[ConversationEntry],
        attempt: ActionAttempt,
        state: GameState,
    ) -> bool:
        decision = random.choice([True, False])
        logger.info(
            "RandomPlayer reroll decision for %s (cost=%d): %s",
            character.name,
            state.next_reroll_cost(character, attempt.option),
            "yes" if decision else "no",
        )
        return decision


class ActionFirstRandomPlayer(RandomPlayer):
    """Random-like player that prioritises taking the first available action."""

    def select_action(
        self,
        character: Character,
        conversation: Sequence[ConversationEntry],
        actions: List[ResponseOption],
        state: GameState,
    ) -> ResponseOption:
        for option in actions:
            if option.is_action:
                logger.info(
                    "ActionFirstRandomPlayer chose immediate action '%s' for %s",
                    option.text,
                    character.name,
                )
                return option
        action = random.choice(actions)
        logger.info(
            "ActionFirstRandomPlayer fell back to random dialogue '%s' for %s",
            action.text,
            character.name,
        )
        return action

    def should_reroll(
        self,
        character: Character,
        conversation: Sequence[ConversationEntry],
        attempt: ActionAttempt,
        state: GameState,
    ) -> bool:
        cost = state.next_reroll_cost(character, attempt.option)
        logger.info(
            "ActionFirstRandomPlayer reroll decision for %s (cost=%d): yes",
            character.name,
            cost,
        )
        return True


class GeminiCivilSocietyPlayer(Player):
    """Gemini-based player using the civil society victory guide."""

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        _ensure_gemini_configured()
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

    def should_reroll(
        self,
        character: Character,
        conversation: Sequence[ConversationEntry],
        attempt: ActionAttempt,
        state: GameState,
    ) -> bool:
        failure_summary = attempt.failure_text or (
            f"Failed {attempt.label} with roll {attempt.roll}"
        )
        cost = state.next_reroll_cost(character, attempt.option)
        numbered_history = _format_conversation(conversation)
        prompt = (
            "You are guiding a civil society coalition assessing whether a failed action "
            "is important enough to reroll in the 'Keep the future human' RPG. "
            f"Use the victory guide: {state.how_to_win}\n"
            f"Character: {character.display_name}\n"
            f"Conversation so far:\n{numbered_history}\n"
            f"Action attempted: {attempt.option.text}\n"
            f"Outcome: {failure_summary}\n"
            f"Next reroll credibility cost: {cost}\n"
            "Reply with YES to reroll if the action is critical for winning; otherwise respond NO."
        )
        logger.debug("GeminiCivilSocietyPlayer reroll prompt: %s", prompt)
        try:
            response = self._model.generate_content(prompt).text
        except Exception:  # pragma: no cover - defensive fallback
            logger.exception("GeminiCivilSocietyPlayer reroll query failed")
            return False
        logger.debug("GeminiCivilSocietyPlayer reroll response: %s", response)
        return "yes" in response.lower()


class GeminiCorporationPlayer(Player):
    """Gemini-based player focusing solely on corporation faction goals."""

    def __init__(
        self,
        corporation_context: str,
        model: str = "gemini-2.5-flash",
    ) -> None:
        _ensure_gemini_configured()
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

    def should_reroll(
        self,
        character: Character,
        conversation: Sequence[ConversationEntry],
        attempt: ActionAttempt,
        state: GameState,
    ) -> bool:
        logger.info(
            "GeminiCorporationPlayer skips reroll for %s after failure",
            character.name,
        )
        return False


# Backwards compatibility aliases for legacy names
GeminiWinPlayer = GeminiCivilSocietyPlayer
GeminiGovCorpPlayer = GeminiCorporationPlayer
