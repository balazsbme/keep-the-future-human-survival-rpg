"""Game state storage for characters and their action history."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from html import escape
from typing import Dict, Iterable, List, Tuple

from .character import Character, PlayerCharacter, ResponseOption
from .credibility import CREDIBILITY_PENALTY, CREDIBILITY_REWARD, CredibilityMatrix
from .config import GameConfig, load_game_config
from .conversation import ConversationEntry


logger = logging.getLogger(__name__)


GAME_CONFIG = load_game_config()
WIN_THRESHOLD = GAME_CONFIG.win_threshold
MAX_ROUNDS = GAME_CONFIG.max_rounds
PLAYER_FACTION = "CivilSociety"


@dataclass
class ActionAttempt:
    """Outcome of a single action attempt including roll details."""

    success: bool
    option: ResponseOption
    attribute: str | None
    actor_score: int
    player_score: int
    effective_score: int
    roll: float
    targets: Tuple[str, ...]
    credibility_cost: int
    credibility_gain: int
    failure_text: str | None = None


@dataclass
class GameState:
    """Store characters, their progress, and a history of actions."""

    characters: List[Character]
    history: List[Tuple[str, str]] = field(default_factory=list)
    conversations: Dict[str, List[ConversationEntry]] = field(default_factory=dict)
    npc_actions: Dict[str, Dict[str, ResponseOption]] = field(default_factory=dict)
    progress: Dict[str, List[int]] = field(init=False)
    weights: Dict[str, List[int]] = field(init=False)
    how_to_win: str = field(init=False)
    faction_labels: Dict[str, str] = field(init=False)
    config: GameConfig = field(init=False)
    credibility: CredibilityMatrix = field(init=False)
    player_character: PlayerCharacter = field(init=False)
    pending_failures: Dict[Tuple[str, str], ActionAttempt] = field(
        default_factory=dict, init=False
    )
    reroll_counts: Dict[Tuple[str, str], int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Initialize progress tracking and load "how to win" instructions.

        Returns:
            None.
        """
        logger.info("Initializing game state")
        self.progress = {}
        self.weights = {}
        self.faction_labels = {}
        self.config = GAME_CONFIG
        self.credibility = CredibilityMatrix()
        self.credibility.ensure_faction(PLAYER_FACTION)
        self.player_character = PlayerCharacter()
        for character in self.characters:
            key = character.progress_key
            if key not in self.progress:
                self.progress[key] = [0] * len(character.triplets)
                self.weights[key] = getattr(
                    character, "weights", [1] * len(character.triplets)
                )
                self.faction_labels[key] = character.progress_label
            self.credibility.ensure_faction(getattr(character, "faction", None))
        win_path = os.path.join(os.path.dirname(__file__), "..", "how-to-win.md")
        with open(win_path, "r", encoding="utf-8") as f:
            self.how_to_win = f.read()

    def _conversation_key(self, character: Character) -> str:
        """Return the key used to store conversation state for ``character``."""

        return character.name

    def conversation_history(self, character: Character) -> List[ConversationEntry]:
        """Return the stored conversation history for ``character``."""

        key = self._conversation_key(character)
        return list(self.conversations.get(key, []))

    def log_player_response(
        self, character: Character, option: ResponseOption
    ) -> ConversationEntry:
        """Record the player's chosen option in the conversation log."""

        entry = ConversationEntry(
            speaker=self.player_character.display_name,
            text=option.text,
            type=option.type,
        )
        key = self._conversation_key(character)
        self.conversations.setdefault(key, []).append(entry)
        logger.debug(
            "Logged player response for %s: %s (%s)", character.name, option.text, option.type
        )
        return entry

    def log_npc_responses(
        self, character: Character, responses: Iterable[ResponseOption]
    ) -> List[ConversationEntry]:
        """Record NPC responses and track proposed actions."""

        key = self._conversation_key(character)
        history = self.conversations.setdefault(key, [])
        action_bucket = self.npc_actions.setdefault(key, {})
        entries: List[ConversationEntry] = []
        action_candidate: ResponseOption | None = None
        chat_candidate: ResponseOption | None = None
        fallback_option: ResponseOption | None = None
        for option in responses:
            if option.is_action and action_candidate is None:
                action_candidate = option
            elif not option.is_action and chat_candidate is None:
                chat_candidate = option
            fallback_option = fallback_option or option

        selected_option = action_candidate or chat_candidate or fallback_option
        if selected_option is not None:
            entry = ConversationEntry(
                speaker=character.display_name,
                text=selected_option.text,
                type=selected_option.type,
            )
            history.append(entry)
            entries.append(entry)
            if selected_option.is_action:
                action_bucket[selected_option.text] = selected_option
        # Store any additional unique action proposals for later execution.
        for option in responses:
            if option.is_action:
                action_bucket.setdefault(option.text, option)
        logger.debug(
            "Logged %d NPC responses for %s (stored %d actions)",
            len(entries),
            character.name,
            len(action_bucket),
        )
        return entries

    def available_npc_actions(self, character: Character) -> List[ResponseOption]:
        """Return all unique action proposals made by ``character``."""

        key = self._conversation_key(character)
        bucket = self.npc_actions.get(key, {})
        return list(bucket.values())

    def record_action(
        self,
        character: Character,
        action: ResponseOption | str,
        *,
        targets: Iterable[str] | None = None,
    ) -> bool:
        """Resolve an action and automatically log failures when not rerolled."""

        option = (
            action
            if isinstance(action, ResponseOption)
            else ResponseOption(text=str(action), type="action")
        )
        attempt = self.attempt_action(character, option, targets=targets)
        if attempt.success:
            return True
        self.finalize_failed_action(character, option)
        return False

    def attempt_action(
        self,
        character: Character,
        action: ResponseOption | str,
        *,
        targets: Iterable[str] | None = None,
    ) -> ActionAttempt:
        """Attempt an action without immediately logging failure results."""

        option = (
            action
            if isinstance(action, ResponseOption)
            else ResponseOption(text=str(action), type="action")
        )
        if not option.is_action:
            raise ValueError("attempt_action requires an action-type ResponseOption")
        logger.info("Evaluating action '%s' for %s", option.text, character.name)
        attribute_name = option.related_attribute
        attribute_score = character.attribute_score(attribute_name)
        player_score = self.player_character.attribute_score(attribute_name)
        if attribute_name:
            logger.info(
                "Action related attribute for %s: %s", character.name, attribute_name
            )
        else:
            logger.info(
                "Action for %s has no related attribute; defaulting score to 0",
                character.name,
            )
        if player_score > attribute_score:
            logger.info(
                "Player attribute advantage: using player score %s over %s's %s",
                player_score,
                character.name,
                attribute_score,
            )
        effective_score = attribute_score
        if player_score > attribute_score:
            effective_score = player_score
        logger.info(
            "Using attribute score %s for success threshold", effective_score
        )
        sampled_value = random.uniform(0, 10)
        logger.info("Sampled %.2f from uniform[0, 10]", sampled_value)
        success = sampled_value < effective_score
        cost, gain = self._credibility_cost_gain(attribute_score, player_score)
        label = attribute_name or "none"
        failure_text = (
            f"Failed '{option.text}' (attribute {label}: {effective_score}, roll={sampled_value:.2f})"
        )
        targets_tuple = tuple(targets or [])
        attempt = ActionAttempt(
            success=success,
            option=option,
            attribute=attribute_name,
            actor_score=attribute_score,
            player_score=player_score,
            effective_score=effective_score,
            roll=sampled_value,
            targets=targets_tuple,
            credibility_cost=cost,
            credibility_gain=gain,
            failure_text=None if success else failure_text,
        )
        key = (character.name, option.text)
        if success:
            logger.info("Action succeeded; recording in history")
            self.pending_failures.pop(key, None)
            self.reroll_counts.pop(key, None)
            self.history.append((character.display_name, option.text))
            self._apply_credibility_updates(
                character,
                option,
                targets,
                cost=cost,
                gain=gain,
            )
        else:
            logger.info("Action failed; storing failure for potential reroll")
            self.pending_failures[key] = attempt
            self.reroll_counts.setdefault(key, 0)
        return attempt

    def _credibility_cost_gain(self, actor_score: int, player_score: int) -> Tuple[int, int]:
        base_cost = CREDIBILITY_PENALTY
        base_gain = CREDIBILITY_REWARD
        diff = actor_score - player_score
        if diff < 0:
            advantage = -diff
            return max(0, base_cost - advantage), base_gain + advantage
        if diff > 0:
            penalty = diff
            return base_cost + penalty, max(0, base_gain - penalty)
        return base_cost, base_gain

    def _apply_reroll_penalty(
        self,
        character: Character,
        attempt: ActionAttempt,
        reroll_count: int,
    ) -> None:
        actor_faction = getattr(character, "faction", None)
        penalty = attempt.credibility_cost * reroll_count
        if penalty <= 0:
            return
        self.credibility.ensure_faction(PLAYER_FACTION)
        target_factions = list(attempt.targets or [actor_faction])
        for target in target_factions:
            if not target or target == PLAYER_FACTION:
                continue
            self.credibility.ensure_faction(target)
            logger.debug(
                "Applying reroll penalty for source=%s target=%s by -%d (attempt %d)",
                PLAYER_FACTION,
                target,
                penalty,
                reroll_count,
            )
            self.credibility.adjust(PLAYER_FACTION, target, -penalty)

    def finalize_failed_action(
        self,
        character: Character,
        action: ResponseOption | str,
    ) -> None:
        option = (
            action
            if isinstance(action, ResponseOption)
            else ResponseOption(text=str(action), type="action")
        )
        key = (character.name, option.text)
        attempt = self.pending_failures.pop(key, None)
        self.reroll_counts.pop(key, None)
        if not attempt:
            logger.info("No pending failure to finalize for %s", option.text)
            return
        failure_text = attempt.failure_text or (
            f"Failed '{option.text}' (attribute {attempt.attribute or 'none'}: {attempt.effective_score}, roll={attempt.roll:.2f})"
        )
        logger.info("Recording failed action for %s: %s", character.name, failure_text)
        self.history.append((character.display_name, failure_text))

    def reroll_action(
        self,
        character: Character,
        action: ResponseOption | str,
        *,
        targets: Iterable[str] | None = None,
    ) -> ActionAttempt:
        option = (
            action
            if isinstance(action, ResponseOption)
            else ResponseOption(text=str(action), type="action")
        )
        key = (character.name, option.text)
        attempt = self.pending_failures.pop(key, None)
        if attempt is None:
            raise ValueError("No pending failed action to reroll")
        reroll_count = self.reroll_counts.get(key, 0) + 1
        self.reroll_counts[key] = reroll_count
        self._apply_reroll_penalty(character, attempt, reroll_count)
        if targets is not None:
            targets_tuple = tuple(targets)
        else:
            targets_tuple = attempt.targets
        return self.attempt_action(character, option, targets=targets_tuple)

    def next_reroll_cost(
        self,
        character: Character,
        action: ResponseOption | str,
    ) -> int:
        option = (
            action
            if isinstance(action, ResponseOption)
            else ResponseOption(text=str(action), type="action")
        )
        key = (character.name, option.text)
        attempt = self.pending_failures.get(key)
        if not attempt:
            return 0
        reroll_count = self.reroll_counts.get(key, 0) + 1
        return attempt.credibility_cost * reroll_count

    def _apply_credibility_updates(
        self,
        character: Character,
        action: ResponseOption,
        targets: Iterable[str] | None,
        *,
        cost: int,
        gain: int,
    ) -> None:
        """Update credibility values after a successful action."""

        actor_faction = getattr(character, "faction", None)
        self.credibility.ensure_faction(PLAYER_FACTION)
        if actor_faction:
            self.credibility.ensure_faction(actor_faction)
        delta = -cost if action.related_triplet is not None else gain
        target_factions = list(targets or [actor_faction])
        for target in target_factions:
            if not target or target == PLAYER_FACTION:
                continue
            self.credibility.ensure_faction(target)
            logger.debug(
                "Adjusting credibility for source=%s target=%s by %+d",
                PLAYER_FACTION,
                target,
                delta,
            )
            self.credibility.adjust(PLAYER_FACTION, target, delta)

    def update_progress(self, scores: Dict[str, List[int]]) -> None:
        """Update progress scores for all characters.

        Args:
            scores: Mapping of character name to list of progress values.

        Returns:
            None.
        """
        for key, new_scores in scores.items():
            if key not in self.progress:
                continue
            current = self.progress[key]
            for idx, score in enumerate(new_scores):
                if idx < len(current):
                    current[idx] = score

    def _faction_weighted_score(self, key: str) -> int:
        """Return weighted score for a single faction."""

        scores = self.progress.get(key, [])
        weights = self.weights.get(key, [])
        total = sum(weights)
        if not scores or total == 0:
            return 0
        return round(sum(s * w for s, w in zip(scores, weights)) / total)

    def final_weighted_score(self) -> int:
        """Return weighted score across all factions."""

        totals = []
        for key in self.progress:
            weight_total = sum(self.weights.get(key, []))
            faction_score = self._faction_weighted_score(key)
            totals.append((faction_score, weight_total))
        grand_total = sum(w for _, w in totals)
        if grand_total == 0:
            return 0
        return round(sum(score * w for score, w in totals) / grand_total)

    def render_state(self) -> str:
        """Return HTML rendering of the current game state.

        Returns:
            HTML string describing character progress and history.
        """
        logger.info("Rendering game state")
        lines = []
        for key, scores in self.progress.items():
            label = escape(self.faction_labels.get(key, key), quote=False)
            lines.append(
                f"{label}: {scores} (weighted: {self._faction_weighted_score(key)})"
            )
        if self.history:
            hist_items = "".join(
                f"<li><strong>{escape(n, quote=False)}</strong>: {escape(a, quote=False)}</li>"
                for n, a in self.history
            )
            lines.append(f"<h2>Action History</h2><ol>{hist_items}</ol>")
        credibility_snapshot = self.credibility.snapshot()
        if credibility_snapshot:
            headers = "".join(
                f"<th>{escape(target, quote=False)}</th>" for target in self.credibility.factions
            )
            body_rows = []
            for source in self.credibility.factions:
                row_cells = [f"<th scope='row'>{escape(source, quote=False)}</th>"]
                for target in self.credibility.factions:
                    value = credibility_snapshot.get(source, {}).get(target, 0)
                    row_cells.append(f"<td>{int(value)}</td>")
                body_rows.append("<tr>" + "".join(row_cells) + "</tr>")
            lines.append(
                "<h2>Credibility Matrix</h2>"
                + "<table><thead><tr><th>Source \\ Target</th>"
                + headers
                + "</tr></thead><tbody>"
                + "".join(body_rows)
                + "</tbody></table>"
            )
        lines.append(f"Final weighted score: {self.final_weighted_score()}")
        return "<div id='state'>" + "<br>".join(lines) + "</div>"
