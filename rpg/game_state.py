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


PLAYER_FACTION = "CivilSociety"


@dataclass
class ActionAttempt:
    """Outcome of a single action attempt including roll details."""

    success: bool
    option: ResponseOption
    label: str
    attribute: str | None
    actor_score: int
    player_score: int
    effective_score: int
    roll: int
    targets: Tuple[str, ...]
    credibility_cost: int
    credibility_gain: int
    failure_text: str | None = None


@dataclass
class GameState:
    """Store characters, their progress, and a history of actions."""

    characters: List[Character]
    config_override: GameConfig | None = None
    player_override: PlayerCharacter | None = None
    history: List[Tuple[str, str]] = field(default_factory=list)
    conversations: Dict[str, List[ConversationEntry]] = field(default_factory=dict)
    faction_conversations: Dict[str, List[ConversationEntry]] = field(
        default_factory=dict
    )
    npc_actions: Dict[str, Dict[str, ResponseOption]] = field(default_factory=dict)
    action_labels: Dict[str, Dict[str, str]] = field(default_factory=dict)
    action_label_indices: Dict[str, Dict[str, int]] = field(default_factory=dict)
    progress: Dict[str, List[int]] = field(init=False)
    weights: Dict[str, List[int]] = field(init=False)
    how_to_win: str = field(init=False)
    scenario_summary: str = field(init=False, default="")
    faction_labels: Dict[str, str] = field(init=False)
    config: GameConfig = field(init=False)
    credibility: CredibilityMatrix = field(init=False)
    player_character: PlayerCharacter = field(init=False)
    player_faction: str = field(init=False)
    pending_failures: Dict[Tuple[str, str], ActionAttempt] = field(
        default_factory=dict, init=False
    )
    reroll_counts: Dict[Tuple[str, str], int] = field(default_factory=dict, init=False)
    player_action_records: Dict[str, Dict[int, ResponseOption]] = field(
        default_factory=dict, init=False
    )
    last_action_attempt: ActionAttempt | None = field(default=None, init=False)
    last_action_actor: str | None = field(default=None, init=False)
    last_reroll_count: int = field(default=0, init=False)
    time_elapsed_years: float = field(default=0.0, init=False)
    next_action_label_index: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        """Initialize progress tracking and load "how to win" instructions.

        Returns:
            None.
        """
        logger.info("Initializing game state")
        self.progress = {}
        self.weights = {}
        self.faction_labels = {}
        self.config = self.config_override or load_game_config()
        self.credibility = CredibilityMatrix()
        self.player_character = self.player_override or PlayerCharacter(
            config=self.config
        )
        player_summary = getattr(self.player_character, "scenario_summary", "")
        character_summary = ""
        for npc in self.characters:
            candidate = getattr(npc, "scenario_summary", "")
            if candidate:
                character_summary = candidate
                break
        self.scenario_summary = character_summary or player_summary
        self.player_faction = self.player_character.faction or PLAYER_FACTION
        self.credibility.ensure_faction(self.player_faction)
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

    def _update_faction_cache(self, character: Character) -> None:
        faction = getattr(character, "faction", None)
        if not faction:
            return
        key = self._conversation_key(character)
        history = self.conversations.get(key, [])
        self.faction_conversations[faction] = list(history)

    def conversation_cache_for_player(
        self, character: Character
    ) -> Dict[str, List[ConversationEntry]]:
        current_faction = getattr(character, "faction", None)
        caches: Dict[str, List[ConversationEntry]] = {}
        for faction, entries in self.faction_conversations.items():
            if faction == current_faction or not entries:
                continue
            caches[faction] = list(entries)
        return caches

    def should_force_action(self, character: Character) -> bool:
        limit = getattr(self.config, "conversation_force_action_after", 0)
        if limit <= 0:
            return False
        return len(self.conversation_history(character)) >= limit

    def log_player_response(
        self, character: Character, option: ResponseOption
    ) -> ConversationEntry:
        """Record the player's chosen option in the conversation log."""

        key = self._conversation_key(character)
        history = self.conversations.setdefault(key, [])
        entry_text = option.text
        if option.is_action:
            label = self._resolve_action_label(character, option)
            entry_text = f"Agreed on performing action {label}"
            record = self.player_action_records.setdefault(key, {})
            record[len(history)] = option
        entry = ConversationEntry(
            speaker=self.player_character.display_name,
            text=entry_text,
            type=option.type,
        )
        history.append(entry)
        self._update_faction_cache(character)
        logger.debug(
            "Logged player response for %s: %s (%s)",
            character.name,
            entry_text,
            option.type,
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
            if selected_option.is_action:
                label = self._resolve_action_label(character, selected_option)
                text = f"{label}: {selected_option.text}"
            else:
                text = selected_option.text
            entry = ConversationEntry(
                speaker=character.display_name,
                text=text,
                type=selected_option.type,
            )
            history.append(entry)
            entries.append(entry)
            self._update_faction_cache(character)
        if selected_option is None and action_bucket:
            self._refresh_action_labels(key)
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

    def _format_action_label(self, index: int, option: ResponseOption) -> str:
        attribute = option.related_attribute.title() if option.related_attribute else "None"
        return f"Action {index} [{attribute}]"

    def _refresh_action_labels(self, key: str) -> None:
        bucket = self.npc_actions.get(key, {})
        labels = self.action_labels.setdefault(key, {})
        indices = self.action_label_indices.setdefault(key, {})
        active_option_texts = set()
        for option in bucket.values():
            option_text = option.text
            active_option_texts.add(option_text)
            if option_text not in indices:
                index = self.next_action_label_index
                self.next_action_label_index += 1
                indices[option_text] = index
                labels[option_text] = self._format_action_label(index, option)
                continue
            index = indices[option_text]
            new_label = self._format_action_label(index, option)
            previous_label = labels.get(option_text)
            if previous_label is not None and previous_label != new_label:
                logger.warning(
                    "Using default action label '%s' for %s option '%s'",
                    new_label,
                    key,
                    option_text,
                )
            labels[option_text] = new_label
        stale_texts = [text for text in labels if text not in active_option_texts]
        for text in stale_texts:
            labels.pop(text, None)
            indices.pop(text, None)

    def _resolve_action_label(self, character: Character, option: ResponseOption) -> str:
        key = self._conversation_key(character)
        bucket = self.npc_actions.setdefault(key, {})
        if option.text not in bucket:
            bucket[option.text] = option
        self._refresh_action_labels(key)
        labels = self.action_labels.setdefault(key, {})
        label = labels.get(option.text)
        if label is not None:
            return label
        indices = self.action_label_indices.setdefault(key, {})
        index = indices.get(option.text)
        if index is None:
            index = self.next_action_label_index
            self.next_action_label_index += 1
            indices[option.text] = index
        label = self._format_action_label(index, option)
        labels[option.text] = label
        return label

    def action_label_map(self, character: Character) -> Dict[str, str]:
        key = self._conversation_key(character)
        bucket = self.npc_actions.setdefault(key, {})
        if bucket:
            self._refresh_action_labels(key)
        labels = self.action_labels.get(key, {})
        return dict(labels)

    def clear_available_actions(self, character: Character) -> None:
        key = self._conversation_key(character)
        self.npc_actions[key] = {}
        self.action_labels[key] = {}
        self.action_label_indices[key] = {}

    def current_credibility(self, target_faction: str | None) -> int | None:
        if not target_faction:
            return None
        self.credibility.ensure_faction(self.player_faction)
        self.credibility.ensure_faction(target_faction)
        return self.credibility.value(self.player_faction, target_faction)

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
        advance_time: bool = True,
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
        action_label = self._resolve_action_label(character, option)
        self.last_action_actor = character.display_name
        attribute_name = option.related_attribute
        attribute_score = character.attribute_score(attribute_name)
        player_score = self.player_character.attribute_score(attribute_name)
        if attribute_name:
            logger.info(
                "Action related attribute for %s: %s", character.name, attribute_name
            )
        else:
            logger.warning(
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
        logger.info("Using attribute score %s as roll modifier", effective_score)
        if advance_time:
            self.time_elapsed_years += self.config.action_time_cost_years
        sampled_value = random.randint(1, 20)
        logger.info("Sampled %d from randint[1, 20]", sampled_value)
        roll_total = effective_score + sampled_value
        logger.info(
            "Adjusted roll %s (modifier %s + roll %s) vs threshold %s",
            roll_total,
            effective_score,
            sampled_value,
            self.config.roll_success_threshold,
        )
        success = roll_total >= self.config.roll_success_threshold
        cost, gain = self._credibility_cost_gain(attribute_score, player_score)
        attribute_label = attribute_name or "none"
        failure_text = (
            f"Failed {action_label} (attribute {attribute_label}: {effective_score}, "
            f"roll={sampled_value}, total={roll_total}, "
            f"threshold={self.config.roll_success_threshold})"
        )
        targets_tuple = tuple(targets or [])
        attempt = ActionAttempt(
            success=success,
            option=option,
            label=action_label,
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
        self.last_action_attempt = attempt
        key = (character.name, option.text)
        reroll_count = self.reroll_counts.get(key, 0)
        self.last_reroll_count = reroll_count
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
            reroll_value = self.reroll_counts.setdefault(key, reroll_count)
            self.last_reroll_count = reroll_value
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
        self.credibility.ensure_faction(self.player_faction)
        target_factions = list(attempt.targets or [actor_faction])
        for target in target_factions:
            if not target:
                continue
            self.credibility.ensure_faction(target)
            logger.debug(
                "Applying reroll penalty for source=%s target=%s by -%d (attempt %d)",
                self.player_faction,
                target,
                penalty,
                reroll_count,
            )
            self.credibility.adjust(self.player_faction, target, -penalty)

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
        reroll_count = self.reroll_counts.pop(key, None) or 0
        self.last_reroll_count = reroll_count
        self.last_action_actor = character.display_name
        if not attempt:
            logger.info("No pending failure to finalize for %s", option.text)
            return
        failure_text = attempt.failure_text
        if not failure_text:
            attribute_label = attempt.attribute or "none"
            total = attempt.effective_score + attempt.roll
            failure_text = (
                f"Failed {attempt.label} (attribute {attribute_label}: {attempt.effective_score}, "
                f"roll={attempt.roll}, total={total}, "
                f"threshold={self.config.roll_success_threshold})"
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
        self.last_reroll_count = reroll_count
        self.last_action_actor = character.display_name
        self._apply_reroll_penalty(character, attempt, reroll_count)
        if targets is not None:
            targets_tuple = tuple(targets)
        else:
            targets_tuple = attempt.targets
        return self.attempt_action(
            character, option, targets=targets_tuple, advance_time=False
        )

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
        self.credibility.ensure_faction(self.player_faction)
        if actor_faction:
            self.credibility.ensure_faction(actor_faction)
        delta = -cost if action.related_triplet is not None else gain
        target_factions = list(targets or [actor_faction])
        for target in target_factions:
            if not target:
                continue
            self.credibility.ensure_faction(target)
            logger.debug(
                "Adjusting credibility for source=%s target=%s by %+d",
                self.player_faction,
                target,
                delta,
            )
            self.credibility.adjust(self.player_faction, target, delta)

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
        lines.append(f"Final weighted score: {self.final_weighted_score()}")
        return "<div id='state'>" + "<br>".join(lines) + "</div>"
