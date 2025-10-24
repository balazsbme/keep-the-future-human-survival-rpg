"""Database-backed recorder for automated game executions."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Iterable, MutableMapping

from rpg.game_state import ActionAttempt, GameState

from .sqlite3_connector import SQLiteConnector, sanitize_identifier


class GameRunObserver:
    """Minimal interface implemented by observers of game runs."""

    def on_game_start(
        self,
        state: GameState,
        *,
        player_key: str,
        player_class: str,
        automated_player_class: str,
        game_index: int,
    ) -> None:
        raise NotImplementedError

    def before_turn(self, state: GameState, round_index: int) -> None:
        raise NotImplementedError

    def after_turn(self, state: GameState, round_index: int) -> None:
        raise NotImplementedError

    def on_game_end(
        self,
        state: GameState,
        *,
        result: str,
        successful: bool,
        error: str | None = None,
    ) -> None:
        raise NotImplementedError

    def on_game_error(
        self, state: GameState | None, error: BaseException | str
    ) -> None:
        raise NotImplementedError


class GameDatabaseRecorder(GameRunObserver):
    """Persist execution metadata, actions, and assessments to SQLite."""

    def __init__(
        self,
        connector: SQLiteConnector,
        *,
        notes: str | None = None,
    ) -> None:
        self._connector = connector
        self._notes = notes
        self._execution_id: int | None = None
        self._pre_turn_snapshot: Dict[str, list[int]] | None = None
        self._cached_credibility_targets: Iterable[str] | None = None
        self._faction_triplet_counts: Dict[str, int] | None = None
        self._result_recorded = False

    # Interface implementation -----------------------------------------
    def on_game_start(
        self,
        state: GameState,
        *,
        player_key: str,
        player_class: str,
        automated_player_class: str,
        game_index: int,
    ) -> None:
        self._faction_triplet_counts = {
            faction: len(scores)
            for faction, scores in state.progress.items()
        }
        credibility_snapshot = state.credibility.snapshot()
        player_row = credibility_snapshot.get(state.player_faction, {})
        self._cached_credibility_targets = list(player_row.keys())
        self._connector.initialise()
        self._connector.ensure_columns(
            "executions",
            {
                "action_time_cost_years": "REAL",
                "format_prompt_character_limit": "INTEGER",
                "conversation_force_action_after": "INTEGER",
            },
        )
        self._connector.ensure_dynamic_schema(
            self._faction_triplet_counts,
            self._cached_credibility_targets,
        )
        config_payload = asdict(state.config)
        log_level = os.environ.get("LOG_LEVEL")
        enable_parallelism = os.environ.get("ENABLE_PARALLELISM")
        automated_agent_max_exchanges = os.environ.get(
            "AUTOMATED_AGENT_MAX_EXCHANGES"
        )
        try:
            max_exchanges_int = (
                int(automated_agent_max_exchanges)
                if automated_agent_max_exchanges is not None
                else None
            )
        except ValueError:
            max_exchanges_int = None
        metadata = {
            "player_class": player_class,
            "automated_player_class": automated_player_class,
            "config_json": json.dumps(config_payload, sort_keys=True),
            "log_level": log_level,
            "enable_parallelism": enable_parallelism,
            "automated_agent_max_exchanges": max_exchanges_int,
            "scenario": state.config.scenario,
            "win_threshold": state.config.win_threshold,
            "max_rounds": state.config.max_rounds,
            "roll_success_threshold": state.config.roll_success_threshold,
            "action_time_cost_years": state.config.action_time_cost_years,
            "format_prompt_character_limit": state.config.format_prompt_character_limit,
            "conversation_force_action_after": state.config.conversation_force_action_after,
            "notes": self._notes or f"{player_key}-game-{game_index}-{datetime.utcnow().isoformat()}",
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}
        self._execution_id = self._connector.insert_execution(metadata)
        self._connector.commit()
        self._result_recorded = False

    def before_turn(self, state: GameState, round_index: int) -> None:
        self._pre_turn_snapshot = self._snapshot_progress(state)

    def after_turn(self, state: GameState, round_index: int) -> None:
        if self._execution_id is None:
            return
        attempt = state.last_action_attempt
        if attempt is None:
            return
        action_id = self._record_action(state, attempt, round_index)
        self._record_assessment(state, action_id)
        self._record_credibility(state, attempt, action_id)
        self._connector.commit()
        self._pre_turn_snapshot = self._snapshot_progress(state)

    def on_game_end(
        self,
        state: GameState,
        *,
        result: str,
        successful: bool,
        error: str | None = None,
    ) -> None:
        if self._execution_id is not None:
            self._record_result(
                successful_execution=successful,
                result=result,
                error_info=error,
            )
            self._connector.commit()
        self._reset()

    def on_game_error(
        self, state: GameState | None, error: BaseException | str
    ) -> None:
        if self._execution_id is None:
            self._reset()
            return
        self._record_result(
            successful_execution=False,
            result="N/A",
            error_info=str(error),
        )
        self._connector.commit()
        self._reset()

    # Internal helpers ---------------------------------------------------
    def _snapshot_progress(self, state: GameState) -> Dict[str, list[int]]:
        return {key: list(values) for key, values in state.progress.items()}

    def _record_action(self, state: GameState, attempt: ActionAttempt, round_index: int) -> int:
        option_payload = attempt.option.to_payload()
        data: MutableMapping[str, object] = {
            "execution_id": self._execution_id,
            "actor": state.last_action_actor,
            "title": attempt.label,
            "option_text": attempt.option.text,
            "option_type": attempt.option.type,
            "related_triplet": attempt.option.related_triplet,
            "related_attribute": attempt.option.related_attribute,
            "success": int(attempt.success),
            "roll_total": attempt.roll + attempt.effective_score,
            "actor_score": attempt.actor_score,
            "player_score": attempt.player_score,
            "effective_score": attempt.effective_score,
            "credibility_cost": attempt.credibility_cost,
            "credibility_gain": attempt.credibility_gain,
            "targets_json": list(attempt.targets),
            "failure_text": attempt.failure_text,
            "round_number": round_index,
            "option_json": option_payload,
        }
        return self._connector.insert_action(data)

    def _record_assessment(self, state: GameState, action_id: int) -> None:
        if self._faction_triplet_counts is None:
            self._faction_triplet_counts = {
                faction: len(scores)
                for faction, scores in state.progress.items()
            }
        assessment_data: Dict[str, object] = {
            "execution_id": self._execution_id,
            "action_id": action_id,
            "scenario": state.config.scenario,
            "final_weighted_score": state.final_weighted_score(),
        }
        detailed_scores: Dict[str, Dict[str, int]] = {}
        for faction, scores in state.progress.items():
            faction_key = sanitize_identifier(faction)
            faction_detail: Dict[str, int] = {}
            for index, score in enumerate(scores, 1):
                column = f"{faction_key}_triplet_{index}"
                assessment_data[column] = score
                faction_detail[str(index)] = score
            detailed_scores[faction] = faction_detail
        assessment_data["assessment_json"] = {
            "before": self._pre_turn_snapshot,
            "after": detailed_scores,
        }
        self._connector.insert_assessment(assessment_data)

    def _record_credibility(self, state: GameState, attempt: ActionAttempt, action_id: int) -> None:
        snapshot = state.credibility.snapshot()
        player_row = snapshot.get(state.player_faction, {})
        if self._cached_credibility_targets is None:
            self._cached_credibility_targets = list(player_row.keys())
        credibility_data: Dict[str, object] = {
            "execution_id": self._execution_id,
            "action_id": action_id,
            "cost": attempt.credibility_cost,
            "reroll_attempt_count": state.last_reroll_count,
            "credibility_json": player_row,
        }
        for target in self._cached_credibility_targets:
            column = f"credibility_{sanitize_identifier(target)}"
            credibility_data[column] = player_row.get(target)
        self._connector.insert_credibility(credibility_data)

    def _record_result(
        self,
        *,
        successful_execution: bool,
        result: str | None,
        error_info: str | None,
    ) -> None:
        if self._execution_id is None or self._result_recorded:
            return
        payload: Dict[str, object] = {
            "execution_id": self._execution_id,
            "successful_execution": successful_execution,
            "result": result or "N/A",
        }
        if error_info:
            payload["error_info"] = error_info
        self._connector.insert_result(payload)
        self._result_recorded = True

    def _reset(self) -> None:
        self._execution_id = None
        self._pre_turn_snapshot = None
        self._cached_credibility_targets = None
        self._faction_triplet_counts = None
        self._result_recorded = False


__all__ = ["GameRunObserver", "GameDatabaseRecorder"]
