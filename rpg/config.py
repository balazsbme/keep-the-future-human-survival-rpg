"""Configuration loading utilities for game parameters."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GameConfig:
    """Container for gameplay configuration values."""

    scenario: str = "complete"
    win_threshold: int = 71
    max_rounds: int = 10
    roll_success_threshold: int = 10
    action_time_cost_years: float = 0.5
    format_prompt_character_limit: int = 400
    conversation_force_action_after: int = 8


_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "game_config.yaml"
)


def _coerce_int(value: Any, fallback: int) -> int:
    """Return ``value`` coerced to ``int`` when possible."""

    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid integer value %r encountered in configuration; using %d",
            value,
            fallback,
        )
        return fallback


def _coerce_float(value: Any, fallback: float) -> float:
    """Return ``value`` coerced to ``float`` when possible."""

    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid float value %r encountered in configuration; using %s",
            value,
            fallback,
        )
        return fallback


def load_game_config(path: str | None = None) -> GameConfig:
    """Load the gameplay configuration from ``path`` if available."""

    config_path = path or _DEFAULT_CONFIG_PATH
    data: Dict[str, Any] = {}
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        logger.warning(
            "Game configuration file %s not found; falling back to defaults",
            config_path,
        )
        return GameConfig()
    except yaml.YAMLError as exc:
        logger.warning(
            "Failed to parse game configuration %s: %s; using defaults",
            config_path,
            exc,
        )
        return GameConfig()
    if isinstance(payload, dict):
        if "game" in payload and isinstance(payload["game"], dict):
            data = payload["game"]
        else:
            data = payload
    scenario = str(data.get("scenario", "complete")).strip() or "complete"
    win_threshold = _coerce_int(data.get("win_threshold", 71), 71)
    max_rounds = _coerce_int(data.get("max_rounds", 10), 10)
    roll_success_threshold = _coerce_int(data.get("roll_success_threshold", 10), 10)
    action_time_cost_years = _coerce_float(
        data.get("action_time_cost_years", 0.5), 0.5
    )

    char_limit = _coerce_int(
        data.get("format_prompt_character_limit", 400), 400
    )

    conversation_force_action_after = _coerce_int(
        data.get("conversation_force_action_after", 8), 8
    )
    return GameConfig(
        scenario=scenario.lower(),
        win_threshold=max(0, win_threshold),
        max_rounds=max(1, max_rounds),
        roll_success_threshold=max(1, roll_success_threshold),
        action_time_cost_years=max(0.0, action_time_cost_years),
        format_prompt_character_limit=max(1, char_limit),
        conversation_force_action_after=max(0, conversation_force_action_after),
    )


__all__ = ["GameConfig", "load_game_config"]
