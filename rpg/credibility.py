"""Utilities for managing inter-faction credibility relationships."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


logger = logging.getLogger(__name__)

DEFAULT_BASE_CREDIBILITY = 50
DIAGONAL_CREDIBILITY = 100
CREDIBILITY_REWARD = 20
CREDIBILITY_PENALTY = 20

_FALLBACK_DATA: Dict[str, object] = {
    "factions": [
        "Governments",
        "Corporations",
        "HardwareManufacturers",
        "Regulators",
        "CivilSociety",
        "ScientificCommunity",
    ],
    "credibility": {
        "Governments": {
            "Governments": 100,
            "Corporations": 35,
            "HardwareManufacturers": 70,
            "Regulators": 65,
            "CivilSociety": 55,
            "ScientificCommunity": 75,
        },
        "Corporations": {
            "Governments": 45,
            "Corporations": 100,
            "HardwareManufacturers": 85,
            "Regulators": 30,
            "CivilSociety": 20,
            "ScientificCommunity": 55,
        },
        "HardwareManufacturers": {
            "Governments": 70,
            "Corporations": 80,
            "HardwareManufacturers": 100,
            "Regulators": 55,
            "CivilSociety": 35,
            "ScientificCommunity": 60,
        },
        "Regulators": {
            "Governments": 75,
            "Corporations": 25,
            "HardwareManufacturers": 65,
            "Regulators": 100,
            "CivilSociety": 70,
            "ScientificCommunity": 80,
        },
        "CivilSociety": {
            "Governments": 50,
            "Corporations": 15,
            "HardwareManufacturers": 30,
            "Regulators": 75,
            "CivilSociety": 100,
            "ScientificCommunity": 85,
        },
        "ScientificCommunity": {
            "Governments": 65,
            "Corporations": 35,
            "HardwareManufacturers": 55,
            "Regulators": 85,
            "CivilSociety": 80,
            "ScientificCommunity": 100,
        },
    },
}

_MATRIX_FILE = Path(__file__).with_name("credibility_matrix.json")


def _safe_int(value: object, default: int = DEFAULT_BASE_CREDIBILITY) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _load_initial_data() -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    factions = [str(f) for f in _FALLBACK_DATA["factions"]]  # type: ignore[list-item]
    credibility = {
        source: {target: _safe_int(value) for target, value in targets.items()}
        for source, targets in (
            _FALLBACK_DATA["credibility"].items()  # type: ignore[union-attr]
        )
    }
    try:
        with _MATRIX_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return factions, credibility
    except json.JSONDecodeError as exc:  # pragma: no cover - guarded fallback
        logger.warning("Failed to parse credibility matrix JSON: %s", exc)
        return factions, credibility
    if not isinstance(data, dict):
        logger.warning("Credibility matrix JSON root must be an object; using fallback")
        return factions, credibility
    file_factions = data.get("factions")
    file_matrix = data.get("credibility")
    if isinstance(file_factions, list):
        factions = [str(item) for item in file_factions]
    else:
        logger.warning("Missing 'factions' list in credibility matrix; using fallback order")
    if isinstance(file_matrix, dict):
        parsed: Dict[str, Dict[str, int]] = {}
        for source, targets in file_matrix.items():
            if not isinstance(targets, dict):
                continue
            parsed[str(source)] = {
                str(target): _safe_int(value)
                for target, value in targets.items()
            }
        if parsed:
            credibility = parsed
    else:
        logger.warning("Missing 'credibility' mapping in matrix; using fallback values")
    return factions, credibility


DEFAULT_FACTIONS, _DEFAULT_MATRIX = _load_initial_data()


def _clamp(value: int) -> int:
    return max(0, min(100, value))


def _coerce(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _ordered_union(primary: Iterable[str], secondary: Iterable[str]) -> List[str]:
    seen = set()
    order: List[str] = []
    for collection in (primary, secondary):
        for item in collection:
            if item not in seen:
                seen.add(item)
                order.append(item)
    return order


@dataclass
class CredibilityMatrix:
    """Track directed credibility values between factions."""

    _values: Dict[str, Dict[str, int]] = field(init=False, repr=False)
    _order: List[str] = field(init=False, repr=False)

    def __init__(self, values: Dict[str, Dict[str, int]] | None = None) -> None:
        base = values or _DEFAULT_MATRIX
        all_sources = _ordered_union(DEFAULT_FACTIONS, base.keys())
        all_targets = _ordered_union(DEFAULT_FACTIONS, {
            target for mapping in base.values() for target in mapping.keys()
        })
        factions = _ordered_union(all_sources, all_targets)
        self._order = list(factions)
        self._values = {}
        for source in factions:
            row: Dict[str, int] = {}
            provided_row = base.get(source, {})
            for target in factions:
                if source == target:
                    row[target] = DIAGONAL_CREDIBILITY
                else:
                    raw_value = provided_row.get(target, DEFAULT_BASE_CREDIBILITY)
                    row[target] = _clamp(_coerce(raw_value, DEFAULT_BASE_CREDIBILITY))
            self._values[source] = row

    def ensure_faction(self, faction: str | None) -> None:
        """Guarantee that ``faction`` is present in the matrix."""

        if not faction:
            return
        if faction not in self._order:
            self._order.append(faction)
        if faction not in self._values:
            self._values[faction] = {}
        for source in self._order:
            row = self._values.setdefault(source, {})
            for target in self._order:
                if source == target:
                    row.setdefault(target, DIAGONAL_CREDIBILITY)
                else:
                    row.setdefault(target, DEFAULT_BASE_CREDIBILITY)

    def adjust(self, source: str | None, target: str | None, delta: int) -> None:
        """Modify credibility from ``source`` to ``target`` by ``delta`` within [0, 100]."""

        if not source or not target:
            logger.warning(
                "Skipping credibility adjustment due to missing source/target: %s -> %s",
                source,
                target,
            )
            return
        if delta == 0:
            logger.warning(
                "Skipping credibility adjustment for %s -> %s because delta is zero",
                source,
                target,
            )
            return
        logger.debug(
            "Processing credibility adjustment for %s -> %s with delta %+d",
            source,
            target,
            delta,
        )
        self.ensure_faction(source)
        self.ensure_faction(target)
        row = self._values[source]
        current_value = row[target]
        logger.debug(
            "Current credibility value for %s -> %s: %d",
            source,
            target,
            current_value,
        )
        proposed_value = current_value + delta
        logger.debug(
            "Proposed credibility value after applying delta %+d: %d",
            delta,
            proposed_value,
        )
        clamped_value = _clamp(proposed_value)
        if clamped_value != proposed_value:
            logger.debug(
                "Clamped credibility value from %d to %d to enforce bounds",
                proposed_value,
                clamped_value,
            )
        row[target] = clamped_value
        if clamped_value == current_value:
            logger.info(
                "Credibility %s -> %s remained at %d after attempting delta %+d",
                source,
                target,
                current_value,
                delta,
            )
        else:
            logger.info(
                "Credibility %s -> %s changed from %d to %d (delta %+d)",
                source,
                target,
                current_value,
                clamped_value,
                delta,
            )

    def value(self, source: str, target: str) -> int:
        """Return the credibility value for ``source`` as viewed by ``target``."""

        self.ensure_faction(source)
        self.ensure_faction(target)
        return self._values[source][target]

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        """Return a deep copy of the current matrix values."""

        return {source: dict(targets) for source, targets in self._values.items()}

    @property
    def factions(self) -> List[str]:
        """Return the list of known factions in insertion order."""

        return list(self._order)


__all__ = [
    "CredibilityMatrix",
    "DEFAULT_FACTIONS",
    "DEFAULT_BASE_CREDIBILITY",
    "DIAGONAL_CREDIBILITY",
    "CREDIBILITY_REWARD",
    "CREDIBILITY_PENALTY",
]
