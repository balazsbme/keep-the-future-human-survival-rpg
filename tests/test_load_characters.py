from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cli_game import load_characters
from rpg.config import GameConfig


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _scenario_payload(factions: dict) -> dict:
    return {"factions": factions}


def _faction_block(summary: str) -> dict:
    return {
        "ScenarioSummary": summary,
        "initial_states": ["initial"],
        "end_states": ["end"],
        "gaps": [{"severity": "Critical", "explanation": "gap"}],
        "referenced_quotes": [f"Quote for {summary}"],
    }


def test_load_characters_filters_disabled_factions(tmp_path: Path) -> None:
    characters_path = tmp_path / "characters.yaml"
    scenario_path = tmp_path / "scenario.yaml"
    context_path = tmp_path / "context.yaml"

    characters_payload = {
        "Characters": [
            {"name": "Enabled NPC", "faction": "Enabled"},
            {"name": "Disabled NPC", "faction": "Disabled"},
        ]
    }
    scenario_payload = _scenario_payload(
        {
            "Enabled": _faction_block("Enabled faction"),
            "Disabled": _faction_block("Disabled faction"),
        }
    )
    context_payload = _scenario_payload(
        {
            "Enabled": {"MarkdownContext": "Context for enabled"},
            "Disabled": {"MarkdownContext": "Context for disabled"},
        }
    )

    _write_yaml(characters_path, characters_payload)
    _write_yaml(scenario_path, scenario_payload)
    _write_yaml(context_path, context_payload)

    config = GameConfig(enabled_factions=("Enabled",))
    roster = load_characters(
        character_file=str(characters_path),
        scenario_file=str(scenario_path),
        factions_file=str(context_path),
        config=config,
    )

    assert [character.faction for character in roster] == ["Enabled"]
    assert roster[0].referenced_quotes == ["Quote for Enabled faction"]


def test_load_characters_missing_scenario_for_enabled_logs_error(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    characters_path = tmp_path / "characters.yaml"
    scenario_path = tmp_path / "scenario.yaml"
    context_path = tmp_path / "context.yaml"

    _write_yaml(
        characters_path,
        {"Characters": [{"name": "Enabled NPC", "faction": "Enabled"}]},
    )
    _write_yaml(
        scenario_path,
        _scenario_payload({"Enabled": _faction_block("Enabled faction")}),
    )
    _write_yaml(context_path, _scenario_payload({}))

    config = GameConfig(enabled_factions=("Enabled", "Missing"))

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError):
            load_characters(
                character_file=str(characters_path),
                scenario_file=str(scenario_path),
                factions_file=str(context_path),
                config=config,
            )
    error_messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "Missing" in error_messages


def test_load_characters_missing_characters_for_enabled_raises(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    characters_path = tmp_path / "characters.yaml"
    scenario_path = tmp_path / "scenario.yaml"
    context_path = tmp_path / "context.yaml"

    _write_yaml(
        characters_path,
        {"Characters": [{"name": "Enabled NPC", "faction": "Enabled"}]},
    )
    _write_yaml(
        scenario_path,
        _scenario_payload(
            {
                "Enabled": _faction_block("Enabled faction"),
                "Missing": _faction_block("Missing faction"),
            }
        ),
    )
    _write_yaml(context_path, _scenario_payload({}))

    config = GameConfig(enabled_factions=("Enabled", "Missing"))

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError):
            load_characters(
                character_file=str(characters_path),
                scenario_file=str(scenario_path),
                factions_file=str(context_path),
                config=config,
            )
    error_messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "Missing" in error_messages


def test_load_characters_with_all_factions_real_data() -> None:
    config = GameConfig(
        enabled_factions=(
            "Governments",
            "Corporations",
            "HardwareManufacturers",
            "Regulators",
            "CivilSociety",
            "ScientificCommunity",
        ),
        scenario="01-race-to-contain-power",
    )
    roster = load_characters(config=config)

    factions = {character.faction for character in roster}
    assert factions == set(config.enabled_factions)
    assert all(getattr(character, "scenario_summary", "") for character in roster)
    assert all(getattr(character, "referenced_quotes", None) for character in roster)


def test_load_characters_with_subset_of_factions_real_data() -> None:
    config = GameConfig(
        enabled_factions=("Governments", "CivilSociety"),
        scenario="01-race-to-contain-power",
    )
    roster = load_characters(config=config)

    factions = {character.faction for character in roster}
    assert factions == set(config.enabled_factions)
    assert len(roster) == len(config.enabled_factions)
    assert all(getattr(character, "scenario_summary", "") for character in roster)
    assert all(getattr(character, "referenced_quotes", None) for character in roster)
