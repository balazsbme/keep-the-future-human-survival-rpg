"""Helpers for collapsing verbose sections in debug logging output."""

# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import os
import re
from functools import lru_cache


@lru_cache(maxsize=1)
def collapse_prompt_sections_enabled() -> bool:
    """Return ``True`` when repeated prompt sections should be collapsed."""

    value = os.environ.get("COLLAPSE_PROMPT_SECTIONS_IN_DEBUG_LOGS", "1")
    return value.strip().lower() in {"1", "true", "yes", "on"}


_SECTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Collapse repeated How-to-win guide inserts.
    (
        re.compile(
            r"(Use the following guide to win:\s*)(.*?)(\nCharacter:)",
            re.DOTALL,
        ),
        "[HOW-TO-WIN GUIDE]",
    ),
    (
        re.compile(
            r"(Use the victory guide:\s*)(.*?)(\nCharacter:)",
            re.DOTALL,
        ),
        "[HOW-TO-WIN GUIDE]",
    ),
    (
        re.compile(
            r"(The baseline script:\s*)(.*?)(\nAssess all triplets)",
            re.DOTALL,
        ),
        "[HOW-TO-WIN GUIDE]",
    ),
    # Collapse faction context markdown blocks.
    (
        re.compile(
            r"(Faction context:\s*)(.*?)(\nConversation so far:)",
            re.DOTALL,
        ),
        "[FACTION CONTEXT]",
    ),
    (
        re.compile(
            r"(\*\*MarkdownContext\*\*\n)(.*?)(\n\*\*End of MarkdownContext\*\*)",
            re.DOTALL,
        ),
        "[FACTION CONTEXT]",
    ),
    (
        re.compile(
            r"([A-Za-z ]+ context:\n)(.*?)(\n(?:Scenario summary|Use the cached|Your profile|$))",
            re.DOTALL,
        ),
        "[FACTION CONTEXT]",
    ),
    # Collapse persona summaries for both NPCs and the player.
    (
        re.compile(
            r"(Persona for [^\n]+:\n)(.*?)(\n(?:MarkdownContext|Triplet definitions|Scenario summary|Use the cached|$))",
            re.DOTALL,
        ),
        "[PERSONA DETAILS]",
    ),
    (
        re.compile(
            r"(Your persona is described below:\n)(.*?)(\nGround your thinking)",
            re.DOTALL,
        ),
        "[PERSONA DETAILS]",
    ),
    (
        re.compile(
            r"(Player persona overview:\n)(.*?)(\nPlayer profile details:)",
            re.DOTALL,
        ),
        "[PERSONA DETAILS]",
    ),
    (
        re.compile(
            r"(Player profile details:\n)(.*?)(\n(?:[A-Za-z ]+ context:|Scenario summary:|Use the cached|$))",
            re.DOTALL,
        ),
        "[PERSONA DETAILS]",
    ),
    (
        re.compile(
            r"(Your profile:\n)(.*?)(\n)",
            re.DOTALL,
        ),
        "[PERSONA DETAILS]",
    ),
)


def _collapse_sections(text: str) -> str:
    """Replace verbose sections in ``text`` with short placeholders."""

    result = text
    for pattern, placeholder in _SECTION_PATTERNS:
        def _repl(match: re.Match[str]) -> str:
            groups = match.groups()
            if len(groups) >= 3:
                prefix, _, suffix = groups[0], groups[1], groups[2]
                return f"{prefix}{placeholder}{suffix}"
            return match.group(0)

        result = pattern.sub(_repl, result)
    return result


def collapse_prompt_sections(text: str) -> str:
    """Return ``text`` with verbose sections collapsed when configured."""

    if not collapse_prompt_sections_enabled():
        return text
    return _collapse_sections(text)


__all__ = ["collapse_prompt_sections_enabled", "collapse_prompt_sections"]
