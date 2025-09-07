# SPDX-License-Identifier: GPL-3.0-or-later

import os
from abc import ABC, abstractmethod
from typing import List

try:  # pragma: no cover - optional dependency
    import google.generativeai as genai
except ModuleNotFoundError:  # pragma: no cover
    genai = None


class Character(ABC):
    """Abstract base class defining character interactions."""

    def __init__(self, name: str, context: str, model: str = "gemini-2.5-flash"):
        self.name = name
        self.context = context
        if genai is None:  # pragma: no cover - env without dependency
            raise ModuleNotFoundError("google-generativeai not installed")
        self._model = genai.GenerativeModel(model)

    @abstractmethod
    def generate_actions(self) -> List[str]:
        """Return three possible actions a player might request."""

    @abstractmethod
    def perform_action(self, action: str) -> str:
        """Return the result of the character performing ``action``."""


class MarkdownCharacter(Character):
    """Character defined by a Markdown file sent to Gemini."""

    def __init__(self, name: str, md_path: str, model: str = "gemini-2.5-flash"):
        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()
        super().__init__(name, text, model)
        # Generate a base context using the description
        self.base_context = self._model.generate_content(text).text

    def generate_actions(self) -> List[str]:
        prompt = (
            f"{self.base_context}\n"
            "List three numbered actions a player might ask you to perform."
        )
        response = self._model.generate_content(prompt)
        lines = [line.strip() for line in response.text.splitlines() if line.strip()]
        actions: List[str] = []
        for line in lines:
            if line[0].isdigit():
                parts = line.split(".", 1)
                act = parts[1].strip() if len(parts) > 1 else line
                actions.append(act)
            else:
                if actions:
                    actions.append(line)
            if len(actions) == 3:
                break
        return actions

    def perform_action(self, action: str) -> str:
        prompt = f"{self.base_context}\nPlayer requests: {action}\n{self.name}:"
        response = self._model.generate_content(prompt)
        return response.text.strip()
