# SPDX-License-Identifier: GPL-3.0-or-later

"""Character abstractions backed by folder-defined context."""

import os
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import yaml

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
    def generate_actions(self, history: List[Tuple[str, str]]) -> List[str]:
        """Return three possible actions a player might request."""

    @abstractmethod
    def perform_action(
        self, action: str, history: List[Tuple[str, str]]
    ) -> Tuple[str, List[int]]:
        """Return result of performing ``action`` and triplet progress."""


class FolderCharacter(Character):
    """Character defined by a folder containing context and yaml lists."""

    def __init__(self, folder: str, model: str = "gemini-2.5-flash"):
        name = os.path.basename(folder.rstrip(os.sep))

        md_files = [f for f in os.listdir(folder) if f.endswith(".md")]
        if not md_files:
            raise FileNotFoundError("character folder missing markdown context")
        md_path = os.path.join(folder, md_files[0])
        with open(md_path, "r", encoding="utf-8") as f:
            base_context = f.read()

        def load_yaml(filename: str) -> List[Any]:
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or []
            if not isinstance(data, list):
                raise ValueError(f"{filename} must contain a list")
            return data

        self.current = load_yaml("current.yaml")
        self.conditions = load_yaml("conditions.yaml")
        self.gaps = load_yaml("gaps.yaml")
        self.triplets = list(zip(self.current, self.conditions, self.gaps))
        super().__init__(name, base_context, model)
        self.base_context = base_context

    def _triplet_text(self) -> str:
        lines = []
        for idx, (cur, cond, gap) in enumerate(self.triplets, 1):
            gap_text = gap if isinstance(gap, str) else gap.get("explanation", str(gap))
            lines.append(
                f"{idx}. Current: {cur}\n   Condition: {cond}\n   Gap: {gap_text}"
            )
        return "\n".join(lines)

    def _history_text(self, history: List[Tuple[str, str]]) -> str:
        return "\n".join(f"{actor}: {act}" for actor, act in history) or "None"

    def generate_actions(self, history: List[Tuple[str, str]]) -> List[str]:
        prompt = (
            f"{self.base_context}\n{self._triplet_text()}\n"
            f"Previous actions:\n{self._history_text(history)}\n"
            "List three numbered actions you might take next."
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

    def perform_action(
        self, action: str, history: List[Tuple[str, str]]
    ) -> Tuple[str, List[int]]:
        full_history = history + [(self.name, action)]
        context_block = (
            f"{self.base_context}\n{self._triplet_text()}\n"
            f"Action history:\n{self._history_text(full_history)}\n"
        )
        action_prompt = f"{context_block}Player requests: {action}\n{self.name}:"
        response = self._model.generate_content(action_prompt)
        result = response.text.strip()

        assess_prompt = (
            f"{context_block}"
            "Provide progress (0-100) for each triplet on separate lines."
        )
        assess_resp = self._model.generate_content(assess_prompt)
        scores: List[int] = []
        for line in assess_resp.text.splitlines():
            line = line.strip()
            if not line:
                continue
            num = ''.join(ch for ch in line if ch.isdigit())
            if not num:
                continue
            scores.append(max(0, min(100, int(num))))
            if len(scores) == len(self.triplets):
                break
        return result, scores
