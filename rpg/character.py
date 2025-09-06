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
    def generate_questions(self) -> List[str]:
        """Return three possible questions a player might ask."""

    @abstractmethod
    def answer_question(self, question: str) -> str:
        """Return the character's answer to ``question``."""


class MarkdownCharacter(Character):
    """Character defined by a Markdown file sent to Gemini."""

    def __init__(self, name: str, md_path: str, model: str = "gemini-2.5-flash"):
        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()
        super().__init__(name, text, model)
        # Generate a base context using the description
        self.base_context = self._model.generate_content(text).text

    def generate_questions(self) -> List[str]:
        prompt = (
            f"{self.base_context}\n"
            "List three numbered questions a player might ask you." 
        )
        response = self._model.generate_content(prompt)
        lines = [line.strip() for line in response.text.splitlines() if line.strip()]
        questions: List[str] = []
        for line in lines:
            if line[0].isdigit():
                parts = line.split(".", 1)
                q = parts[1].strip() if len(parts) > 1 else line
                questions.append(q)
            else:
                if questions:
                    questions.append(line)
            if len(questions) == 3:
                break
        return questions

    def answer_question(self, question: str) -> str:
        prompt = f"{self.base_context}\nPlayer: {question}\n{self.name}:"
        response = self._model.generate_content(prompt)
        return response.text.strip()
