# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rpg.character import MarkdownCharacter

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "test_character.md")


class MarkdownCharacterTest(unittest.TestCase):
    @patch("rpg.character.genai")
    def test_generate_and_answer(self, mock_genai):
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = [
            MagicMock(text="Base context"),
            MagicMock(text="1. Where are you?\n2. Who are you?\n3. What do you do?"),
            MagicMock(text="I stand guard."),
        ]
        mock_genai.GenerativeModel.return_value = mock_model

        char = MarkdownCharacter("Tester", FIXTURE)
        questions = char.generate_questions()
        self.assertEqual(len(questions), 3)
        answer = char.answer_question(questions[0])
        self.assertEqual(answer, "I stand guard.")


if __name__ == "__main__":
    unittest.main()
