import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from web_service import create_app
from rpg.character import MarkdownCharacter

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "test_character.md")


class WebServiceTest(unittest.TestCase):
    def test_html_flow(self):
        with patch("rpg.character.genai") as mock_genai:
            mock_model = MagicMock()
            mock_model.generate_content.side_effect = [
                MagicMock(text="Base context"),
                MagicMock(text="1. Where are you?\n2. Who are you?\n3. What do you do?"),
                MagicMock(text="I stand guard."),
            ]
            mock_genai.GenerativeModel.return_value = mock_model

            character = MarkdownCharacter("Tester", FIXTURE)

        with patch("web_service.load_characters", return_value=[character]):
            app = create_app()
            client = app.test_client()

            resp = client.get("/")
            self.assertEqual(resp.status_code, 200)
            self.assertIn("Tester", resp.data.decode())

            resp = client.post("/questions", data={"character": "0"})
            self.assertEqual(resp.status_code, 200)
            self.assertIn("Where are you?", resp.data.decode())

            resp = client.post(
                "/answer", data={"character": "0", "question": "Where are you?"}
            )
            self.assertEqual(resp.status_code, 200)
            self.assertIn("I stand guard.", resp.data.decode())


if __name__ == "__main__":
    unittest.main()
