# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from web_service import create_app
from rpg.character import FolderCharacter

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "test_character")


class WebServiceTest(unittest.TestCase):
    def test_html_flow(self):
        with patch("rpg.character.genai") as mock_genai:
            mock_model = MagicMock()
            mock_model.generate_content.side_effect = [
                MagicMock(text="1. A\n2. B\n3. C"),
                MagicMock(text="Result."),
                MagicMock(text="10\n20\n30"),
            ]
            mock_genai.GenerativeModel.return_value = mock_model
            character = FolderCharacter(FIXTURE_DIR)

        with patch("web_service.load_characters", return_value=[character]):
            app = create_app()
            client = app.test_client()

            resp = client.get("/")
            page = resp.data.decode()
            self.assertEqual(resp.status_code, 200)
            self.assertIn("test_character", page)
            self.assertIn("[0, 0, 0]", page)

            resp = client.post("/actions", data={"character": "0"})
            page = resp.data.decode()
            self.assertEqual(resp.status_code, 200)
            self.assertIn("A", page)
            self.assertIn("id='state'", page)

            resp = client.post(
                "/perform", data={"character": "0", "action": "A"}
            )
            page = resp.data.decode()
            self.assertEqual(resp.status_code, 200)
            self.assertIn("Result.", page)
            self.assertIn("[10, 20, 30]", page)


if __name__ == "__main__":
    unittest.main()
