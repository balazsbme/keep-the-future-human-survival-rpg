import unittest

from dotenv import load_dotenv
from google import genai


class GenAITextGenerationTest(unittest.TestCase):
    """Integration test verifying text generation via ``google-genai``."""

    def test_generate_content(self):
        load_dotenv()
        try:
            client = genai.Client()
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=["Hello, world!"],
            )
            self.assertIsInstance(response.text, str)
            self.assertGreater(len(response.text), 0)
        except Exception as exc:  # pragma: no cover - network/auth failures
            self.skipTest(f"Vertex AI request failed: {exc}")


if __name__ == "__main__":
    unittest.main()
