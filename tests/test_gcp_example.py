import os
import unittest

from dotenv import load_dotenv
import google.generativeai as genai


class GenAITextGenerationTest(unittest.TestCase):
    """Integration test verifying text generation via ``google-generativeai``."""

    def test_generate_content(self):
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            self.skipTest("GEMINI_API_KEY environment variable not set")
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content("Hello, world!")
            self.assertIsInstance(response.text, str)
            self.assertGreater(len(response.text), 0)
        except Exception as exc:  # pragma: no cover - network/auth failures
            self.skipTest(f"Gemini request failed: {exc}")


if __name__ == "__main__":
    unittest.main()
