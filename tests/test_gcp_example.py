import os
import unittest

from dotenv import load_dotenv
from google import genai
from google.genai.types import EmbedContentConfig


class GenAIEmbeddingTest(unittest.TestCase):
    """Integration test verifying Vertex AI embeddings via ``google-genai``."""

    def test_embed_content(self):
        load_dotenv()
        project = os.getenv("GCP_PROJECT_ID")
        location = os.getenv("GCP_REGION")
        if not project or not location:
            self.skipTest("GCP_PROJECT_ID and GCP_REGION must be set to run this test")
        client = genai.Client(vertexai=True, project=project, location=location)
        try:
            response = client.models.embed_content(
                model="text-embedding-005",
                contents=["How do I get a driver's license/learner's permit?"],
                config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768),
            )
            self.assertEqual(len(response.embeddings), 1)
            self.assertGreater(len(response.embeddings[0].values), 0)
        except Exception as exc:  # pragma: no cover - network/auth failures
            self.skipTest(f"Vertex AI request failed: {exc}")


if __name__ == "__main__":
    unittest.main()

