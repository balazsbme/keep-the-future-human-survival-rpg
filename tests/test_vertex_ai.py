import os
import unittest

import vertexai
from google.auth.exceptions import GoogleAuthError
from dotenv import load_dotenv
from vertexai.preview.language_models import TextEmbeddingModel


class VertexAIIntegrationTest(unittest.TestCase):
    """Simple integration test that queries Vertex AI for embeddings."""

    def test_vertex_ai_embeddings(self):
        load_dotenv()
        project = os.getenv("GCP_PROJECT_ID")
        location = os.getenv("GCP_REGION")
        if not project or not location:
            self.skipTest("GCP_PROJECT_ID and GCP_REGION must be set to run this test")
        vertexai.init(project=project, location=location)
        try:
            model = TextEmbeddingModel.from_pretrained("text-embedding-005")
            embeddings = model.get_embeddings(["hello world"])
            self.assertEqual(len(embeddings), 1)
            self.assertGreater(len(embeddings[0].values), 0)
        except GoogleAuthError as exc:
            self.skipTest(f"Google Cloud authentication failed: {exc}")


if __name__ == "__main__":
    unittest.main()
