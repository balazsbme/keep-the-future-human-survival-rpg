import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from langchain_core.embeddings import Embeddings
from ingest_opennebula_docs import scrape_urls
from rag_pipeline import build_rag_and_answer
from unittest.mock import patch


class DummyEmbeddings(Embeddings):
    """Deterministic embedding model for tests."""

    def embed_documents(self, texts):
        return [[float(len(t))] for t in texts]

    def embed_query(self, text):
        return [float(len(text))]


class RAGPipelineTests(unittest.TestCase):

    @patch("rag_pipeline.requests.get")
    def test_scrape_urls_fallback_to_body(self, mock_get):
        class DummyResponse:
            status_code = 200

            def __init__(self, text):
                self.text = text

            def raise_for_status(self):
                pass

        html = "<html><body><p>Hello World!</p></body></html>"
        mock_get.return_value = DummyResponse(html)
        text = scrape_urls(["http://example.com"])
        self.assertIn("Hello World!", text)

    @patch("rag_pipeline.generate_answer", return_value="dummy answer")
    @patch(
        "rag_pipeline.retrieve_chunks_db",
        return_value=[("content1", "ref1"), ("content2", "ref2")],
    )
    @patch("rag_pipeline.VertexAIEmbeddings")
    @patch("rag_pipeline.genai.Client")
    @patch("rag_pipeline.init_engine")
    def test_build_rag_and_answer_appends_references(
        self,
        mock_engine,
        mock_client,
        mock_embed_cls,
        mock_retrieve,
        mock_generate,
    ):
        mock_embed = mock_embed_cls.return_value
        mock_embed.embed_query.return_value = [0.1]
        os.environ["APPEND_REFERENCES"] = "true"
        try:
            answer = build_rag_and_answer("question", k=2)
        finally:
            os.environ.pop("APPEND_REFERENCES", None)
        self.assertIn("dummy answer", answer)
        self.assertIn("ref1", answer)
        self.assertIn("ref2", answer)

    @patch("rag_pipeline.generate_answer", return_value="dummy answer")
    @patch("rag_pipeline.retrieve_chunks_db", return_value=[("content1",), ("content2",)])
    @patch("rag_pipeline.VertexAIEmbeddings")
    @patch("rag_pipeline.genai.Client")
    @patch("rag_pipeline.init_engine")
    def test_build_rag_and_answer_handles_missing_references(
        self,
        mock_engine,
        mock_client,
        mock_embed_cls,
        mock_retrieve,
        mock_generate,
    ):
        mock_embed = mock_embed_cls.return_value
        mock_embed.embed_query.return_value = [0.1]
        os.environ["APPEND_REFERENCES"] = "true"
        try:
            answer = build_rag_and_answer("question", k=2)
        finally:
            os.environ.pop("APPEND_REFERENCES", None)
        self.assertIn("dummy answer", answer)
        self.assertNotIn("References:", answer)


if __name__ == "__main__":
    unittest.main()
