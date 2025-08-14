import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from langchain_core.embeddings import Embeddings
from rag_pipeline import chunk_text, build_vector_store, retrieve_chunks


class DummyEmbeddings(Embeddings):
    """Deterministic embedding model for tests."""

    def embed_documents(self, texts):
        return [[float(len(t))] for t in texts]

    def embed_query(self, text):
        return [float(len(text))]


class RAGPipelineTests(unittest.TestCase):
    def test_chunk_text_respects_size(self):
        text = "a" * 2500
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=100)
        self.assertGreaterEqual(len(chunks), 3)
        for ch in chunks:
            self.assertLessEqual(len(ch), 1000)

    def test_vector_store_retrieval(self):
        docs = ["alpha", "beta", "gamma gamma"]
        store = build_vector_store(docs, embedding_model=DummyEmbeddings())
        retrieved = retrieve_chunks(store, "alpha", k=1)
        self.assertEqual(retrieved[0], "alpha")


if __name__ == "__main__":
    unittest.main()
