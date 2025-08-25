"""Minimal RAG pipeline for OpenNebula documentation using Vertex AI."""
from __future__ import annotations

import os
import logging
from typing import List

import requests  # Used in tests
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from google import genai
from google.genai.types import EmbedContentConfig
from google.cloud.sql.connector import Connector
import sqlalchemy
from pgvector.sqlalchemy import Vector

if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

class VertexAIEmbeddings(Embeddings):
    """Wrapper around Vertex AI text embedding model for LangChain."""

    def __init__(self, *, client: genai.Client | None = None, model_name: str = "text-embedding-005") -> None:
        self._client = client or genai.Client(vertexai=True)
        self._model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logging.debug("Embedding %d document(s) with model %s", len(texts), self._model_name)
        response = self._client.models.embed_content(
            model=self._model_name,
            contents=texts,
            config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        vectors = [emb.values for emb in response.embeddings]
        logging.debug("Generated %d embedding(s)", len(vectors))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        logging.info("Embedding user query")
        return self.embed_documents([text])[0]


def init_engine() -> sqlalchemy.engine.Engine:
    """Create a SQLAlchemy engine for the Cloud SQL database."""
    logging.info("Initializing Cloud SQL engine")
    connection_name = os.environ["CLOUD_SQL_CONNECTION_NAME"]
    db_user = os.environ["DB_USER"]
    db_pass = os.environ["DB_PASS"]
    db_name = os.environ["DB_NAME"]

    connector = Connector()

    def getconn() -> any:
        return connector.connect(connection_name, "pg8000", user=db_user, password=db_pass, db=db_name)

    engine = sqlalchemy.create_engine("postgresql+pg8000://", creator=getconn)
    logging.debug("Cloud SQL engine initialized")
    return engine


def retrieve_chunks_db(
    engine: sqlalchemy.engine.Engine, query_embedding: List[float], k: int = 3
) -> List[tuple[str, str]]:
    """Return top ``k`` document texts and references from Cloud SQL."""
    logging.info("Retrieving top %d document chunk(s) from database", k)
    documents = sqlalchemy.Table(
        "documents",
        sqlalchemy.MetaData(),
        sqlalchemy.Column("content", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("embedding", Vector()),
        sqlalchemy.Column("reference", sqlalchemy.Text, nullable=False),
    )
    with engine.connect() as conn:
        stmt = (
            sqlalchemy.select(documents.c.content, documents.c.reference)
            .order_by(documents.c.embedding.cosine_distance(query_embedding))
            .limit(k)
        )
        rows = [(row[0], row[1]) for row in conn.execute(stmt)]
        logging.debug("Retrieved %d chunk(s) from database", len(rows))
        return rows


def generate_answer(question: str, context: str, *, client: genai.Client) -> str:
    """Generate an answer to ``question`` given ``context`` using Vertex AI."""
    prompt = (
        f"You are OpenNebula software engineer and solution architect, with a deep understanding of virtualization technologies, Linux networking and administration."
        "\n\n Answer the question based ONLY on the provided context.\n\n"
        "CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    )
    logging.info("Generating answer with LLM")
    response = client.models.generate_content(contents=[prompt], model="gemini-2.5-flash")
    answer = response.text
    logging.debug("LLM returned %d characters", len(answer))
    return answer


def build_rag_and_answer(query: str, k: int = 3) -> str:
    """High-level helper that performs the full RAG pipeline using Cloud SQL."""
    logging.info("Starting RAG pipeline for query: %s", query)
    load_dotenv()
    logging.debug("Environment variables loaded")
    client = genai.Client()
    embedder = VertexAIEmbeddings(client=client)
    logging.debug("Vertex AI embeddings client initialized")
    engine = init_engine()
    query_embedding = embedder.embed_query(query)
    logging.debug("Query embedding length: %d", len(query_embedding))
    rows = retrieve_chunks_db(engine, query_embedding, k=k)
    logging.info("Retrieved %d chunk(s) for context", len(rows))
    chunks = [row[0] for row in rows]
    references = [str(row[1]) for row in rows if row[1] is not None]
    context = "\n\n".join(chunks)
    answer = generate_answer(query, context, client=client)
    if os.getenv("APPEND_REFERENCES", "false").lower() in {"1", "true", "yes"} and references:
        refs_text = "\n\nReferences:\n" + "\n".join(references)
        answer = f"{answer}{refs_text}"
    logging.info("RAG pipeline completed for query: %s", query)
    return answer


if __name__ == "__main__":
    USER_QUERY = "What should I pay attention to when planning to hot-plug devices to the VM?"
    answer = build_rag_and_answer(USER_QUERY)
    print("Question:", USER_QUERY)
    print("Answer:", answer)
