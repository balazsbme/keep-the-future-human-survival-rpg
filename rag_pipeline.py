"""Minimal RAG pipeline for OpenNebula documentation using Vertex AI."""
from __future__ import annotations

import os
import logging
from typing import Iterable, List

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from google import genai
from google.genai.types import EmbedContentConfig
from google.cloud.sql.connector import Connector
import sqlalchemy
from pgvector.sqlalchemy import Vector

class VertexAIEmbeddings(Embeddings):
    """Wrapper around Vertex AI text embedding model for LangChain."""

    def __init__(self, *, client: genai.Client | None = None, model_name: str = "text-embedding-005") -> None:
        self._client = client or genai.Client(vertexai=True)
        self._model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self._client.models.embed_content(
            model=self._model_name,
            contents=texts,
            config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return [emb.values for emb in response.embeddings]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def init_engine() -> sqlalchemy.engine.Engine:
    """Create a SQLAlchemy engine for the Cloud SQL database."""
    connection_name = os.environ["CLOUD_SQL_CONNECTION_NAME"]
    db_user = os.environ["DB_USER"]
    db_pass = os.environ["DB_PASS"]
    db_name = os.environ["DB_NAME"]

    connector = Connector()

    def getconn() -> any:
        return connector.connect(connection_name, "pg8000", user=db_user, password=db_pass, db=db_name)

    return sqlalchemy.create_engine("postgresql+pg8000://", creator=getconn)


def retrieve_chunks_db(
    engine: sqlalchemy.engine.Engine, query_embedding: List[float], k: int = 3
) -> List[str]:
    """Return top ``k`` document texts from Cloud SQL similar to ``query_embedding``."""
    documents = sqlalchemy.Table(
        "documents",
        sqlalchemy.MetaData(),
        sqlalchemy.Column("content", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("embedding", Vector()),
    )
    with engine.connect() as conn:
        stmt = (
            sqlalchemy.select(documents.c.content)
            .order_by(documents.c.embedding.cosine_distance(query_embedding))
            .limit(k)
        )
        return [row[0] for row in conn.execute(stmt)]


def generate_answer(question: str, context: str, *, client: genai.Client) -> str:
    """Generate an answer to ``question`` given ``context`` using Vertex AI."""
    prompt = (
        "Answer the question based ONLY on the provided context.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    )
    response = client.models.generate_content(contents=[prompt], model="gemini-2.5-flash")
    return response.text


def build_rag_and_answer(query: str, k: int = 3) -> str:
    """High-level helper that performs the full RAG pipeline using Cloud SQL."""
    load_dotenv()
    client = genai.Client()
    embedder = VertexAIEmbeddings(client=client)
    engine = init_engine()

    query_embedding = embedder.embed_query(query)
    chunks = retrieve_chunks_db(engine, query_embedding, k=k)
    context = "\n\n".join(chunks)
    return generate_answer(query, context, client=client)


if __name__ == "__main__":
    USER_QUERY = "What should I pay attention to when planning to hot-plug devices to the VM?"
    answer = build_rag_and_answer(USER_QUERY)
    print("Question:", USER_QUERY)
    print("Answer:", answer)
