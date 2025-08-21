import os
import json
from collections import deque
from typing import Iterable, List, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
from google.cloud.sql.connector import Connector
import sqlalchemy

from rag_pipeline import scrape_urls, chunk_text, VertexAIEmbeddings


BASE_URL = "https://docs.opennebula.io/7.0/"


def crawl_site(base_url: str = BASE_URL) -> List[str]:
    """Return all links within ``base_url``.

    A breadth-first crawl restricted to the ``base_url`` domain. Fragments and
    query parameters are stripped to avoid duplicates.
    """
    visited: Set[str] = set()
    to_visit = deque([base_url])
    results: List[str] = []

    while to_visit:
        url = to_visit.popleft()
        norm = _normalize(url)
        if norm in visited:
            continue
        visited.add(norm)
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except requests.RequestException:
            continue
        results.append(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("#"):
                continue
            joined = urljoin(url, href)
            if not joined.startswith(base_url):
                continue
            candidate_norm = _normalize(joined)
            if candidate_norm in visited:
                continue
            to_visit.append(joined)
    return results


def _normalize(url: str) -> str:
    """Normalize URLs to avoid duplicates."""
    parsed = urlparse(url)
    clean = parsed._replace(fragment="", query="").geturl()
    return clean.rstrip("/")


def init_engine() -> sqlalchemy.Engine:
    """Create a SQLAlchemy engine for Cloud SQL."""
    connection_name = os.environ["CLOUD_SQL_CONNECTION_NAME"]
    db_user = os.environ["DB_USER"]
    db_pass = os.environ["DB_PASS"]
    db_name = os.environ["DB_NAME"]

    connector = Connector()

    def getconn() -> any:
        return connector.connect(connection_name, "pg8000", user=db_user, password=db_pass, db=db_name)

    return sqlalchemy.create_engine("postgresql+pg8000://", creator=getconn)


def store_chunks(engine: sqlalchemy.Engine, url: str, chunks: Iterable[str], embeddings: Iterable[List[float]]) -> None:
    """Persist ``chunks`` and their ``embeddings`` for ``url``."""
    insert_stmt = sqlalchemy.text(
        """
        INSERT INTO embeddings (url, chunk_index, content, embedding)
        VALUES (:url, :chunk_index, :content, :embedding)
        """
    )
    with engine.begin() as conn:
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            conn.execute(
                insert_stmt,
                {
                    "url": url,
                    "chunk_index": idx,
                    "content": chunk,
                    "embedding": json.dumps(emb),
                },
            )


def ingest() -> None:
    """Scrape OpenNebula docs and store embeddings in Cloud SQL."""
    load_dotenv()
    urls = crawl_site(BASE_URL)
    client = genai.Client()
    embedder = VertexAIEmbeddings(client=client)
    engine = init_engine()
    for url in urls:
        text = scrape_urls([url])
        if not text:
            continue
        chunks = chunk_text(text)
        vectors = embedder.embed_documents(chunks)
        store_chunks(engine, url, chunks, vectors)


if __name__ == "__main__":
    ingest()
