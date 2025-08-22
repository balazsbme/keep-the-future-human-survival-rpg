import os
import logging
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
HEADERS = {"User-Agent": "OpenNebulaDocScraper/1.0"}


def crawl_site(base_url: str = BASE_URL) -> List[str]:
    """Return all links within ``base_url``.

    A breadth-first crawl restricted to the ``base_url`` domain. Fragments and
    query parameters are stripped to avoid duplicates.
    """
    logging.info("Starting crawl for %s", base_url)
    visited: Set[str] = set()
    to_visit = deque([base_url])
    results: List[str] = []

    while to_visit:
        url = to_visit.popleft()
        logging.debug("Crawling %s", url)
        norm = _normalize(url)
        if norm in visited:
            continue
        visited.add(norm)
        try:
            try:
                resp = requests.get(url, timeout=10, headers=HEADERS)
            except TypeError:
                resp = requests.get(url, timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logging.warning("Failed to crawl %s: %s", url, exc)
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
    logging.info("Completed crawl for %s with %d urls", base_url, len(results))
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


def store_chunks(engine: sqlalchemy.Engine, chunks: Iterable[str], embeddings: Iterable[List[float]]) -> None:
    """Persist ``chunks`` and their ``embeddings`` into the ``documents`` table."""
    insert_stmt = sqlalchemy.text(
        """
        INSERT INTO documents (content, embedding)
        VALUES (:content, :embedding::vector)
        """
    )
    with engine.begin() as conn:
        for chunk, emb in zip(chunks, embeddings):
            vector = "[" + ",".join(str(v) for v in emb) + "]"
            conn.execute(insert_stmt, {"content": chunk, "embedding": vector})


def ingest() -> None:
    """Scrape OpenNebula docs and store embeddings in Cloud SQL."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    urls = crawl_site(BASE_URL)
    client = genai.Client()
    embedder = VertexAIEmbeddings(client=client)
    engine = init_engine()
    for url in urls:
        logging.info("Fetching %s", url)
        text = scrape_urls([url])
        if not text:
            logging.warning("No content extracted from %s", url)
            continue
        chunks = chunk_text(text)
        logging.info("Embedding %d chunks from %s", len(chunks), url)
        try:
            vectors = embedder.embed_documents(chunks)
        except Exception as exc:
            logging.error("Embedding failed for %s: %s", url, exc)
            continue
        try:
            store_chunks(engine, chunks, vectors)
            logging.info("Stored %d chunks for %s", len(chunks), url)
        except Exception as exc:
            logging.error("Database insert failed for %s: %s", url, exc)


if __name__ == "__main__":
    ingest()
