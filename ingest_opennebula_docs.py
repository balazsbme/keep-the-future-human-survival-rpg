import os
import logging
import time
from collections import deque
from typing import Iterable, List, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
from google.cloud.sql.connector import Connector
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sqlalchemy
from pgvector.sqlalchemy import Vector

from rag_pipeline import VertexAIEmbeddings


BASE_URL = "https://docs.opennebula.io/7.0/"
HEADERS = {"User-Agent": "OpenNebulaDocScraper/1.0"}

def scrape_urls(
    urls: Iterable[str], *, session: requests.Session | None = None, retries: int = 3, delay: int = 5
) -> str:
    """Return concatenated text content from ``urls``.

    Only the main content of each page is extracted. Failures are skipped after
    several retry attempts with a delay between them.

    This function is intentionally defensive: websites structure their HTML
    differently and may not always expose a ``<div role="main">`` element.
    We therefore attempt several strategies in order, falling back to the
    entire ``<body>`` or the whole document text when needed. Script and style
    tags are stripped from the extracted content.
    """

    texts: List[str] = []
    session = session or requests
    headers = {"User-Agent": "OpenNebulaDocScraper/1.0"}
    for url in urls:
        logging.info("Fetching %s", url)
        for attempt in range(1, retries + 1):
            try:
                response = session.get(url, timeout=10, headers=headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # Try common containers for the main page content
                main_content = soup.find("main") or soup.find("div", role="main") or soup.body

                if main_content:
                    for tag in main_content.find_all(["script", "style", "noscript"]):
                        tag.decompose()
                    text = main_content.get_text(separator=" ", strip=True)
                else:
                    # Fallback to entire document text
                    text = soup.get_text(separator=" ", strip=True)

                if text:
                    texts.append(text)
                break
            except requests.RequestException as exc:
                if attempt == retries:
                    logging.warning("Failed to fetch %s after %d attempts: %s", url, retries, exc)
                else:
                    logging.warning(
                        "Attempt %d/%d failed for %s: %s; retrying in %d seconds",
                        attempt,
                        retries,
                        url,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                continue

    return "\n\n".join(texts)

def chunk_text(text: str, *, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[str]:
    """Split ``text`` into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

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
        if len(results) % 20 == 0:
            logging.debug(
                "Collected %d URLs so far; example: %s", len(results), url
            )
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


def store_chunks(
    engine: sqlalchemy.Engine,
    chunks: Iterable[str],
    embeddings: Iterable[List[float]],
    references: Iterable[str],
) -> None:
    """Persist ``chunks`` and their metadata into the ``documents`` table.

    Uses the ``pgvector`` SQLAlchemy extension to safely bind Python lists to the
    ``vector`` column type, avoiding manual string construction and mitigating
    SQL injection risks.
    """
    chunks_list = list(chunks)
    embeddings_list = list(embeddings)
    references_list = list(references)

    documents = sqlalchemy.Table(
        "documents",
        sqlalchemy.MetaData(),
        sqlalchemy.Column("content", sqlalchemy.Text, nullable=False),
        sqlalchemy.Column("embedding", Vector()),
        sqlalchemy.Column("reference", sqlalchemy.Text, nullable=False),
    )
    with engine.begin() as conn:
        conn.execute(
            documents.insert(),
            [
                {"content": c, "embedding": e, "reference": r}
                for c, e, r in zip(chunks_list, embeddings_list, references_list)
            ],
        )


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
            references = [url] * len(chunks)
            store_chunks(engine, chunks, vectors, references)
            logging.info("Stored %d chunks for %s", len(chunks), url)
        except Exception as exc:
            logging.error("Database insert failed for %s: %s", url, exc)


if __name__ == "__main__":
    ingest()
