"""Minimal RAG pipeline for OpenNebula documentation using Vertex AI."""
from __future__ import annotations

import os
from typing import Iterable, List

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel
from vertexai.preview.generative_models import GenerativeModel


def initialize_vertex_ai(project: str, location: str) -> None:
    """Initialize the Vertex AI SDK."""
    vertexai.init(project=project, location=location)


class VertexAIEmbeddings(Embeddings):
    """Wrapper around Vertex AI text embedding model for LangChain."""

    def __init__(self, model_name: str = "text-embedding-005") -> None:
        self._model = TextEmbeddingModel.from_pretrained(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.get_embeddings(texts)
        return [emb.values for emb in embeddings]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# ----- Core pipeline steps -----

def scrape_urls(urls: Iterable[str]) -> str:
    """Return concatenated text content from ``urls``.

    Only the main content of each page is extracted. Failures are skipped.
    """
    texts: List[str] = []
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            main_content = soup.find("div", role="main")
            if main_content:
                texts.append(main_content.get_text(separator=" ", strip=True))
        except requests.RequestException:
            continue
    return "\n\n".join(texts)


def chunk_text(text: str, *, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[str]:
    """Split ``text`` into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def build_vector_store(chunks: List[str], embedding_model=None) -> FAISS:
    """Create a FAISS vector store from text ``chunks``.

    Parameters
    ----------
    chunks: list of text segments to index.
    embedding_model: model used to embed text. If ``None`` a Vertex AI embedding
        model is instantiated; tests can supply a mock embedding model.
    """
    if embedding_model is None:
        embedding_model = VertexAIEmbeddings()
    return FAISS.from_texts(chunks, embedding=embedding_model)


def retrieve_chunks(store: FAISS, query: str, k: int = 3) -> List[str]:
    """Return the top ``k`` document texts similar to ``query``."""
    docs = store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


def generate_answer(question: str, context: str, *, model: str = "gemini-pro") -> str:
    """Generate an answer to ``question`` given ``context`` using Vertex AI."""
    prompt = (
        "Answer the question based ONLY on the provided context.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    )
    llm = GenerativeModel(model)
    response = llm.generate_content(prompt)
    return response.text


def build_rag_and_answer(urls: Iterable[str], query: str) -> str:
    """High-level helper that performs the full RAG pipeline."""
    load_dotenv()
    project = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_REGION")
    if not project or not location:
        raise ValueError("GCP_PROJECT_ID and GCP_REGION must be set in environment")
    initialize_vertex_ai(project, location)

    raw_text = scrape_urls(urls)
    chunks = chunk_text(raw_text)
    store = build_vector_store(chunks)
    context = "\n\n".join(retrieve_chunks(store, query))
    return generate_answer(query, context)


if __name__ == "__main__":
    DOC_URLS = [
        "https://docs.opennebula.io/7.0/quick_start/understand_opennebula/opennebula_concepts/opennebula_overview/",
        "https://docs.opennebula.io/7.0/product/virtual_machines_operation/virtual_machine_definitions/vm_instances/",
    ]
    USER_QUERY = "What are the main states in a VM lifecycle?"
    answer = build_rag_and_answer(DOC_URLS, USER_QUERY)
    print("Question:", USER_QUERY)
    print("Answer:", answer)
