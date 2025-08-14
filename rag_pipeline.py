"""Minimal RAG pipeline for OpenNebula documentation using Gemini."""
from __future__ import annotations

import os
from typing import Iterable, List

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai


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
    embedding_model: model used to embed text. If ``None`` a Gemini embedding
        model is instantiated; tests can supply a mock embedding model.
    """
    if embedding_model is None:
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return FAISS.from_texts(chunks, embedding=embedding_model)


def retrieve_chunks(store: FAISS, query: str, k: int = 3) -> List[str]:
    """Return the top ``k`` document texts similar to ``query``."""
    docs = store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


def generate_answer(question: str, context: str, *, model: str = "gemini-1.5-pro-latest") -> str:
    """Generate an answer to ``question`` given ``context`` using Gemini."""
    prompt = f"""Answer the question based ONLY on the provided context.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}"""
    llm = genai.GenerativeModel(model)
    response = llm.generate_content(prompt)
    return response.text


def build_rag_and_answer(urls: Iterable[str], query: str) -> str:
    """High-level helper that performs the full RAG pipeline."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment")
    genai.configure(api_key=api_key)

    raw_text = scrape_urls(urls)
    chunks = chunk_text(raw_text)
    store = build_vector_store(chunks)
    context = "\n\n".join(retrieve_chunks(store, query))
    return generate_answer(query, context)


if __name__ == "__main__":
    DOC_URLS = [
        "https://docs.opennebula.io/7.0/intro_and_overview/what_is_opennebula.html",
        "https://docs.opennebula.io/7.0/management_and_operations/vm_management/vm_guide.html",
    ]
    USER_QUERY = "What are the main states in a VM lifecycle?"
    answer = build_rag_and_answer(DOC_URLS, USER_QUERY)
    print("Question:", USER_QUERY)
    print("Answer:", answer)
