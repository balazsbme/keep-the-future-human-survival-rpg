import os
import google.generativeai as genai
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

def build_and_query_rag(urls, query):
    """Scrapes URLs, builds a RAG pipeline, and answers a query."""
    # 1. SETUP API KEY
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in .env file.")

    # 2. SCRAPE & CHUNK TEXT
    print("Scraping and chunking documents...")
    all_text = ""
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            main_content = soup.find('div', role='main')
            if main_content:
                all_text += main_content.get_text(separator=' ', strip=True) + "\n\n"
        except requests.RequestException as e:
            print(f"Skipping {url}: {e}")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(all_text)
    if not chunks:
        raise ValueError("No text extracted from URLs. Check URLs or HTML structure.")

    # 3. EMBED & CREATE VECTOR STORE
    print("Creating vector store with Gemini embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    # 4. RETRIEVE & GENERATE
    print("Retrieving relevant context...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    context_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in context_docs])

    prompt = f"""
    Answer the following question based ONLY on the provided context.

    CONTEXT:
    {context}

    QUESTION: {query}
    """
    
    print("Generating answer...")
    llm = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = llm.generate_content(prompt)
    
    print("\n--- RESULT ---")
    print(f"Question: {query}")
    print(f"Answer: {response.text}")

# --- Main execution ---
if __name__ == "__main__":
    # Define the documentation URLs and the question to ask
    DOC_URLS = [
        "https://docs.opennebula.io/7.0/intro_and_overview/what_is_opennebula.html",
        "https://docs.opennebula.io/7.0/management_and_operations/vm_management/vm_guide.html"
    ]
    USER_QUERY = "What are the main states in a VM lifecycle?"

    build_and_query_rag(DOC_URLS, USER_QUERY)