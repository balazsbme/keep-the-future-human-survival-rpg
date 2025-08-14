Here is a README file for your repository.

-----

# Mini RAG with Gemini 📚

This project is a minimal, self-contained implementation of a **Retrieval-Augmented Generation (RAG)** pipeline using Python. It scrapes text from specified web pages, builds a local knowledge base, and uses the Google Gemini API to answer questions based on the scraped content.

It's designed to be a straightforward example of how to build a powerful Q\&A bot over your own documentation without needing complex cloud infrastructure.

-----

### \#\# How It Works

The script performs the following steps:

1.  **Scrape Data:** It fetches the raw HTML content from a list of URLs you provide.
2.  **Chunk Text:** The extracted text is broken down into smaller, manageable chunks.
3.  **Embed Chunks:** Each text chunk is converted into a numerical vector (embedding) using the Gemini embedding model.
4.  **Store Vectors:** These vectors are stored in a local, in-memory `FAISS` index, which acts as our searchable knowledge base.
5.  **Retrieve & Generate:** When you ask a question, the script searches the `FAISS` index for the most relevant text chunks and passes them—along with your question—to the Gemini Pro model to generate a context-aware answer.

-----

### \#\# ⚙️ Setup and Installation

Follow these steps to get the project running.

1.  **Prerequisites:**

      * Python 3.8 or newer
      * A Google Gemini API Key. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

2.  **Clone the Repository:**

    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *(Note: If you don't have a `requirements.txt` file, create one with the contents from the `pip install` command in the script, or just run the command directly).*

4.  **Set Up Your API Key:**

      * Create a file named `.env` in the root directory of the project.
      * Add your Gemini API key to this file:
        ```
        GOOGLE_API_KEY="YOUR_API_KEY_HERE"
        ```

-----

### \#\# 🚀 Usage

Running the Q\&A pipeline is simple.

1.  **Customize the Script (Optional):**
    Open the `mini_rag.py` file and modify the following variables at the bottom to point to your desired documentation and question:

    ```python
    # Define the documentation URLs and the question to ask
    DOC_URLS = [
        "https://your-doc-url-1.com",
        "https://your-doc-url-2.com"
    ]
    USER_QUERY = "What is the main topic of this documentation?"
    ```

2.  **Run the Script:**
    Execute the script from your terminal:

    ```bash
    python mini_rag.py
    ```

The script will then perform the entire RAG process and print the final question and its generated answer to the console.

-----

### \#\# Technology Stack

  * **Language:** Python
  * **LLM & Embeddings:** Google Gemini API (`gemini-1.5-pro-latest`, `text-embedding-004`)
  * **Core Libraries:**
      * `langchain`: For text splitting and vector store integration.
      * `faiss-cpu`: For efficient, local similarity search.
      * `beautifulsoup4`: For parsing HTML.
      * `requests`: For making HTTP requests.

-----

### \#\# License

This project is licensed under the MIT License. See the `LICENSE` file for details.