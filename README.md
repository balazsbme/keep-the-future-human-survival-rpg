Here is a README file for your repository.

-----

# Mini RAG with Vertex AI 📚

This project is a minimal, self-contained implementation of a **Retrieval-Augmented Generation (RAG)** pipeline using Python. It queries embeddings stored in a Cloud SQL database (populated by a separate ingestion script) and uses Google Vertex AI's Gemini models to answer questions based on those documents.

It's designed to be a straightforward example of how to build a powerful Q\&A bot over your own documentation without needing complex cloud infrastructure.

-----

### \#\# How It Works

The runtime pipeline performs the following steps:

1.  **Embed Query:** Your question is embedded into a vector using the Vertex AI text embedding model.
2.  **Search Cloud SQL:** The query embedding is compared against vectors stored in the `documents` table of Cloud SQL using `pgvector` to retrieve the most relevant text chunks.
3.  **Generate:** The retrieved chunks and your question are sent to the Gemini model on Vertex AI to produce a context-aware answer.

For larger scale ingestion into Google Cloud SQL, see `ingest_opennebula_docs.py`, which crawls the entire OpenNebula 7.0 documentation, embeds the text with Vertex AI, and stores the results in a Cloud SQL table.

-----

### \#\# ⚙️ Setup and Installation

Follow these steps to get the project running.

1.  **Prerequisites:**

      * Python 3.8 or newer
      * A Google Cloud project with Vertex AI enabled and application default credentials configured.

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

4.  **Configure Project Info:**

        * Create a file named `.env` in the root directory of the project.
        * Add your Vertex AI project configuration:
          ```
          GCP_PROJECT_ID="your-project-id"
          GCP_REGION="us-central1"
          ```

-----

### \#\# 🚀 Usage

Running the Q\&A pipeline is simple.

1.  **Customize the Script (Optional):**
    Open the `rag_pipeline.py` file and modify the `USER_QUERY` variable at the bottom to ask your own question:

    ```python
    USER_QUERY = "What is the main topic of this documentation?"
    ```

2.  **Run the Script:**
    Execute the script from your terminal:

    ```bash
    python rag_pipeline.py
    ```

    The script will then perform the entire RAG process and print the final question and its generated answer to the console.

3. **Run the tests (optional):**
    ```bash
    pytest
    ```
    These tests rely on a dummy embedding model and require no cloud access.

4. **Manual verification with your project:**
    * Ensure a `.env` file with `GCP_PROJECT_ID`, `GCP_REGION`, and Cloud SQL credentials exists.
    * Adjust `USER_QUERY` if desired.
    * Run `python rag_pipeline.py` and review the generated answer.

5. **Ingest OpenNebula docs into Cloud SQL (optional):**
    * Provision a Cloud SQL PostgreSQL instance and note the connection name, database, user, and password.
    * Add these values to your `.env`:
      ```
      CLOUD_SQL_CONNECTION_NAME="project:region:instance"
      DB_USER="username"
      DB_PASS="password"
      DB_NAME="database"
      ```
    * Run the ingestion script:
      ```bash
      python ingest_opennebula_docs.py
      ```
        This script crawls all pages under `https://docs.opennebula.io/7.0/`, splits the content into chunks, embeds them with Vertex AI, and stores the embeddings in the `documents` table inside your Cloud SQL database.

-----

### \#\# Technology Stack

  * **Language:** Python
  * **LLM & Embeddings:** Vertex AI Gemini (`gemini-pro`, `text-embedding-004`)
    * **Core Libraries:**
        * `langchain`: For text splitting.
        * `pgvector` + `sqlalchemy`: For similarity search in Cloud SQL.
        * `beautifulsoup4`: For parsing HTML.
        * `requests`: For making HTTP requests.

-----

### \#\# License

This project is licensed under the MIT License. See the `LICENSE` file for details.