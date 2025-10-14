# Ai Legal Assistant

An AI-powered **legal assistant** built with **LlamaIndex + Local LLMs**, leveraging **RAG (Retrieval-Augmented Generation)** to retrieve and answer questions based on local legal documents.

## 📌 Features
- Uses **ChromaDB** as the vector database.
- The web page automatically crawls the relevant legal provisions and saves them locally
- Loads **JSON-formatted legal documents** from local storage.
- Builds and persists vector indexes for efficient retrieval.
- Includes a **ChatML-style prompt template** to ensure answers strictly follow legal provisions.
- Provides an interactive **Streamlit web interface** for Q&A and citation display.

## 📂 Project Structure
Ai_legal_assistant/
│── app.py # Main Streamlit application entry point
│── data/ # Folder for legal JSON files
│── chroma_db/ # Chroma vector database storage
│── storage/ # Index persistence directory
│── requirements.txt # Python dependencies
│── README.md # Project documentation

🚀 Usage
1. Install Dependencies

It is recommended to use Python 3.9+. Create a virtual environment and install dependencies:

pip install -U streamlit chromadb llama-index-core llama-index-llms-huggingface llama-index-embeddings-huggingface

2. Run the Application
streamlit run app.py


Then open http://localhost:8501
 in your browser.

3. How to Use

Configure Embedding model path, LLM model path, data directory, and Top-K retrieval in the sidebar.

Use Force Rebuild Index to clear and rebuild the vector database.

Clear chat history at any time.

Each answer includes retrieved legal citations for transparency and validation.

⚠️ Notes

Do not upload large model weights to GitHub; add them to .gitignore.

Index building may take time if the dataset is large.

On limited GPU memory, consider smaller models (e.g., Qwen-1.8B, bge-small).
