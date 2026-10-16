# 🧠 DSA RAG Chatbot

An AI-powered Data Structures and Algorithms tutor grounded strictly in your curated DSA PDF documents. Built with **Streamlit**, **LangChain**, **Google Gemini**, **Pinecone**, and **PyPDF**.

---

## 🌟 Features

- **Strict Knowledge Grounding:** Answers are generated from retrieved PDF chunks. If the concepts are missing from your uploaded PDFs, the tutor explicitly notifies you rather than making up answers.
- **Step-by-step Algorithm Explanations:** Explains intuition, algorithm steps, edge cases, and provides clean Python code implementations.
- **Big-O Complexity Analysis:** Breaks down time complexity (best, average, worst) and space complexity for every algorithm.
- **Transparent Source Citations:** Cites the exact PDF file name and page number for every piece of retrieved context.
- **Deduplicated Vector Ingestion:** Deterministic chunk hashing prevents duplicate vector records in Pinecone when re-running ingestion.
- **Modern Streamlit UI:** Features chat message history, suggested question starters, system readiness indicators, and an empty-state guide.

---

## 📁 Project Architecture

```
DSA chatbot/
├── app.py                  # Streamlit web application & user interface
├── config.py               # Central configuration, paths, & hyperparameter settings
├── requirements.txt        # Python package dependencies
├── .env.example            # Template for required environment variables
├── .gitignore              # Protects secrets, API keys, and temporary files
├── README.md               # Documentation and execution guide
├── src/
│   ├── __init__.py         # Package initialization
│   ├── pdf_loader.py       # Extracts text and page metadata using PyPDF
│   ├── text_splitter.py    # Recursive chunking & deterministic chunk ID generation
│   ├── embeddings.py       # Google gemini-embedding-001 wrapper (768 dimensions)
│   ├── vector_store.py     # Pinecone client, index creation, stats, and upserts
│   ├── retriever.py        # Similarity retriever & source citation extraction
│   ├── rag_chain.py        # LangChain LCEL pipeline with Gemini and grounded prompt
│   └── ingest.py           # CLI script to load, chunk, and index PDFs into Pinecone
├── data/
│   └── pdfs/               # Directory where your curated DSA PDF files are stored
└── tests/
    └── test_components.py  # Unit tests for core processing logic
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10 to 3.13
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/app/apikey))
- A Pinecone account and API key ([Pinecone Console](https://app.pinecone.io/))

### 2. Environment Setup

Clone or open the project folder in your terminal:
```bash
cd "C:\Projects\DSA chatbot"
```

Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configure Credentials

Duplicate `.env.example` to create your `.env` file:
```bash
cp .env.example .env
```
*(On Windows Command Prompt: `copy .env.example .env`)*

Open `.env` and fill in your keys:
```env
GEMINI_API_KEY=AIzaSy...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=dsa-rag-index
```

### 4. Add your DSA PDFs

Place your curated PDF documents into the `data/pdfs/` folder:
```
data/pdfs/
├── Arrays_and_Strings.pdf
├── Trees_and_Graphs.pdf
└── Dynamic_Programming.pdf
```

### 5. Ingest Documents into Pinecone

Run the automated ingestion script:
```bash
python -m src.ingest
```

> **Options:**
> - Dry run (test extraction and chunking locally without calling APIs):
>   ```bash
>   python -m src.ingest --dry-run
>   ```
> - Resume ingestion safely (automatically skips already indexed chunks):
>   ```bash
>   python -m src.ingest
>   ```
> - Controlled batch size & delay (e.g. 10 chunks per batch, 2.0s delay):
>   ```bash
>   python -m src.ingest --batch-size 10 --batch-delay 2.0
>   ```
> - Force full re-indexing from scratch:
>   ```bash
>   python -m src.ingest --no-resume
>   ```
> - Verify index retrieval after ingestion:
>   ```bash
>   python -m src.ingest --verify "How does binary search work?"
>   ```

### 6. Launch the Streamlit App

Start the interactive chat interface:
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## ⚙️ Model Specifications

| Component | Model / Technology | Dimensions / Spec |
| :--- | :--- | :--- |
| **Embeddings** | `gemini-embedding-001` | 768 dimensions |
| **Vector DB** | Pinecone Serverless | Metric: `cosine`, 768 dims |
| **LLM Generator** | `gemini-3.5-flash` | Temperature: 0.2 |
| **PDF Extraction** | `pypdf` | Page-aware extraction |
| **Chunking** | Recursive Character Splitter | Size: 1000, Overlap: 200 |

---

## 🧪 Running Tests

Run the offline unit tests:
```bash
python -m unittest discover tests
```
