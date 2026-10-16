"""
Configuration module for the DSA RAG Chatbot.
Centralizes environment variables, directory paths, and model hyperparameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"

# Ensure data directories exist
PDF_DIR.mkdir(parents=True, exist_ok=True)

# API Keys and External Credentials
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "dsa-rag-index").strip()

# Embeddings Configuration
# Google Generative AI gemini-embedding-001 produces 768-dimensional embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001").strip()
EMBEDDING_DIMENSION = 768
PINECONE_METRIC = "cosine"
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "20"))
INTER_BATCH_DELAY = float(os.getenv("INTER_BATCH_DELAY", "1.5"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
INITIAL_RETRY_DELAY = float(os.getenv("INITIAL_RETRY_DELAY", "5.0"))
RETRY_BACKOFF_FACTOR = float(os.getenv("RETRY_BACKOFF_FACTOR", "2.0"))
CHECKPOINT_FILE = DATA_DIR / "ingestion_checkpoint.json"

# LLM Generation Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash").strip()
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# Document Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Retrieval Configuration
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))


def get_missing_credentials() -> list[str]:
    """
    Check if required environment variables are set.
    Returns a list of missing configuration names.
    """
    missing = []
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")
    if not PINECONE_INDEX_NAME:
        missing.append("PINECONE_INDEX_NAME")
    return missing


def is_configured() -> bool:
    """Return True if all essential API credentials are set, False otherwise."""
    return len(get_missing_credentials()) == 0
