"""
Text Splitter module for the DSA RAG Chatbot.
Splits PDF documents into semantically coherent chunks while preserving metadata.
"""

import hashlib
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


def get_text_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> RecursiveCharacterTextSplitter:
    """
    Instantiate a RecursiveCharacterTextSplitter tuned for DSA code and theory.
    
    Args:
        chunk_size: Character size of each chunk.
        chunk_overlap: Overlapping characters between adjacent chunks.
        
    Returns:
        Configured RecursiveCharacterTextSplitter instance.
    """
    size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP
    
    # Custom separators prioritize paragraphs, code blocks, and function definitions
    separators = [
        "\n\n",
        "\n",
        "```",
        "def ",
        "class ",
        " ",
        "",
    ]
    
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=separators,
        keep_separator=True,
    )


def generate_chunk_id(doc: Document, chunk_idx: int) -> str:
    """
    Generate a deterministic, reproducible ID for a document chunk.
    Format: <source_filename>_p<page_number>_c<chunk_index>_<content_hash_short>
    """
    source = doc.metadata.get("source", "doc").replace(" ", "_")
    page = doc.metadata.get("page", 1)
    content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()[:8]
    return f"{source}_p{page}_c{chunk_idx}_{content_hash}"


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Split a list of LangChain documents into chunks, enriching each with unique chunk IDs.
    
    Args:
        documents: Raw documents loaded from PDFs.
        chunk_size: Size of chunk in characters.
        chunk_overlap: Overlap in characters.
        
    Returns:
        List of chunked Document instances with metadata.
    """
    if not documents:
        return []
        
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(documents)
    
    # Assign deterministic chunk_id to each chunk
    for idx, doc in enumerate(split_docs):
        doc.metadata["chunk_id"] = generate_chunk_id(doc, idx)
        
    return split_docs
