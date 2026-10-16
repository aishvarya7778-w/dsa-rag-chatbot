"""
Retriever module for the DSA RAG Chatbot.
Configures vector store retrieval and extracts source citations with page numbers.
"""

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
import config


def get_dsa_retriever(
    vector_store,
    top_k: int = config.TOP_K_RESULTS,
    search_type: str = "similarity",
):
    """
    Build a retriever from the PineconeVectorStore.
    
    Args:
        vector_store: PineconeVectorStore instance.
        top_k: Number of relevant chunks to retrieve.
        search_type: 'similarity' or 'mmr' (Maximal Marginal Relevance).
        
    Returns:
        VectorStoreRetriever instance.
    """
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": top_k},
    )


def format_docs_for_prompt(docs: List[Document]) -> str:
    """
    Format a list of retrieved documents into a clean string context block for the LLM.
    Includes source filename and page numbers in the context header for grounded reasoning.
    """
    if not docs:
        return "No relevant context found in the knowledge base."

    formatted_blocks = []
    for idx, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", "Unknown Page")
        content = doc.page_content.strip()
        header = f"--- Document [{idx}]: {source} (Page {page}) ---"
        formatted_blocks.append(f"{header}\n{content}")

    return "\n\n".join(formatted_blocks)


def extract_source_citations(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    Extract structured source information for UI citation display.
    De-duplicates repeated source/page references.
    
    Returns:
        List of dicts: [{'source': str, 'page': int, 'preview': str}]
    """
    seen_references = set()
    citations = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page", 1)
        ref_key = (source, page)

        if ref_key not in seen_references:
            seen_references.add(ref_key)
            # Create a 200-char preview snippet
            snippet = doc.page_content.strip()
            preview = snippet[:220] + "..." if len(snippet) > 220 else snippet
            citations.append({
                "source": source,
                "page": page,
                "preview": preview,
                "chunk_id": doc.metadata.get("chunk_id", ""),
            })

    return citations
