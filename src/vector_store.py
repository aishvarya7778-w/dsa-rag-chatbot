"""
Vector Store module for the DSA RAG Chatbot.
Manages Pinecone client initialization, index creation, index statistics,
and vector upserts with deterministic chunk IDs to prevent duplicate entries.
"""

import time
from typing import List, Optional, Dict, Any, Callable
from langchain_core.documents import Document
import config

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    Pinecone = None
    ServerlessSpec = None

try:
    from langchain_pinecone import PineconeVectorStore
except ImportError:
    PineconeVectorStore = None


def get_pinecone_client(api_key: Optional[str] = None) -> "Pinecone":
    """
    Initialize and return a Pinecone client instance.
    """
    if Pinecone is None:
        raise ImportError("pinecone is not installed. Run `pip install pinecone`.")

    key = config.PINECONE_API_KEY if api_key is None else api_key
    if not key:
        raise ValueError("PINECONE_API_KEY is not set. Please update your .env file.")

    return Pinecone(api_key=key)


def get_index_stats(
    index_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch status and statistics for the specified Pinecone index.
    
    Returns:
        dict: Contains 'exists', 'dimension', 'metric', 'total_vector_count', 'ready'.
    """
    name = index_name or config.PINECONE_INDEX_NAME
    client = get_pinecone_client(api_key=api_key)
    
    existing_indexes = [idx.name for idx in client.list_indexes()]
    if name not in existing_indexes:
        return {
            "exists": False,
            "index_name": name,
            "total_vector_count": 0,
            "ready": False,
            "error": f"Index '{name}' does not exist in your Pinecone account.",
        }

    index_description = client.describe_index(name)
    index_handle = client.Index(name)
    stats = index_handle.describe_index_stats()

    return {
        "exists": True,
        "index_name": name,
        "dimension": index_description.dimension,
        "metric": index_description.metric,
        "ready": index_description.status.get("ready", True) if isinstance(index_description.status, dict) else getattr(index_description.status, "ready", True),
        "total_vector_count": stats.total_vector_count,
        "namespaces": stats.namespaces,
    }


def create_pinecone_index_if_missing(
    index_name: Optional[str] = None,
    dimension: int = config.EMBEDDING_DIMENSION,
    metric: str = config.PINECONE_METRIC,
    cloud: str = "aws",
    region: str = "us-east-1",
    api_key: Optional[str] = None,
) -> bool:
    """
    Create a serverless Pinecone index if it doesn't already exist.
    Matches embedding dimension (768 for gemini-embedding-001) and cosine metric.
    
    Returns:
        bool: True if created, False if it already existed.
    """
    name = index_name or config.PINECONE_INDEX_NAME
    client = get_pinecone_client(api_key=api_key)
    existing_indexes = [idx.name for idx in client.list_indexes()]

    if name in existing_indexes:
        return False

    client.create_index(
        name=name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )

    # Wait for the serverless index to be provisioned and ready (typically 2-5 seconds)
    import time
    for _ in range(30):
        desc = client.describe_index(name)
        status = desc.status if isinstance(desc.status, dict) else getattr(desc, "status", {})
        ready = status.get("ready", False) if isinstance(status, dict) else getattr(status, "ready", False)
        if ready:
            break
        time.sleep(1)

    return True


def get_vector_store(
    embedding_model,
    index_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> "PineconeVectorStore":
    """
    Instantiate and return the LangChain PineconeVectorStore wrapper.
    """
    if PineconeVectorStore is None:
        raise ImportError(
            "langchain-pinecone is not installed. Run `pip install langchain-pinecone`."
        )

    name = index_name or config.PINECONE_INDEX_NAME
    key = api_key or config.PINECONE_API_KEY
    return PineconeVectorStore(
        index_name=name,
        embedding=embedding_model,
        pinecone_api_key=key if key else None,
    )


def upsert_documents(
    documents: List[Document],
    vector_store: "PineconeVectorStore",
    batch_size: int = config.EMBEDDING_BATCH_SIZE,
    inter_batch_delay: float = config.INTER_BATCH_DELAY,
    on_batch_complete: Optional[Callable[[List[str], int, int], None]] = None,
) -> int:
    """
    Upsert documents in controlled batches with deterministic IDs to avoid duplicate records.
    Applies inter-batch delays to prevent API rate limit bursts and notifies via callback
    for progress saving.
    
    Args:
        documents: List of split Document objects with chunk_id metadata.
        vector_store: PineconeVectorStore instance.
        batch_size: Number of chunks per upsert batch. Defaults to config.EMBEDDING_BATCH_SIZE (20).
        inter_batch_delay: Seconds to pause between batches. Defaults to config.INTER_BATCH_DELAY (1.5s).
        on_batch_complete: Optional callback invoked as `on_batch_complete(batch_ids, current_count, total_count)`
            after each successfully upserted batch.
        
    Returns:
        int: Number of document chunks upserted.
    """
    if not documents:
        return 0

    total = len(documents)
    total_batches = (total + batch_size - 1) // batch_size
    upserted_count = 0

    for batch_idx, i in enumerate(range(0, total, batch_size), 1):
        batch = documents[i : i + batch_size]
        ids = [doc.metadata.get("chunk_id") for doc in batch]
        
        # If any doc lacked a chunk_id, LangChain will generate UUIDs, but our text_splitter guarantees it
        if None in ids:
            ids = None
            
        print(f"   [Batch {batch_idx}/{total_batches}] Upserting chunks {i + 1}–{min(i + batch_size, total)} of {total}...")
        vector_store.add_documents(documents=batch, ids=ids)
        upserted_count += len(batch)

        if on_batch_complete and ids:
            on_batch_complete(ids, upserted_count, total)

        # Apply polite inter-batch delay to stay under per-minute rate limits
        if i + batch_size < total and inter_batch_delay > 0:
            time.sleep(inter_batch_delay)

    return upserted_count
