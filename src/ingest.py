"""
Document Ingestion script for the DSA RAG Chatbot.
Loads PDFs from data/pdfs/, splits into semantically rich chunks,
ensures Pinecone index exists with 768 dimensions, and upserts vectors with metadata.
Supports resumable ingestion, controlled batching, and graceful backoff for 429 quota limits.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Set, List

# Configure stdout and stderr to handle UTF-8 cleanly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import config
from src.pdf_loader import load_all_pdfs, list_available_pdfs
from src.text_splitter import split_documents
from src.embeddings import get_embedding_model, QuotaExhaustedError
from src.vector_store import (
    get_pinecone_client,
    get_index_stats,
    create_pinecone_index_if_missing,
    get_vector_store,
    upsert_documents,
)


def load_checkpoint(checkpoint_file: Path = config.CHECKPOINT_FILE) -> Dict[str, Any]:
    """
    Load ingestion progress checkpoint from disk.
    
    Returns:
        dict: Checkpoint state containing 'completed_ids' set, 'is_complete', and metadata.
    """
    if not checkpoint_file.exists():
        return {"completed_ids": set(), "is_complete": False}

    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {
                "completed_ids": set(data.get("completed_ids", [])),
                "is_complete": data.get("is_complete", False),
                "total_chunks": data.get("total_chunks", 0),
                "index_name": data.get("index_name", config.PINECONE_INDEX_NAME),
            }
    except Exception as exc:
        print(f"[!] Warning: Could not read checkpoint file ({exc}). Starting fresh tracking.")
        return {"completed_ids": set(), "is_complete": False}


def save_checkpoint(
    completed_ids: Set[str],
    total_chunks: int,
    is_complete: bool = False,
    checkpoint_file: Path = config.CHECKPOINT_FILE,
) -> None:
    """
    Persist ingestion progress checkpoint to disk atomically.
    Never stores API keys or confidential data.
    """
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "index_name": config.PINECONE_INDEX_NAME,
        "embedding_model": config.EMBEDDING_MODEL,
        "dimension": config.EMBEDDING_DIMENSION,
        "total_chunks": total_chunks,
        "completed_count": len(completed_ids),
        "completed_ids": sorted(list(completed_ids)),
        "is_complete": is_complete,
    }
    temp_file = checkpoint_file.with_suffix(".tmp")
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        temp_file.replace(checkpoint_file)
    except Exception:
        # Fallback to direct write if rename fails
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def clear_checkpoint(checkpoint_file: Path = config.CHECKPOINT_FILE) -> None:
    """Remove the checkpoint file to allow a fresh ingestion."""
    if checkpoint_file.exists():
        checkpoint_file.unlink()


def run_ingestion(
    dry_run: bool = False,
    force_recreate: bool = False,
    batch_size: int = config.EMBEDDING_BATCH_SIZE,
    inter_batch_delay: float = config.INTER_BATCH_DELAY,
    resume: bool = True,
) -> bool:
    """
    Execute end-to-end PDF ingestion into Pinecone with resumability and 429 protection.
    
    Args:
        dry_run: If True, only load and chunk PDFs locally without calling Pinecone or Gemini.
        force_recreate: If True, deletes existing index before recreating (and resets checkpoint).
        batch_size: Chunk upsert batch size. Defaults to config.EMBEDDING_BATCH_SIZE.
        inter_batch_delay: Delay between batches in seconds. Defaults to config.INTER_BATCH_DELAY.
        resume: If True, skips chunks that were already indexed in a previous run.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    print("=" * 65)
    print(" [DSA RAG] Knowledge Base Ingestion Pipeline")
    print("=" * 65)

    # Step 1: Check environment credentials
    if not dry_run:
        missing = config.get_missing_credentials()
        if missing:
            print(f"[!] Error: Missing configuration values in .env: {', '.join(missing)}")
            print("Please edit .env and provide your API keys before proceeding.")
            return False

    # Step 2: Check for PDF files
    available_pdfs = list_available_pdfs(config.PDF_DIR)
    if not available_pdfs:
        print(f"[!] No PDF documents found in: {config.PDF_DIR}")
        print("Please place your DSA PDF files inside `data/pdfs/` and run again.")
        return False

    print(f"[*] Found {len(available_pdfs)} PDF document(s) in `{config.PDF_DIR}`:")
    for p in available_pdfs:
        print(f"   • {p['name']} ({p['pages']} pages, {p['size_kb']} KB)")

    # Step 3: Load PDF documents
    print("\n[1/4] Extracting text and metadata from PDFs...")
    try:
        raw_documents = load_all_pdfs(config.PDF_DIR)
    except Exception as exc:
        print(f"[!] Error loading PDFs: {exc}")
        return False

    print(f"[+] Extracted {len(raw_documents)} non-empty page document(s).")

    # Step 4: Chunk documents
    print("\n[2/4] Splitting documents into semantically coherent chunks...")
    chunks = split_documents(
        documents=raw_documents,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    print(f"[+] Created {len(chunks)} text chunks (Size: {config.CHUNK_SIZE}, Overlap: {config.CHUNK_OVERLAP}).")

    if dry_run:
        print("\n[INFO] DRY RUN SUMMARY (no external API calls made):")
        print(f"   • Total PDF files: {len(available_pdfs)}")
        print(f"   • Total page documents: {len(raw_documents)}")
        print(f"   • Total chunks generated: {len(chunks)}")
        if chunks:
            sample = chunks[0]
            print(f"   • Sample Chunk ID: {sample.metadata.get('chunk_id')}")
            print(f"   • Sample Chunk Source: {sample.metadata.get('source')} (Page {sample.metadata.get('page')})")
            print(f"   • Sample Preview: {sample.page_content[:150]}...")
        print("\n[+] Dry run complete. Everything is verified locally!")
        return True

    # Step 5: Check or Create Pinecone Index (preserves existing index dsa-rag-index)
    print(f"\n[3/4] Connecting to Pinecone index '{config.PINECONE_INDEX_NAME}'...")
    try:
        client = get_pinecone_client()
        existing = [idx.name for idx in client.list_indexes()]

        if force_recreate and config.PINECONE_INDEX_NAME in existing:
            print(f"   Deleting existing index '{config.PINECONE_INDEX_NAME}' for fresh setup...")
            client.delete_index(config.PINECONE_INDEX_NAME)
            existing.remove(config.PINECONE_INDEX_NAME)
            clear_checkpoint()

        if config.PINECONE_INDEX_NAME not in existing:
            print(f"   Index '{config.PINECONE_INDEX_NAME}' not found. Creating serverless index...")
            print(f"   Configuration: Dimension={config.EMBEDDING_DIMENSION}, Metric={config.PINECONE_METRIC}")
            create_pinecone_index_if_missing(
                index_name=config.PINECONE_INDEX_NAME,
                dimension=config.EMBEDDING_DIMENSION,
                metric=config.PINECONE_METRIC,
            )
            print(f"[+] Created Pinecone index '{config.PINECONE_INDEX_NAME}'.")
        else:
            print(f"[+] Pinecone index '{config.PINECONE_INDEX_NAME}' exists.")

        stats_before = get_index_stats()
        print(f"   Current vectors in index: {stats_before.get('total_vector_count', 0)}")

    except Exception as exc:
        print(f"[!] Error connecting to Pinecone: {exc}")
        return False

    # Step 6: Determine chunks to ingest (resuming from checkpoint if available)
    completed_ids: Set[str] = set()
    if resume and not force_recreate:
        checkpoint_data = load_checkpoint()
        completed_ids = checkpoint_data.get("completed_ids", set())

        # If checkpoint claims vectors exist but index was emptied, reset checkpoint
        if completed_ids and stats_before.get("total_vector_count", 0) == 0:
            print("   Pinecone index is empty. Starting fresh ingestion despite checkpoint.")
            completed_ids = set()
            clear_checkpoint()

    # Filter out already indexed chunks
    pending_chunks = [c for c in chunks if c.metadata.get("chunk_id") not in completed_ids]

    if not pending_chunks:
        print(f"\n[+] All {len(chunks)} chunks are already indexed in Pinecone index '{config.PINECONE_INDEX_NAME}'!")
        print("   Knowledge base is fully up to date. To force re-indexing, run: `python -m src.ingest --no-resume`")
        print("   Start the UI with: `streamlit run app.py`")
        return True

    if completed_ids:
        print(f"\n[*] RESUMING INGESTION:")
        print(f"   • Already indexed: {len(completed_ids)}/{len(chunks)} chunks")
        print(f"   • Chunks remaining: {len(pending_chunks)} chunks")
    else:
        print(f"\n[*] Starting fresh ingestion: {len(pending_chunks)} chunks to process.")

    print(f"   Controlled batch size: {batch_size} chunks | Inter-batch delay: {inter_batch_delay}s")
    print(f"   Using embedding model: {config.EMBEDDING_MODEL} (Dimension: {config.EMBEDDING_DIMENSION})")

    # Step 7: Generate Embeddings and Upsert Vectors
    print("\n[4/4] Generating Gemini embeddings & upserting to Pinecone...")

    def on_batch_progress(batch_ids: List[str], current_upserted: int, total_upserted: int):
        completed_ids.update(batch_ids)
        save_checkpoint(completed_ids, total_chunks=len(chunks), is_complete=False)
        print(f"   ✓ Checkpoint updated: {len(completed_ids)}/{len(chunks)} total chunks indexed.")

    try:
        embedding_model = get_embedding_model()
        vector_store = get_vector_store(embedding_model=embedding_model)

        upserted_count = upsert_documents(
            documents=pending_chunks,
            vector_store=vector_store,
            batch_size=batch_size,
            inter_batch_delay=inter_batch_delay,
            on_batch_complete=on_batch_progress,
        )

        save_checkpoint(completed_ids, total_chunks=len(chunks), is_complete=True)
        print(f"\n[+] Successfully upserted {upserted_count} document chunks.")

        # Final verification
        stats_after = get_index_stats()
        print("\n[+] INGESTION COMPLETE - INDEX STATUS:")
        print(f"   • Index Name: {stats_after.get('index_name')}")
        print(f"   • Vector Dimension: {stats_after.get('dimension')}")
        print(f"   • Metric: {stats_after.get('metric')}")
        print(f"   • Total Vectors in Pinecone: {stats_after.get('total_vector_count')}")
        print("\n[+] Knowledge base is ready! Start the UI with: `streamlit run app.py`")
        return True

    except QuotaExhaustedError as quota_err:
        print("\n" + "=" * 68)
        print(" [!] INGESTION PAUSED: Google Gemini Free-Tier Quota Reached")
        print("=" * 68)
        print(" • Status: HTTP 429 RESOURCE_EXHAUSTED")
        print(f" • Reason: {quota_err}")
        print(f" • Progress Preserved: {len(completed_ids)}/{len(chunks)} chunks safely indexed in Pinecone.")
        print(f" • Checkpoint File: `{config.CHECKPOINT_FILE}`")
        print("\n IMPORTANT - What to do next:")
        print(" 1. Wait for Quota Refresh:")
        print("    - Per-minute rate limits automatically reset within ~30–60 seconds.")
        print("    - Free-tier daily request limits (100 requests) reset every 24 hours (midnight PT).")
        print(" 2. Enable Paid Billing (Optional):")
        print("    - You can enable Pay-as-you-go in Google AI Studio (https://ai.google.dev/pricing)")
        print("      to unlock higher throughput and eliminate free-tier quota limits.")
        print(" 3. Safe Resume:")
        print("    - Ingestion is fully resumable: deterministic chunk IDs prevent duplication,")
        print("      and already-indexed chunks will be skipped automatically.")
        print("    - To resume whenever you are ready, run:")
        print("        python -m src.ingest")
        print("=" * 68)
        return False

    except Exception as exc:
        print(f"[!] Error during embeddings generation or vector upsert: {exc}")
        print(f"   Progress saved: {len(completed_ids)}/{len(chunks)} chunks recorded in checkpoint.")
        return False


def verify_retrieval(query: str = "What is an array and its time complexity?"):
    """
    Test vector retrieval against the indexed Pinecone database.
    Requires populated Pinecone index and valid credentials.
    """
    missing = config.get_missing_credentials()
    if missing:
        print(f"[!] Error: Cannot verify retrieval. Missing configuration in .env: {', '.join(missing)}")
        print("Please edit .env and configure your credentials before running verification.")
        return

    print(f"[?] Testing retrieval with query: '{query}'")
    try:
        # Check index existence and vector count first
        stats = get_index_stats()
        if not stats.get("exists", False):
            print(f"[!] Pinecone index '{config.PINECONE_INDEX_NAME}' does not exist.")
            print("Please run ingestion first using: python -m src.ingest")
            return

        vector_count = stats.get("total_vector_count", 0)
        if vector_count == 0:
            print(f"[!] Pinecone index '{config.PINECONE_INDEX_NAME}' is currently empty (0 vectors).")
            print("Please run ingestion first using: python -m src.ingest")
            return

        embedding_model = get_embedding_model()
        vector_store = get_vector_store(embedding_model=embedding_model)
        results = vector_store.similarity_search_with_score(query, k=3)

        if not results:
            print("[!] No matching results returned. The index might be empty or query unindexed.")
            return

        print(f"\n[+] Found {len(results)} relevant chunk(s):")
        for idx, (doc, score) in enumerate(results, 1):
            print(f"\n--- Result {idx} (Similarity Score: {score:.4f}) ---")
            print(f"Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}")
            print(f"Chunk ID: {doc.metadata.get('chunk_id')}")
            snippet = doc.page_content.strip()
            print(f"Content: {snippet[:200]}...")

    except Exception as exc:
        print(f"[!] Retrieval verification failed: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest DSA PDFs into Pinecone vector store with 429 quota protection.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test PDF reading and chunking locally without calling external APIs.",
    )
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Delete existing index and recreate it from scratch.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore checkpoint and re-ingest all chunks from the beginning.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.EMBEDDING_BATCH_SIZE,
        help=f"Number of document chunks per upsert batch (default: {config.EMBEDDING_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--batch-delay",
        type=float,
        default=config.INTER_BATCH_DELAY,
        help=f"Seconds to wait between batches to avoid rate limit spikes (default: {config.INTER_BATCH_DELAY}s).",
    )
    parser.add_argument(
        "--verify",
        type=str,
        nargs="?",
        const="What is an array and its time complexity?",
        help="Verify retrieval by searching the index with an optional query.",
    )

    args = parser.parse_args()

    if args.verify is not None:
        verify_retrieval(args.verify)
    else:
        success = run_ingestion(
            dry_run=args.dry_run,
            force_recreate=args.recreate_index,
            batch_size=args.batch_size,
            inter_batch_delay=args.batch_delay,
            resume=not args.no_resume,
        )
        sys.exit(0 if success else 1)
