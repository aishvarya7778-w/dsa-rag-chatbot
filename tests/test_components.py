"""
Unit tests for core DSA RAG Chatbot modules.
Tests configuration, text chunking, and document citation extraction.
Does not make external network or API calls.
"""

import unittest
from pathlib import Path
from langchain_core.documents import Document
import config
from src.text_splitter import split_documents, generate_chunk_id
from src.retriever import format_docs_for_prompt, extract_source_citations


class TestDSAConfig(unittest.TestCase):
    """Test configuration values and credential validation."""

    def test_default_config_values(self):
        self.assertEqual(config.EMBEDDING_DIMENSION, 768)
        self.assertEqual(config.PINECONE_METRIC, "cosine")
        self.assertIn("gemini-embedding-001", config.EMBEDDING_MODEL)
        self.assertEqual(config.LLM_MODEL, "gemini-3.5-flash")
        self.assertGreater(config.CHUNK_SIZE, 0)
        self.assertGreater(config.CHUNK_OVERLAP, 0)

    def test_missing_credentials_detection(self):
        missing = config.get_missing_credentials()
        self.assertIsInstance(missing, list)


class TestTextSplitter(unittest.TestCase):
    """Test document chunking and deterministic ID generation."""

    def setUp(self):
        self.sample_doc = Document(
            page_content=(
                "An array is a linear data structure that stores elements in contiguous memory locations.\n\n"
                "In Python, arrays are typically implemented using lists.\n\n"
                "```python\ndef linear_search(arr, target):\n    for i in range(len(arr)):\n"
                "        if arr[i] == target:\n            return i\n    return -1\n```\n\n"
                "Time Complexity:\n- Best Case: O(1)\n- Worst Case: O(n)\n- Space Complexity: O(1)"
            ),
            metadata={"source": "arrays_guide.pdf", "page": 3},
        )

    def test_deterministic_chunk_id(self):
        id_1 = generate_chunk_id(self.sample_doc, chunk_idx=0)
        id_2 = generate_chunk_id(self.sample_doc, chunk_idx=0)
        self.assertEqual(id_1, id_2)
        self.assertTrue(id_1.startswith("arrays_guide.pdf_p3_c0_"))

    def test_split_documents(self):
        chunks = split_documents([self.sample_doc], chunk_size=200, chunk_overlap=30)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIn("chunk_id", chunk.metadata)
            self.assertEqual(chunk.metadata["source"], "arrays_guide.pdf")
            self.assertEqual(chunk.metadata["page"], 3)


class TestRetrieverFormatting(unittest.TestCase):
    """Test context string formatting and citation extraction."""

    def setUp(self):
        self.docs = [
            Document(
                page_content="Binary search operates on sorted arrays in O(log n) time.",
                metadata={"source": "searching.pdf", "page": 5},
            ),
            Document(
                page_content="Merge sort has O(n log n) time complexity in all cases.",
                metadata={"source": "sorting.pdf", "page": 12},
            ),
        ]

    def test_format_docs_for_prompt(self):
        prompt_text = format_docs_for_prompt(self.docs)
        self.assertIn("searching.pdf", prompt_text)
        self.assertIn("Page 5", prompt_text)
        self.assertIn("sorting.pdf", prompt_text)
        self.assertIn("Page 12", prompt_text)

    def test_extract_source_citations(self):
        citations = extract_source_citations(self.docs)
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0]["source"], "searching.pdf")
        self.assertEqual(citations[0]["page"], 5)
        self.assertEqual(citations[1]["source"], "sorting.pdf")
        self.assertEqual(citations[1]["page"], 12)


class TestPDFLoader(unittest.TestCase):
    """Test PDF loader path resolution and handling of non-existent files."""

    def test_nonexistent_pdf_raises_error(self):
        from src.pdf_loader import load_single_pdf
        with self.assertRaises(FileNotFoundError):
            load_single_pdf(Path("data/pdfs/does_not_exist.pdf"))

    def test_nonexistent_directory_raises_error(self):
        from src.pdf_loader import load_all_pdfs
        with self.assertRaises(FileNotFoundError):
            load_all_pdfs(Path("non_existent_folder_xyz"))

    def test_empty_directory_handling(self):
        import tempfile
        from src.pdf_loader import list_available_pdfs, load_all_pdfs
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_path = Path(tmpdir)
            available = list_available_pdfs(empty_path)
            self.assertEqual(len(available), 0)
            docs = load_all_pdfs(empty_path)
            self.assertEqual(len(docs), 0)


class TestCredentialSafetyAndErrors(unittest.TestCase):
    """Test that missing credentials raise clean ValueErrors without leaking secrets."""

    def test_missing_gemini_key_raises_clear_error(self):
        from src.embeddings import get_embedding_model
        with self.assertRaises(ValueError) as ctx:
            get_embedding_model(api_key="")
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    def test_missing_llm_key_raises_clear_error(self):
        from src.rag_chain import get_llm
        with self.assertRaises(ValueError) as ctx:
            get_llm(api_key="")
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    def test_missing_pinecone_key_raises_clear_error(self):
        from src.vector_store import get_pinecone_client
        with self.assertRaises(ValueError) as ctx:
            get_pinecone_client(api_key="")
        self.assertIn("PINECONE_API_KEY", str(ctx.exception))


class TestOfflineModelInstantiation(unittest.TestCase):
    """Verify Google GenAI model classes can be instantiated offline with mock keys."""

    def test_embedding_model_instantiation(self):
        from src.embeddings import get_embedding_model
        emb = get_embedding_model(api_key="mock-key-for-test")
        self.assertEqual(emb.model, "gemini-embedding-001")
        self.assertEqual(emb.output_dimensionality, 768)

    def test_embedding_model_custom_dim(self):
        from src.embeddings import get_embedding_model
        emb = get_embedding_model(api_key="mock-key-for-test", output_dimensionality=512)
        self.assertEqual(emb.model, "gemini-embedding-001")
        self.assertEqual(emb.output_dimensionality, 512)

    def test_llm_model_instantiation(self):
        from src.rag_chain import get_llm, create_dsa_prompt_template
        llm = get_llm(api_key="mock-key-for-test")
        self.assertEqual(llm.model, "gemini-3.5-flash")
        
        prompt = create_dsa_prompt_template()
        messages = prompt.format_messages(context="Array context", question="What is an array?")
        self.assertEqual(len(messages), 2)
        self.assertIn("Array context", messages[0].content)


class TestRetryBackoffAndResumability(unittest.TestCase):
    """Test retry delay extraction, 429 quota detection, backoff, and checkpointing."""

    def test_extract_retry_delay_from_message(self):
        from src.embeddings import extract_retry_delay
        e1 = Exception("429 RESOURCE_EXHAUSTED. Please retry in 34.2s.")
        self.assertAlmostEqual(extract_retry_delay(e1), 34.2)

        e2 = Exception("Please retry after 15 seconds due to rate limits.")
        self.assertAlmostEqual(extract_retry_delay(e2), 15.0)

        e3 = Exception("Some other error without retry delay.")
        self.assertIsNone(extract_retry_delay(e3))

    def test_is_quota_or_rate_limit_error(self):
        from src.embeddings import is_quota_or_rate_limit_error
        e1 = Exception("429 RESOURCE_EXHAUSTED: quota limit exceeded")
        self.assertTrue(is_quota_or_rate_limit_error(e1))

        class MockClientError(Exception):
            code = 429
            status = "RESOURCE_EXHAUSTED"

        self.assertTrue(is_quota_or_rate_limit_error(MockClientError()))
        self.assertFalse(is_quota_or_rate_limit_error(ValueError("Invalid syntax")))

    def test_embed_with_backoff_exhaustion(self):
        from src.embeddings import embed_with_backoff, QuotaExhaustedError
        from unittest.mock import patch

        call_count = 0

        def failing_fn():
            nonlocal call_count
            call_count += 1
            raise Exception("429 RESOURCE_EXHAUSTED: Please retry in 0.1s.")

        with patch("time.sleep") as mock_sleep:
            with self.assertRaises(QuotaExhaustedError) as ctx:
                embed_with_backoff(
                    failing_fn,
                    operation_name="test",
                    max_retries=3,
                    initial_delay=0.1,
                    backoff_factor=1.5,
                )
            self.assertEqual(call_count, 3)
            self.assertEqual(mock_sleep.call_count, 2)
            self.assertIn("Gemini API quota remains exhausted", str(ctx.exception))

    def test_checkpoint_save_and_load(self):
        import tempfile
        from src.ingest import save_checkpoint, load_checkpoint, clear_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            cp_path = Path(tmpdir) / "ingestion_checkpoint.json"

            # Initially empty
            data = load_checkpoint(cp_path)
            self.assertEqual(len(data["completed_ids"]), 0)
            self.assertFalse(data["is_complete"])

            # Save partial progress
            ids = {"chunk_1", "chunk_2"}
            save_checkpoint(ids, total_chunks=10, is_complete=False, checkpoint_file=cp_path)

            loaded = load_checkpoint(cp_path)
            self.assertEqual(loaded["completed_ids"], ids)
            self.assertEqual(loaded["total_chunks"], 10)
            self.assertFalse(loaded["is_complete"])

            # Save complete progress
            ids.add("chunk_3")
            save_checkpoint(ids, total_chunks=10, is_complete=True, checkpoint_file=cp_path)
            loaded2 = load_checkpoint(cp_path)
            self.assertEqual(len(loaded2["completed_ids"]), 3)
            self.assertTrue(loaded2["is_complete"])

            # Clear
            clear_checkpoint(cp_path)
            self.assertFalse(cp_path.exists())


if __name__ == "__main__":
    unittest.main()
