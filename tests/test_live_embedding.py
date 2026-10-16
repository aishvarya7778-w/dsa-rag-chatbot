"""
Minimal Live Embedding Test for DSA RAG Chatbot.
Verifies connectivity with Google's gemini-embedding-001 model
and confirms output dimensionality is exactly 768 for both queries and documents.
Never prints or exposes secrets.
"""

import sys
import unittest
import config
from src.embeddings import get_embedding_model


class TestLiveGeminiEmbedding(unittest.TestCase):
    """Live verification tests for Gemini embeddings with 768 dimensions."""

    def setUp(self):
        if not config.GEMINI_API_KEY:
            self.skipTest("GEMINI_API_KEY is not configured in .env")
        self.embedding_model = get_embedding_model()

    def test_model_configuration(self):
        """Verify model name and output_dimensionality attributes."""
        self.assertEqual(self.embedding_model.model, "gemini-embedding-001")
        self.assertEqual(self.embedding_model.output_dimensionality, 768)

    def test_live_query_embedding_dimension(self):
        """Verify query embedding produces exactly 768 dimensions."""
        test_text = "Data Structures and Algorithms"
        vector = self.embedding_model.embed_query(test_text)
        self.assertIsInstance(vector, list)
        self.assertEqual(len(vector), 768)
        self.assertTrue(all(isinstance(val, float) for val in vector))

    def test_live_document_embedding_dimension(self):
        """Verify document embedding produces exactly 768 dimensions."""
        test_docs = ["An array is a collection of items stored at contiguous memory locations."]
        vectors = self.embedding_model.embed_documents(test_docs)
        self.assertIsInstance(vectors, list)
        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), 768)
        self.assertTrue(all(isinstance(val, float) for val in vectors[0]))


if __name__ == "__main__":
    unittest.main()
