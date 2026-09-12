from __future__ import annotations

import unittest

from langchain_core.documents import Document

from rag_eval.src.vector_store import ChromaVectorIndex, _chroma_safe_document


class FakeChroma:
    def similarity_search_with_score(self, query, k):
        self.call = (query, k)
        return [(Document(page_content="answer", metadata={"source": "a.md"}), 0.25)]


class VectorStoreTests(unittest.TestCase):
    def test_chroma_adapter_preserves_scores_and_rank(self) -> None:
        store = FakeChroma()
        results = ChromaVectorIndex(store).similarity_search("question", 5)
        self.assertEqual(store.call, ("question", 5))
        self.assertEqual(results[0].metadata["retrieval_score"], 0.8)
        self.assertEqual(results[0].metadata["retrieval_distance"], 0.25)
        self.assertEqual(results[0].metadata["retrieval_rank"], 1)

    def test_chroma_metadata_is_scalar_only(self) -> None:
        document = Document(
            page_content="content",
            metadata={"source": "a.md", "none": None, "nested": {"page": 1}},
        )
        safe = _chroma_safe_document(document)
        self.assertNotIn("none", safe.metadata)
        self.assertEqual(safe.metadata["nested"], '{"page": 1}')


if __name__ == "__main__":
    unittest.main()
