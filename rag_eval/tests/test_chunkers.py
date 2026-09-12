from __future__ import annotations

import unittest

from langchain_core.documents import Document

from rag_eval.src.chunkers import chunk_fixed, chunk_recursive


class ChunkerTests(unittest.TestCase):
    def test_fixed_windows_have_expected_overlap_and_stable_ids(self) -> None:
        document = Document(page_content="abcdefghij", metadata={"source": "a.md", "page": 1})
        first = chunk_fixed([document], chunk_size=6, chunk_overlap=2)
        second = chunk_fixed([document], chunk_size=6, chunk_overlap=2)
        self.assertEqual([chunk.page_content for chunk in first], ["abcdef", "efghij"])
        self.assertEqual([chunk.metadata["start_index"] for chunk in first], [0, 4])
        self.assertEqual(
            [chunk.metadata["chunk_id"] for chunk in first],
            [chunk.metadata["chunk_id"] for chunk in second],
        )

    def test_chunk_indices_restart_for_each_source_document(self) -> None:
        documents = [
            Document(page_content="one " * 20, metadata={"source": "a.md", "page": 1}),
            Document(page_content="two " * 20, metadata={"source": "b.md", "page": 1}),
        ]
        chunks = chunk_recursive(documents, chunk_size=30, chunk_overlap=5)
        first_for_b = next(chunk for chunk in chunks if chunk.metadata["source"] == "b.md")
        self.assertEqual(first_for_b.metadata["chunk_index"], 0)

    def test_invalid_overlap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chunk_fixed([Document(page_content="text")], chunk_size=10, chunk_overlap=10)


if __name__ == "__main__":
    unittest.main()
