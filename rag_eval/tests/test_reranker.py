from __future__ import annotations

import unittest

from langchain_core.documents import Document

from rag_eval.src.reranker import CrossEncoderReranker


class FakeModel:
    def predict(self, pairs):
        self.pairs = pairs
        return [0.1, 0.9, 0.5]


class RerankerTests(unittest.TestCase):
    def test_cross_encoder_resorts_and_preserves_original_rank(self) -> None:
        model = FakeModel()
        reranker = CrossEncoderReranker("fake", model=model)
        docs = [Document(page_content=value) for value in ["a", "b", "c"]]
        result = reranker.rerank("query", docs, top_k=2)
        self.assertEqual([doc.page_content for doc in result.documents], ["b", "c"])
        self.assertEqual(
            [doc.metadata["pre_rerank_rank"] for doc in result.documents], [2, 3]
        )
        self.assertEqual(result.scores, [0.9, 0.5])
        self.assertEqual(model.pairs[0], ("query", "a"))


if __name__ == "__main__":
    unittest.main()
