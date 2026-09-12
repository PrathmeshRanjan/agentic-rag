from __future__ import annotations

import unittest

from rag_eval.src.metrics import (
    GoldEvidence,
    QueryResult,
    RetrievedChunk,
    chunk_matches_evidence,
    full_recall_at_k,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


class MetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first = GoldEvidence("e1", "a.md", "alpha beta gamma delta", page=1)
        self.second = GoldEvidence("e2", "b.md", "one two three four", page=2)

    def test_source_and_page_are_part_of_relevance(self) -> None:
        text = "alpha beta gamma delta"
        self.assertTrue(chunk_matches_evidence(RetrievedChunk("a.md", text, page=1), self.first))
        self.assertFalse(chunk_matches_evidence(RetrievedChunk("b.md", text, page=1), self.first))
        self.assertFalse(chunk_matches_evidence(RetrievedChunk("a.md", text, page=2), self.first))

    def test_duplicate_chunks_do_not_inflate_evidence_recall(self) -> None:
        result = QueryResult(
            query_id="q1",
            query="question",
            gold_evidence=[self.first, self.second],
            retrieved=[
                RetrievedChunk("a.md", "alpha beta gamma delta", page=1),
                RetrievedChunk("a.md", "alpha beta gamma delta repeated", page=1),
            ],
            retrieval_latency_ms=1,
        )
        self.assertEqual(hit_rate_at_k([result], 2), 1.0)
        self.assertEqual(mrr_at_k([result], 2), 1.0)
        self.assertEqual(recall_at_k([result], 2), 0.5)
        self.assertEqual(full_recall_at_k([result], 2), 0.0)

    def test_multi_evidence_ranking_metrics(self) -> None:
        result = QueryResult(
            query_id="q1",
            query="question",
            gold_evidence=[self.first, self.second],
            retrieved=[
                RetrievedChunk("irrelevant.md", "noise"),
                RetrievedChunk("a.md", "alpha beta gamma delta", page=1),
                RetrievedChunk("b.md", "one two three four", page=2),
            ],
            retrieval_latency_ms=1,
        )
        self.assertEqual(mrr_at_k([result], 3), 0.5)
        self.assertEqual(recall_at_k([result], 3), 1.0)
        self.assertEqual(full_recall_at_k([result], 3), 1.0)
        self.assertGreater(ndcg_at_k([result], 3), 0.6)


if __name__ == "__main__":
    unittest.main()
