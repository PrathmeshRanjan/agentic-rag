from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_eval.generate_report import (
    README_END,
    README_START,
    generate_report,
    update_readme,
)


def payload(eligible: bool = True):
    metrics = {
        "hit_rate_at_1": 0.5,
        "hit_rate_at_3": 0.7,
        "hit_rate_at_5": 0.8,
        "recall_at_5": 0.75,
        "full_recall_at_5": 0.6,
        "mrr_at_5": 0.65,
        "ndcg_at_5": 0.7,
        "p95_latency_ms": 12.0,
        "failed_queries": 0,
    }
    return {
        "run": {
            "run_id": "run-1",
            "quality_claim_eligible": eligible,
            "vector_backend": "chroma",
            "embedding_provider": "google",
            "embedding_model": "model",
            "retrieval_k": 20,
            "output_k": 10,
        },
        "dataset": {
            "sha256": "abc",
            "evaluated_queries": 40,
            "num_sources": 3,
            "num_multi_evidence_queries": 7,
        },
        "experiments": [
            {
                "config_name": "recursive_512_100",
                "base_config": "recursive_512_100",
                "chunking": {"strategy": "recursive"},
                "reranker": "none",
                "num_chunks": 10,
                "metrics": metrics,
            }
        ],
    }


class ReportTests(unittest.TestCase):
    def test_report_contains_ranked_metrics_and_findings(self) -> None:
        report = generate_report(payload())
        self.assertIn("Evidence Recall@5", report)
        self.assertIn("Key findings", report)
        self.assertIn("| Rank |", report)

    def test_smoke_results_cannot_overwrite_readme_claims(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            readme = Path(directory) / "README.md"
            readme.write_text(f"before\n{README_START}\nold\n{README_END}\nafter")
            with self.assertRaises(ValueError):
                update_readme(readme, payload(False))

    def test_readme_update_is_marker_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            readme = Path(directory) / "README.md"
            readme.write_text(f"before\n{README_START}\nold\n{README_END}\nafter")
            update_readme(readme, payload(True))
            updated = readme.read_text()
            self.assertTrue(updated.startswith("before"))
            self.assertTrue(updated.endswith("after"))
            self.assertIn("Top five configurations", updated)
            self.assertNotIn("\nold\n", updated)


if __name__ == "__main__":
    unittest.main()
