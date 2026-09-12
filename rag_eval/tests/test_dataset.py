from __future__ import annotations

import unittest
from pathlib import Path

from rag_eval.src.dataset import load_eval_dataset, validate_dataset
from rag_eval.src.pipeline import load_raw_documents

ROOT = Path(__file__).resolve().parents[1]


class DatasetIntegrationTests(unittest.TestCase):
    def test_all_gold_quotes_exist_in_declared_source_and_page(self) -> None:
        dataset = load_eval_dataset(ROOT / "data" / "eval_dataset.json")
        documents = load_raw_documents(ROOT / "data" / "raw_documents")
        self.assertEqual(len(dataset.examples), 40)
        self.assertEqual(validate_dataset(dataset, documents), [])
        self.assertGreaterEqual(
            sum(len(example.gold_evidence) > 1 for example in dataset.examples), 5
        )


if __name__ == "__main__":
    unittest.main()
