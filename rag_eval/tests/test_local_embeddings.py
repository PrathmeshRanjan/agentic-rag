from __future__ import annotations

import unittest

from rag_eval.src.vector_store import SentenceTransformerEmbeddings


class FakeSentenceTransformer:
    def encode(self, value, normalize_embeddings):
        self.call = (value, normalize_embeddings)
        if isinstance(value, list):
            return [[1, 2] for _ in value]
        return [3, 4]


class LocalEmbeddingTests(unittest.TestCase):
    def test_adapter_normalizes_document_and_query_vectors(self) -> None:
        embeddings = SentenceTransformerEmbeddings("fake")
        embeddings._model = FakeSentenceTransformer()
        self.assertEqual(embeddings.embed_documents(["a", "b"]), [[1.0, 2.0], [1.0, 2.0]])
        self.assertEqual(embeddings.embed_query("q"), [3.0, 4.0])
        self.assertEqual(embeddings._model.call, ("q", True))


if __name__ == "__main__":
    unittest.main()
