from __future__ import annotations

import unittest

from langchain_core.documents import Document

from rag_eval.src.pipeline import PipelineConfig, RetrievalPipeline
from rag_eval.src.vector_store import HashingEmbeddings


class PipelineTests(unittest.TestCase):
    def test_retrieve_keeps_full_candidate_pool_before_output_slice(self) -> None:
        config = PipelineConfig(
            chunking={"strategy": "fixed", "chunk_size": 80, "chunk_overlap": 0},
            retrieval_k=3,
            output_k=2,
            vector_backend="memory",
            embedding_provider="hash",
        )
        pipeline = RetrievalPipeline(config, embeddings=HashingEmbeddings(64))
        documents = [
            Document(page_content=f"document {index} unique{index}", metadata={"source": f"{index}.md"})
            for index in range(4)
        ]
        pipeline.build(documents)
        raw = pipeline.retrieve("document")
        final = pipeline.query("document")
        self.assertEqual(len(raw.documents), 3)
        self.assertEqual(len(final.documents), 2)


if __name__ == "__main__":
    unittest.main()
