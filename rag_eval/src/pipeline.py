"""Configurable retrieval pipeline with reusable first-stage candidates."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_eval.src.chunkers import apply_chunking, config_name
from rag_eval.src.reranker import NoOpReranker, Reranker
from rag_eval.src.vector_store import VectorIndex, build_index, cleanup, get_embeddings

RAW_DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_documents"


@dataclass(frozen=True)
class PipelineConfig:
    chunking: dict[str, Any]
    retrieval_k: int = 10
    output_k: int = 10
    vector_backend: str = "chroma"
    embedding_provider: str = "google"
    embedding_model: str = "gemini-embedding-001"

    def __post_init__(self) -> None:
        if self.retrieval_k <= 0 or self.output_k <= 0:
            raise ValueError("retrieval_k and output_k must be positive")
        if self.output_k > self.retrieval_k:
            raise ValueError("output_k cannot exceed retrieval_k")

    @property
    def name(self) -> str:
        return config_name(self.chunking)


@dataclass
class RetrievalResult:
    documents: list[Document]
    retrieval_latency_ms: float
    rerank_latency_ms: float = 0.0
    reranker: str = "none"


def load_raw_documents(
    docs_dir: Path | None = None, *, strict: bool = True
) -> list[Document]:
    """Load PDF, Markdown, and text files with normalized one-based page metadata."""
    directory = docs_dir or RAW_DOCS_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"Raw document directory does not exist: {directory}")
    documents: list[Document] = []
    failures: list[str] = []
    for filepath in sorted(directory.iterdir()):
        if not filepath.is_file() or filepath.suffix.lower() not in {".pdf", ".md", ".txt"}:
            continue
        try:
            if filepath.suffix.lower() == ".pdf":
                loaded = PyPDFLoader(str(filepath)).load()
            else:
                loaded = TextLoader(str(filepath), encoding="utf-8").load()
            for document_index, document in enumerate(loaded):
                original_page = document.metadata.get("page")
                page = int(original_page) + 1 if original_page is not None else 1
                document.metadata.update(
                    {
                        "source": filepath.name,
                        "page": page,
                        "document_index": document_index,
                    }
                )
            documents.extend(loaded)
        except Exception as exc:  # pragma: no cover - malformed external files
            failures.append(f"{filepath.name}: {exc}")
    if strict and failures:
        raise RuntimeError("Failed to load source documents: " + "; ".join(failures))
    if not documents:
        raise RuntimeError(f"No supported documents found in {directory}")
    return documents


class RetrievalPipeline:
    """Build once, retrieve once, and rerank identical candidates for fair A/B tests."""

    def __init__(self, config: PipelineConfig, embeddings: Embeddings | None = None):
        self.config = config
        self.embeddings = embeddings or get_embeddings(
            provider=config.embedding_provider,
            model_name=config.embedding_model,
        )
        self._index: VectorIndex | None = None
        self._chunks: list[Document] = []

    @property
    def chunks(self) -> Sequence[Document]:
        return tuple(self._chunks)

    def build(self, documents: Sequence[Document]) -> int:
        self._chunks = apply_chunking(documents, self.config.chunking, self.embeddings)
        self._index = build_index(
            self._chunks,
            self.config.name,
            embeddings=self.embeddings,
            backend=self.config.vector_backend,
        )
        return len(self._chunks)

    def retrieve(self, query: str) -> RetrievalResult:
        if self._index is None:
            raise RuntimeError("Pipeline not built; call build() first")
        start = time.perf_counter()
        documents = self._index.similarity_search(query, k=self.config.retrieval_k)
        elapsed = (time.perf_counter() - start) * 1000
        return RetrievalResult(
            # Keep the complete candidate pool. Baseline and reranked output
            # are sliced identically only after this shared retrieval step.
            documents=list(documents),
            retrieval_latency_ms=elapsed,
        )

    def rerank(
        self,
        query: str,
        candidates: Sequence[Document],
        retrieval_latency_ms: float,
        reranker: Reranker,
    ) -> RetrievalResult:
        result = reranker.rerank(query, candidates, top_k=self.config.output_k)
        return RetrievalResult(
            documents=result.documents,
            retrieval_latency_ms=retrieval_latency_ms,
            rerank_latency_ms=result.latency_ms,
            reranker=reranker.name,
        )

    def query(self, query: str, reranker: Reranker | None = None) -> RetrievalResult:
        """Convenience API; benchmark.py uses retrieve/rerank for paired isolation."""
        raw = self.retrieve(query)
        active = reranker or NoOpReranker()
        return self.rerank(
            query,
            raw.documents,
            raw.retrieval_latency_ms,
            active,
        )

    def cleanup(self) -> None:
        self._index = None
        self._chunks = []
        cleanup(self.config.name, self.config.vector_backend)
