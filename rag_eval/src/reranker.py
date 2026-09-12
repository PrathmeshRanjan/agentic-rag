"""Cross-encoder reranking with explicit cold-start and inference timings."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from langchain_core.documents import Document

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RerankResult:
    documents: list[Document]
    latency_ms: float
    model_load_latency_ms: float = 0.0
    scores: list[float] = field(default_factory=list)


class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    def warmup(self) -> float: ...

    def rerank(
        self, query: str, documents: Sequence[Document], top_k: int
    ) -> RerankResult: ...


class NoOpReranker:
    @property
    def name(self) -> str:
        return "none"

    def warmup(self) -> float:
        return 0.0

    def rerank(
        self, query: str, documents: Sequence[Document], top_k: int
    ) -> RerankResult:
        del query
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        start = time.perf_counter()
        selected = list(documents[:top_k])
        return RerankResult(
            documents=selected,
            latency_ms=(time.perf_counter() - start) * 1000,
        )


class CrossEncoderReranker:
    """Locally rescore the exact same candidate set as the baseline."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, model: Any = None):
        self.model_name = model_name
        self._model = model

    @property
    def name(self) -> str:
        return self.model_name

    def _load_model(self) -> tuple[Any, float]:
        if self._model is not None:
            return self._model, 0.0
        start = time.perf_counter()
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Cross-encoder experiments require sentence-transformers"
            ) from exc
        self._model = CrossEncoder(self.model_name)
        return self._model, (time.perf_counter() - start) * 1000

    def warmup(self) -> float:
        """Load the model once; return setup latency excluded from query latency."""
        _, load_ms = self._load_model()
        return load_ms

    def rerank(
        self, query: str, documents: Sequence[Document], top_k: int
    ) -> RerankResult:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not documents:
            return RerankResult(documents=[], latency_ms=0.0)

        model, load_ms = self._load_model()
        pairs = [(query, document.page_content) for document in documents]
        start = time.perf_counter()
        raw_scores = model.predict(pairs)
        inference_ms = (time.perf_counter() - start) * 1000
        scores = [float(score) for score in raw_scores]
        scored = sorted(
            zip(scores, range(len(documents)), documents),
            key=lambda item: (-item[0], item[1]),
        )[:top_k]

        output = []
        output_scores = []
        for rerank_rank, (score, original_index, document) in enumerate(scored, start=1):
            metadata = {
                **document.metadata,
                "reranker_score": score,
                "rerank_rank": rerank_rank,
                "pre_rerank_rank": original_index + 1,
            }
            output.append(Document(page_content=document.page_content, metadata=metadata))
            output_scores.append(score)

        return RerankResult(
            documents=output,
            latency_ms=inference_ms + load_ms,
            model_load_latency_ms=load_ms,
            scores=output_scores,
        )
