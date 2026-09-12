"""Pure, evidence-aware retrieval metrics for auditable RAG evaluation."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence


@dataclass(frozen=True)
class GoldEvidence:
    evidence_id: str
    source: str
    text: str
    section: str = ""
    page: int | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    content: str
    chunk_id: str = ""
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    query_id: str
    query: str
    gold_evidence: list[GoldEvidence]
    retrieved: list[RetrievedChunk]
    retrieval_latency_ms: float
    rerank_latency_ms: float = 0.0
    error: str | None = None


def _normalized_source(value: str) -> str:
    return Path(value.replace("\\", "/")).name.casefold()


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def passage_coverage(retrieved_content: str, gold_text: str) -> float:
    """Fraction of gold tokens present, retaining duplicate-token counts."""
    gold = Counter(_tokens(gold_text))
    if not gold:
        return 0.0
    retrieved = Counter(_tokens(retrieved_content))
    overlap = sum((gold & retrieved).values())
    return overlap / sum(gold.values())


def chunk_matches_evidence(
    chunk: RetrievedChunk,
    evidence: GoldEvidence,
    minimum_coverage: float = 0.6,
) -> bool:
    """Match by source/page and passage coverage, never by heading alone."""
    if _normalized_source(chunk.source) != _normalized_source(evidence.source):
        return False
    if evidence.page is not None and chunk.page is not None and evidence.page != chunk.page:
        return False
    normalized_gold = " ".join(_tokens(evidence.text))
    normalized_chunk = " ".join(_tokens(chunk.content))
    if normalized_gold and normalized_gold in normalized_chunk:
        return True
    return passage_coverage(chunk.content, evidence.text) >= minimum_coverage


def _unique_matches(result: QueryResult, k: int) -> list[tuple[int, str]]:
    """Greedily match ranked chunks to distinct evidence IDs."""
    matched: set[str] = set()
    matches: list[tuple[int, str]] = []
    for rank, chunk in enumerate(result.retrieved[:k], start=1):
        for evidence in result.gold_evidence:
            if evidence.evidence_id in matched:
                continue
            if chunk_matches_evidence(chunk, evidence):
                matched.add(evidence.evidence_id)
                matches.append((rank, evidence.evidence_id))
                break
    return matches


def hit_rate_at_k(results: Sequence[QueryResult], k: int) -> float:
    return mean(bool(_unique_matches(result, k)) for result in results) if results else 0.0


def mrr_at_k(results: Sequence[QueryResult], k: int) -> float:
    if not results:
        return 0.0
    reciprocal_ranks = []
    for result in results:
        matches = _unique_matches(result, k)
        reciprocal_ranks.append(1.0 / matches[0][0] if matches else 0.0)
    return mean(reciprocal_ranks)


def recall_at_k(results: Sequence[QueryResult], k: int) -> float:
    """Macro-average recall over distinct gold evidence passages."""
    if not results:
        return 0.0
    recalls = []
    for result in results:
        denominator = len(result.gold_evidence)
        recalls.append(len(_unique_matches(result, k)) / denominator if denominator else 0.0)
    return mean(recalls)


def full_recall_at_k(results: Sequence[QueryResult], k: int) -> float:
    """Fraction of questions for which every required passage was retrieved."""
    if not results:
        return 0.0
    return mean(
        bool(result.gold_evidence)
        and len(_unique_matches(result, k)) == len(result.gold_evidence)
        for result in results
    )


def precision_at_k(results: Sequence[QueryResult], k: int) -> float:
    if not results:
        return 0.0
    values = []
    for result in results:
        denominator = min(k, len(result.retrieved))
        values.append(len(_unique_matches(result, k)) / denominator if denominator else 0.0)
    return mean(values)


def ndcg_at_k(results: Sequence[QueryResult], k: int) -> float:
    """Binary NDCG with duplicate evidence matches suppressed."""
    if not results:
        return 0.0
    scores = []
    for result in results:
        ranks = [rank for rank, _ in _unique_matches(result, k)]
        dcg = sum(1.0 / math.log2(rank + 1) for rank in ranks)
        ideal_count = min(k, len(result.gold_evidence))
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        scores.append(dcg / ideal if ideal else 0.0)
    return mean(scores)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@dataclass
class BenchmarkMetrics:
    config_name: str
    base_config: str
    reranker: str
    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_5: float
    recall_at_5: float
    full_recall_at_5: float
    mrr_at_5: float
    ndcg_at_5: float
    precision_at_5: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    avg_retrieval_latency_ms: float
    avg_rerank_latency_ms: float
    num_queries: int
    failed_queries: int


def compute_all_metrics(
    config_name: str,
    base_config: str,
    reranker: str,
    results: Sequence[QueryResult],
) -> BenchmarkMetrics:
    total_latencies = [
        result.retrieval_latency_ms + result.rerank_latency_ms for result in results
    ]
    retrieval_latencies = [result.retrieval_latency_ms for result in results]
    rerank_latencies = [result.rerank_latency_ms for result in results]
    return BenchmarkMetrics(
        config_name=config_name,
        base_config=base_config,
        reranker=reranker,
        hit_rate_at_1=hit_rate_at_k(results, 1),
        hit_rate_at_3=hit_rate_at_k(results, 3),
        hit_rate_at_5=hit_rate_at_k(results, 5),
        recall_at_5=recall_at_k(results, 5),
        full_recall_at_5=full_recall_at_k(results, 5),
        mrr_at_5=mrr_at_k(results, 5),
        ndcg_at_5=ndcg_at_k(results, 5),
        precision_at_5=precision_at_k(results, 5),
        avg_latency_ms=mean(total_latencies) if total_latencies else 0.0,
        p50_latency_ms=median(total_latencies) if total_latencies else 0.0,
        p95_latency_ms=_percentile(total_latencies, 0.95),
        avg_retrieval_latency_ms=mean(retrieval_latencies) if retrieval_latencies else 0.0,
        avg_rerank_latency_ms=mean(rerank_latencies) if rerank_latencies else 0.0,
        num_queries=len(results),
        failed_queries=sum(result.error is not None for result in results),
    )
