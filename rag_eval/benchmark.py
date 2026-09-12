"""Reproducible chunking and cross-encoder retrieval benchmark.

Production: ``python -m rag_eval.benchmark``
CI smoke test: ``python -m rag_eval.benchmark --quick``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv

load_dotenv()

from rag_eval.src.chunkers import CHUNKING_CONFIGS, config_name
from rag_eval.src.dataset import (
    EvalExample,
    dataset_summary,
    load_eval_dataset,
    validate_dataset,
)
from rag_eval.src.metrics import (
    QueryResult,
    RetrievedChunk,
    chunk_matches_evidence,
    compute_all_metrics,
)
from rag_eval.src.pipeline import PipelineConfig, RetrievalPipeline, RetrievalResult, load_raw_documents
from rag_eval.src.reranker import CrossEncoderReranker, NoOpReranker, Reranker
from rag_eval.prepare_data import check_snapshot

ROOT = Path(__file__).resolve().parent
EVAL_DATASET_PATH = ROOT / "data" / "eval_dataset.json"
RESULTS_PATH = ROOT / "data" / "benchmark_results.json"
RAW_DOCUMENTS_PATH = ROOT / "data" / "raw_documents"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _to_retrieved(document: Any) -> RetrievedChunk:
    metadata = dict(document.metadata)
    return RetrievedChunk(
        source=str(metadata.get("source", "")),
        content=document.page_content,
        chunk_id=str(metadata.get("chunk_id", "")),
        page=metadata.get("page"),
        metadata=metadata,
    )


def _query_result(
    example: EvalExample, retrieval: RetrievalResult, output_k: int
) -> QueryResult:
    return QueryResult(
        query_id=example.query_id,
        query=example.query,
        gold_evidence=example.gold_evidence,
        retrieved=[_to_retrieved(doc) for doc in retrieval.documents[:output_k]],
        retrieval_latency_ms=retrieval.retrieval_latency_ms,
        rerank_latency_ms=retrieval.rerank_latency_ms,
    )


def _failed_query(example: EvalExample, message: str) -> QueryResult:
    return QueryResult(
        query_id=example.query_id,
        query=example.query,
        gold_evidence=example.gold_evidence,
        retrieved=[],
        retrieval_latency_ms=0.0,
        error=message,
    )


def _serialize_query(result: QueryResult) -> dict[str, Any]:
    retrieved = []
    for rank, chunk in enumerate(result.retrieved, start=1):
        matched = [
            evidence.evidence_id
            for evidence in result.gold_evidence
            if chunk_matches_evidence(chunk, evidence)
        ]
        retrieved.append(
            {
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "page": chunk.page,
                "retrieval_score": chunk.metadata.get("retrieval_score"),
                "reranker_score": chunk.metadata.get("reranker_score"),
                "pre_rerank_rank": chunk.metadata.get("pre_rerank_rank"),
                "matched_evidence_ids": matched,
                "content": chunk.content,
            }
        )
    return {
        "query_id": result.query_id,
        "query": result.query,
        "gold_evidence_ids": [item.evidence_id for item in result.gold_evidence],
        "retrieval_latency_ms": result.retrieval_latency_ms,
        "rerank_latency_ms": result.rerank_latency_ms,
        "error": result.error,
        "retrieved": retrieved,
    }


def select_chunking_configs(names: Sequence[str] | None) -> list[dict[str, Any]]:
    if not names:
        return list(CHUNKING_CONFIGS)
    available = {config_name(config): config for config in CHUNKING_CONFIGS}
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(
            f"Unknown configs: {', '.join(unknown)}. Available: {', '.join(available)}"
        )
    return [available[name] for name in names]


def run_chunking_experiment(
    chunking: dict[str, Any],
    documents: Sequence[Any],
    examples: Sequence[EvalExample],
    *,
    backend: str,
    embedding_provider: str,
    embedding_model: str,
    retrieval_k: int,
    output_k: int,
    rerankers: Sequence[Reranker],
    keep_index: bool,
    fail_fast: bool,
) -> list[dict[str, Any]]:
    base_name = config_name(chunking)
    config = PipelineConfig(
        chunking=chunking,
        retrieval_k=retrieval_k,
        output_k=output_k,
        vector_backend=backend,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )
    pipeline = RetrievalPipeline(config)
    print(f"\n[{base_name}] building isolated {backend} index...")
    build_start = time.perf_counter()
    try:
        num_chunks = pipeline.build(documents)
    except Exception:
        pipeline.cleanup()
        raise
    index_latency_ms = (time.perf_counter() - build_start) * 1000
    print(f"[{base_name}] indexed {num_chunks} chunks in {index_latency_ms:.0f} ms")

    raw_results: list[RetrievalResult | Exception] = []
    for position, example in enumerate(examples, start=1):
        try:
            raw_results.append(pipeline.retrieve(example.query))
        except Exception as exc:
            if fail_fast:
                pipeline.cleanup()
                raise
            raw_results.append(exc)
        if position % 10 == 0 or position == len(examples):
            print(f"[{base_name}] retrieved {position}/{len(examples)} queries")

    experiment_outputs = []
    for reranker in rerankers:
        query_results = []
        for example, raw in zip(examples, raw_results):
            if isinstance(raw, Exception):
                query_results.append(_failed_query(example, f"{type(raw).__name__}: {raw}"))
                continue
            try:
                ranked = pipeline.rerank(
                    example.query,
                    raw.documents,
                    raw.retrieval_latency_ms,
                    reranker,
                )
                query_results.append(_query_result(example, ranked, output_k))
            except Exception as exc:
                if fail_fast:
                    pipeline.cleanup()
                    raise
                query_results.append(_failed_query(example, f"{type(exc).__name__}: {exc}"))

        name = base_name if reranker.name == "none" else f"{base_name}__rerank"
        metrics = compute_all_metrics(name, base_name, reranker.name, query_results)
        print(
            f"[{name}] Hit@5={metrics.hit_rate_at_5:.3f} "
            f"MRR@5={metrics.mrr_at_5:.3f} Recall@5={metrics.recall_at_5:.3f} "
            f"p95={metrics.p95_latency_ms:.1f} ms failures={metrics.failed_queries}"
        )
        experiment_outputs.append(
            {
                "config_name": name,
                "base_config": base_name,
                "chunking": chunking,
                "reranker": reranker.name,
                "num_chunks": num_chunks,
                "index_latency_ms": index_latency_ms,
                "metrics": asdict(metrics),
                "queries": [_serialize_query(result) for result in query_results],
            }
        )

    if not keep_index:
        pipeline.cleanup()
    return experiment_outputs


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if Path(args.documents).resolve() == RAW_DOCUMENTS_PATH.resolve():
        snapshot_errors = check_snapshot()
        if snapshot_errors:
            details = "\n".join(f"  - {error}" for error in snapshot_errors)
            raise ValueError(
                f"Evaluation corpus snapshot is stale:\n{details}\n"
                "Run: python -m rag_eval.prepare_data --sync"
            )
    dataset = load_eval_dataset(Path(args.dataset))
    documents = load_raw_documents(Path(args.documents))
    validation_errors = validate_dataset(dataset, documents)
    if validation_errors:
        details = "\n".join(f"  - {error}" for error in validation_errors)
        raise ValueError(f"Golden dataset validation failed:\n{details}")
    print(f"Validated {len(dataset.examples)} grounded evaluation questions")
    if args.validate_only:
        return {"dataset": dataset_summary(dataset), "validation": "passed"}

    configs = select_chunking_configs(args.config)
    examples = dataset.examples[: args.limit] if args.limit else dataset.examples
    rerankers: list[Reranker] = []
    if args.reranker in {"off", "both"}:
        rerankers.append(NoOpReranker())
    reranker_load_ms = 0.0
    if args.reranker in {"on", "both"}:
        cross_encoder = CrossEncoderReranker(args.reranker_model)
        print(f"Loading reranker {args.reranker_model}...")
        reranker_load_ms = cross_encoder.warmup()
        rerankers.append(cross_encoder)

    started = datetime.now(timezone.utc)
    experiments = []
    failed_experiments = []
    for chunking in configs:
        try:
            experiments.extend(
                run_chunking_experiment(
                    chunking,
                    documents,
                    examples,
                    backend=args.backend,
                    embedding_provider=args.embedding_provider,
                    embedding_model=args.embedding_model,
                    retrieval_k=args.retrieval_k,
                    output_k=args.output_k,
                    rerankers=rerankers,
                    keep_index=args.keep_index,
                    fail_fast=args.fail_fast,
                )
            )
        except Exception as exc:
            if args.fail_fast:
                raise
            failed_experiments.append(
                {
                    "base_config": config_name(chunking),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[{config_name(chunking)}] FAILED: {exc}", file=sys.stderr)

    finished = datetime.now(timezone.utc)
    document_files = sorted(path for path in Path(args.documents).iterdir() if path.is_file())
    all_config_names = {config_name(item) for item in CHUNKING_CONFIGS}
    selected_config_names = {config_name(item) for item in configs}
    failed_queries = sum(item["metrics"]["failed_queries"] for item in experiments)
    claim_reasons = []
    if args.embedding_provider == "hash":
        claim_reasons.append("hash embeddings are smoke-test only")
    if selected_config_names != all_config_names:
        claim_reasons.append("not all registered chunking configurations were evaluated")
    if len(examples) != len(dataset.examples):
        claim_reasons.append("only a subset of the golden queries was evaluated")
    if args.reranker != "both":
        claim_reasons.append("both baseline and reranker arms were not evaluated")
    if failed_queries or failed_experiments:
        claim_reasons.append("the run contains failed queries or configurations")
    result = {
        "schema_version": 2,
        "run": {
            "run_id": str(uuid.uuid4()),
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "vector_backend": args.backend,
            "embedding_provider": args.embedding_provider,
            "embedding_model": args.embedding_model,
            "reranker_model": args.reranker_model if args.reranker != "off" else None,
            "reranker_load_latency_ms": reranker_load_ms,
            "retrieval_k": args.retrieval_k,
            "output_k": args.output_k,
            "quality_claim_eligible": not claim_reasons,
            "quality_claim_ineligibility_reasons": claim_reasons,
            "document_hashes": {path.name: _file_sha256(path) for path in document_files},
        },
        "dataset": {**dataset_summary(dataset), "evaluated_queries": len(examples)},
        "experiments": experiments,
        "failed_experiments": failed_experiments,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(f"\nSaved {len(experiments)} experiment results to {output}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG retrieval evaluation harness")
    parser.add_argument("--dataset", default=str(EVAL_DATASET_PATH))
    parser.add_argument("--documents", default=str(RAW_DOCUMENTS_PATH))
    parser.add_argument("--output", default=str(RESULTS_PATH))
    parser.add_argument("--config", action="append", help="Repeat to select named configs")
    parser.add_argument("--backend", choices=["chroma", "memory"], default="chroma")
    parser.add_argument(
        "--embedding-provider",
        choices=["google", "sentence-transformers", "hash"],
        default="google",
    )
    parser.add_argument("--embedding-model", default="gemini-embedding-001")
    parser.add_argument("--reranker", choices=["both", "on", "off"], default="both")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--retrieval-k", type=int, default=20)
    parser.add_argument("--output-k", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--keep-index", action="store_true")
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--quick", action="store_true", help="Credential-free 10-query smoke run"
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.quick:
        args.backend = "memory"
        args.embedding_provider = "hash"
        args.embedding_model = "signed-feature-hash-768"
        args.reranker = "off"
        args.config = ["recursive_512_100"]
        args.limit = args.limit or 10
        if args.output == str(RESULTS_PATH):
            args.output = str(ROOT / "data" / "benchmark_results_smoke.json")
    if args.output_k > args.retrieval_k:
        parser.error("--output-k cannot exceed --retrieval-k")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    try:
        run_benchmark(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
