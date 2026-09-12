"""Generate an auditable Markdown report and optionally update README markers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "data" / "benchmark_results.json"
REPORT_PATH = ROOT / "data" / "benchmark_report.md"
README_PATH = ROOT.parent / "README.md"
README_START = "<!-- RAG_EVAL_RESULTS_START -->"
README_END = "<!-- RAG_EVAL_RESULTS_END -->"


def load_results(path: Path = RESULTS_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError("Expected benchmark result schema_version=2")
    if not payload.get("experiments"):
        raise ValueError("Benchmark contains no completed experiments")
    return payload


def _metrics(experiment: dict[str, Any]) -> dict[str, Any]:
    return experiment["metrics"]


def _ranked_experiments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Rank quality first, using latency only after retrieval outcomes tie."""
    return sorted(
        payload["experiments"],
        key=lambda item: (
            -_metrics(item)["mrr_at_5"],
            -_metrics(item)["recall_at_5"],
            -_metrics(item)["hit_rate_at_5"],
            _metrics(item)["p95_latency_ms"],
        ),
    )


def generate_comparison_table(
    payload: dict[str, Any], *, limit: int | None = None, heading: str = "Retrieval benchmark ranking"
) -> str:
    experiments = _ranked_experiments(payload)
    if limit is not None:
        experiments = experiments[:limit]
    lines = [
        f"## {heading}",
        "",
        "Ranked by MRR@5, then Evidence Recall@5, Hit@5, and p95 latency.",
        "",
        "| Rank | Configuration | Chunks | Reranker | Hit@1 | Hit@3 | Hit@5 | Evidence Recall@5 | Full Recall@5 | MRR@5 | nDCG@5 | p95 latency |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, experiment in enumerate(experiments, start=1):
        metric = _metrics(experiment)
        reranker = "None" if experiment["reranker"] == "none" else experiment["reranker"]
        lines.append(
            f"| {rank} | `{experiment['config_name']}` | {experiment['num_chunks']} | {reranker} | "
            f"{metric['hit_rate_at_1']:.3f} | {metric['hit_rate_at_3']:.3f} | "
            f"{metric['hit_rate_at_5']:.3f} | {metric['recall_at_5']:.3f} | "
            f"{metric['full_recall_at_5']:.3f} | {metric['mrr_at_5']:.3f} | "
            f"{metric['ndcg_at_5']:.3f} | {metric['p95_latency_ms']:.1f} ms |"
        )
    return "\n".join(lines)


def _paired_deltas(payload: dict[str, Any]) -> list[dict[str, float]]:
    experiments = payload["experiments"]
    baselines = {
        item["base_config"]: item
        for item in experiments
        if item["reranker"] == "none"
    }
    reranked = {
        item["base_config"]: item
        for item in experiments
        if item["reranker"] != "none"
    }
    output = []
    for name in sorted(set(baselines) & set(reranked)):
        before = _metrics(baselines[name])
        after = _metrics(reranked[name])
        output.append(
            {
                "hit3": after["hit_rate_at_3"] - before["hit_rate_at_3"],
                "mrr5": after["mrr_at_5"] - before["mrr_at_5"],
                "recall5": after["recall_at_5"] - before["recall_at_5"],
                "latency": after["p95_latency_ms"] - before["p95_latency_ms"],
            }
        )
    return output


def generate_analysis(payload: dict[str, Any]) -> str:
    experiments = payload["experiments"]
    best = max(experiments, key=lambda item: _metrics(item)["mrr_at_5"])
    fastest = min(experiments, key=lambda item: _metrics(item)["p95_latency_ms"])
    best_metric = _metrics(best)
    lines = [
        "## Analysis",
        "",
        f"- Best rank quality: `{best['config_name']}` with MRR@5 "
        f"**{best_metric['mrr_at_5']:.3f}** and Hit@5 **{best_metric['hit_rate_at_5']:.3f}**.",
        f"- Lowest p95 query latency: `{fastest['config_name']}` at "
        f"**{_metrics(fastest)['p95_latency_ms']:.1f} ms**.",
    ]
    deltas = _paired_deltas(payload)
    if deltas:
        lines.append(
            "- Across paired candidate sets, reranking changed Hit@3 by "
            f"**{mean(item['hit3'] for item in deltas) * 100:+.1f} points**, "
            f"MRR@5 by **{mean(item['mrr5'] for item in deltas):+.3f}**, and p95 "
            f"latency by **{mean(item['latency'] for item in deltas):+.1f} ms** on average."
        )
    failed = sum(_metrics(item)["failed_queries"] for item in experiments)
    lines.append(f"- Failed query/config evaluations: **{failed}**.")
    return "\n".join(lines)


def generate_key_findings(payload: dict[str, Any]) -> str:
    """Summarize measured engineering outcomes without promotional language."""
    experiments = payload["experiments"]
    best = _ranked_experiments(payload)[0]
    metric = _metrics(best)
    dataset = payload["dataset"]
    strategies = {item["chunking"]["strategy"] for item in experiments}
    lines = [
        "## Key findings",
        "",
        f"- Evaluated **{len(experiments)} retrieval configurations** across "
        f"**{len(strategies)} chunking strategies** on **{dataset['evaluated_queries']} "
        f"grounded questions** from **{dataset['num_sources']} documents**.",
        f"- Best-ranked configuration: `{best['config_name']}`, achieving "
        f"**{metric['hit_rate_at_5'] * 100:.1f}% "
        f"Hit@5**, **{metric['mrr_at_5']:.3f} MRR@5**, and "
        f"**{metric['recall_at_5'] * 100:.1f}% evidence Recall@5**.",
    ]
    deltas = _paired_deltas(payload)
    if deltas:
        lines.append(
            "- Mean cross-encoder impact on identical first-stage candidates: "
            f"**{mean(item['hit3'] for item in deltas) * 100:+.1f} percentage points Hit@3** "
            f"and **{mean(item['mrr5'] for item in deltas):+.3f} MRR@5**, at "
            f"**{mean(item['latency'] for item in deltas):+.1f} ms p95 latency**."
        )
    return "\n".join(lines)


def generate_methodology(payload: dict[str, Any]) -> str:
    run = payload["run"]
    dataset = payload["dataset"]
    return "\n".join(
        [
            "## Reproducibility and metric definitions",
            "",
            f"- Run ID: `{run['run_id']}`; dataset SHA-256: `{dataset['sha256']}`.",
            f"- Backend: `{run['vector_backend']}`; embeddings: "
            f"`{run['embedding_provider']}:{run['embedding_model']}`; candidate K: "
            f"{run['retrieval_k']}; evaluated K: {run['output_k']}.",
            "- Hit@K asks whether any gold passage is present. Evidence Recall@K counts "
            "distinct required passages, so duplicate overlapping chunks receive no extra credit.",
            "- MRR uses the first relevant rank; Full Recall@5 requires every passage for a "
            "compound question; latency reports end-to-end retrieval plus reranking.",
            "- The reranker sees the exact cached first-stage candidates used by its paired "
            "baseline; model-load time is recorded separately from query latency.",
        ]
    )


def generate_report(payload: dict[str, Any]) -> str:
    qualifier = (
        "Production-quality embedding run"
        if payload["run"].get("quality_claim_eligible")
        else "Partial/smoke run — not suitable for headline quality claims"
    )
    return "\n\n---\n\n".join(
        [
            f"# RAG evaluation benchmark report\n\n> {qualifier}",
            generate_comparison_table(payload),
            generate_analysis(payload),
            generate_methodology(payload),
            generate_key_findings(payload),
        ]
    ) + "\n"


def update_readme(readme_path: Path, payload: dict[str, Any]) -> None:
    if not payload["run"].get("quality_claim_eligible", False):
        raise ValueError("Refusing to publish smoke-test metrics to README")
    readme = readme_path.read_text(encoding="utf-8")
    start = readme.find(README_START)
    end = readme.find(README_END)
    if start < 0 or end < 0 or end < start:
        raise ValueError("README benchmark markers are missing or malformed")
    block = "\n\n".join(
        [
            generate_key_findings(payload),
            generate_comparison_table(
                payload, limit=5, heading="Top five configurations"
            ),
            "[View the complete 18-configuration report](rag_eval/data/benchmark_report.md).",
        ]
    )
    replacement = f"{README_START}\n{block}\n{README_END}"
    updated = readme[:start] + replacement + readme[end + len(README_END) :]
    temporary = readme_path.with_suffix(".md.tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(readme_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RAG benchmark report")
    parser.add_argument("--input", type=Path, default=RESULTS_PATH)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument("--readme", type=Path, default=README_PATH)
    args = parser.parse_args()
    payload = load_results(args.input)
    report = generate_report(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    if args.update_readme:
        update_readme(args.readme, payload)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
