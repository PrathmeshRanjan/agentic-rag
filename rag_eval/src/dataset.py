"""Golden-dataset loading, schema checks, and source-grounding validation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from langchain_core.documents import Document

from rag_eval.src.metrics import GoldEvidence

SCHEMA_VERSION = 2
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}


@dataclass(frozen=True)
class EvalExample:
    query_id: str
    query: str
    reference_answer: str
    gold_evidence: list[GoldEvidence]
    difficulty: str
    tags: list[str]


@dataclass(frozen=True)
class EvalDataset:
    name: str
    schema_version: int
    description: str
    examples: list[EvalExample]
    source_path: Path

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.source_path.read_bytes()).hexdigest()


def load_eval_dataset(path: Path) -> EvalDataset:
    """Load schema v2 while failing loudly on incomplete or ambiguous labels."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected evaluation dataset schema_version={SCHEMA_VERSION}")
    raw_examples = raw.get("examples")
    if not isinstance(raw_examples, list):
        raise ValueError("Dataset 'examples' must be a list")

    examples = []
    for item in raw_examples:
        evidence = [GoldEvidence(**gold) for gold in item.get("gold_evidence", [])]
        examples.append(
            EvalExample(
                query_id=str(item["query_id"]),
                query=str(item["query"]),
                reference_answer=str(item["reference_answer"]),
                gold_evidence=evidence,
                difficulty=str(item["difficulty"]),
                tags=[str(tag) for tag in item.get("tags", [])],
            )
        )
    return EvalDataset(
        name=str(raw.get("name", path.stem)),
        schema_version=int(raw["schema_version"]),
        description=str(raw.get("description", "")),
        examples=examples,
        source_path=path,
    )


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def validate_dataset(
    dataset: EvalDataset,
    source_documents: Sequence[Document],
    *,
    minimum_examples: int = 20,
    maximum_examples: int = 50,
) -> list[str]:
    """Return all label and grounding errors rather than failing at the first one."""
    errors: list[str] = []
    if len(dataset.examples) < minimum_examples:
        errors.append(
            f"Dataset has {len(dataset.examples)} examples; minimum is {minimum_examples}"
        )
    if len(dataset.examples) > maximum_examples:
        errors.append(
            f"Dataset has {len(dataset.examples)} examples; maximum is {maximum_examples}"
        )
    ids = [example.query_id for example in dataset.examples]
    if len(ids) != len(set(ids)):
        errors.append("query_id values must be unique")

    evidence_ids: list[str] = []
    documents_by_source: dict[str, list[Document]] = {}
    for document in source_documents:
        source = Path(str(document.metadata.get("source", ""))).name.casefold()
        documents_by_source.setdefault(source, []).append(document)

    for example in dataset.examples:
        prefix = example.query_id
        if not example.query.strip() or not example.reference_answer.strip():
            errors.append(f"{prefix}: query and reference_answer are required")
        if example.difficulty not in ALLOWED_DIFFICULTIES:
            errors.append(f"{prefix}: invalid difficulty {example.difficulty!r}")
        if not example.gold_evidence:
            errors.append(f"{prefix}: at least one gold_evidence item is required")
        for evidence in example.gold_evidence:
            evidence_ids.append(evidence.evidence_id)
            source_key = Path(evidence.source).name.casefold()
            candidates = documents_by_source.get(source_key, [])
            if not candidates:
                errors.append(f"{prefix}/{evidence.evidence_id}: unknown source {evidence.source}")
                continue
            if evidence.page is not None:
                candidates = [
                    doc for doc in candidates if doc.metadata.get("page") == evidence.page
                ]
                if not candidates:
                    errors.append(
                        f"{prefix}/{evidence.evidence_id}: page {evidence.page} not found"
                    )
                    continue
            needle = _normalize(evidence.text)
            if not needle or not any(needle in _normalize(doc.page_content) for doc in candidates):
                errors.append(
                    f"{prefix}/{evidence.evidence_id}: quoted evidence is not verbatim in "
                    f"{evidence.source}" + (f" page {evidence.page}" if evidence.page else "")
                )

    if len(evidence_ids) != len(set(evidence_ids)):
        errors.append("evidence_id values must be globally unique")
    return errors


def dataset_summary(dataset: EvalDataset) -> dict[str, Any]:
    sources = {
        evidence.source
        for example in dataset.examples
        for evidence in example.gold_evidence
    }
    return {
        "name": dataset.name,
        "schema_version": dataset.schema_version,
        "sha256": dataset.sha256,
        "num_queries": len(dataset.examples),
        "num_sources": len(sources),
        "num_multi_evidence_queries": sum(
            len(example.gold_evidence) > 1 for example in dataset.examples
        ),
        "difficulty_counts": {
            difficulty: sum(example.difficulty == difficulty for example in dataset.examples)
            for difficulty in sorted(ALLOWED_DIFFICULTIES)
        },
    }
