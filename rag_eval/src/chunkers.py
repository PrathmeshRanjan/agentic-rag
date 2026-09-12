"""Deterministic, metadata-preserving chunking strategies."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _validate_window(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def _chunk_id(document: Document, strategy: str, index: int, start: int) -> str:
    source = str(document.metadata.get("source", "unknown"))
    page = str(document.metadata.get("page", ""))
    payload = f"{source}|{page}|{strategy}|{index}|{start}|{document.page_content}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _attach_metadata(
    original: Document,
    chunks: Sequence[Document],
    strategy: str,
    chunk_size: int | None,
    chunk_overlap: int | None,
) -> list[Document]:
    """Copy provenance and add stable metadata to chunks from one document."""
    output: list[Document] = []
    search_from = 0
    for index, chunk in enumerate(chunks):
        metadata = {**original.metadata, **chunk.metadata}
        start = metadata.get("start_index")
        if start is None:
            start = original.page_content.find(chunk.page_content, search_from)
            start = max(0, start)
        search_from = int(start) + 1
        metadata.update(
            {
                "chunk_strategy": strategy,
                "chunk_index": index,
                "start_index": int(start),
            }
        )
        if chunk_size is not None:
            metadata["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            metadata["chunk_overlap"] = chunk_overlap
        enriched = Document(page_content=chunk.page_content, metadata=metadata)
        enriched.metadata["chunk_id"] = _chunk_id(enriched, strategy, index, int(start))
        output.append(enriched)
    return output


def chunk_fixed(
    documents: Sequence[Document], chunk_size: int = 512, chunk_overlap: int = 100
) -> list[Document]:
    """Split text into true fixed-width character windows as a control."""
    _validate_window(chunk_size, chunk_overlap)
    step = chunk_size - chunk_overlap
    output: list[Document] = []
    for document in documents:
        chunks: list[Document] = []
        for start in range(0, len(document.page_content), step):
            text = document.page_content[start : start + chunk_size]
            if not text:
                break
            chunks.append(Document(page_content=text, metadata={"start_index": start}))
            if start + chunk_size >= len(document.page_content):
                break
        output.extend(_attach_metadata(document, chunks, "fixed", chunk_size, chunk_overlap))
    return output


def chunk_recursive(
    documents: Sequence[Document], chunk_size: int = 512, chunk_overlap: int = 100
) -> list[Document]:
    """Split each source page with structure-aware recursive separators."""
    _validate_window(chunk_size, chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    output: list[Document] = []
    for document in documents:
        chunks = splitter.split_documents([document])
        output.extend(_attach_metadata(document, chunks, "recursive", chunk_size, chunk_overlap))
    return output


def chunk_semantic(
    documents: Sequence[Document],
    embeddings: Any,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float | None = None,
    min_chunk_size: int | None = 100,
) -> list[Document]:
    """Split on embedding-similarity breakpoints using the indexed embedder."""
    if embeddings is None:
        raise ValueError("semantic chunking requires an embeddings implementation")
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Semantic chunking requires langchain-experimental; install the eval dependencies"
        ) from exc

    kwargs: dict[str, Any] = {
        "embeddings": embeddings,
        "breakpoint_threshold_type": breakpoint_threshold_type,
    }
    if breakpoint_threshold_amount is not None:
        kwargs["breakpoint_threshold_amount"] = breakpoint_threshold_amount
    if min_chunk_size is not None:
        kwargs["min_chunk_size"] = min_chunk_size
    splitter = SemanticChunker(**kwargs)

    output: list[Document] = []
    for document in documents:
        chunks = splitter.split_documents([document])
        output.extend(_attach_metadata(document, chunks, "semantic", None, None))
    return output


# Overlap changes while strategy and size are held constant, enabling clean
# one-variable-at-a-time comparisons.
CHUNKING_CONFIGS: list[dict[str, Any]] = [
    {"strategy": "fixed", "chunk_size": 256, "chunk_overlap": 0},
    {"strategy": "fixed", "chunk_size": 256, "chunk_overlap": 64},
    {"strategy": "fixed", "chunk_size": 512, "chunk_overlap": 0},
    {"strategy": "fixed", "chunk_size": 512, "chunk_overlap": 100},
    {"strategy": "recursive", "chunk_size": 512, "chunk_overlap": 0},
    {"strategy": "recursive", "chunk_size": 512, "chunk_overlap": 100},
    {"strategy": "recursive", "chunk_size": 1024, "chunk_overlap": 0},
    {"strategy": "recursive", "chunk_size": 1024, "chunk_overlap": 200},
    {
        "strategy": "semantic",
        "breakpoint_threshold_type": "percentile",
        "breakpoint_threshold_amount": 90,
    },
]


def apply_chunking(
    documents: Sequence[Document], config: Mapping[str, Any], embeddings: Any = None
) -> list[Document]:
    """Apply a validated chunking configuration."""
    strategy = str(config.get("strategy", ""))
    if strategy == "fixed":
        return chunk_fixed(
            documents,
            chunk_size=int(config.get("chunk_size", 512)),
            chunk_overlap=int(config.get("chunk_overlap", 100)),
        )
    if strategy == "recursive":
        return chunk_recursive(
            documents,
            chunk_size=int(config.get("chunk_size", 512)),
            chunk_overlap=int(config.get("chunk_overlap", 100)),
        )
    if strategy == "semantic":
        return chunk_semantic(
            documents,
            embeddings=embeddings,
            breakpoint_threshold_type=str(config.get("breakpoint_threshold_type", "percentile")),
            breakpoint_threshold_amount=config.get("breakpoint_threshold_amount"),
            min_chunk_size=config.get("min_chunk_size", 100),
        )
    raise ValueError(f"Unknown chunking strategy: {strategy!r}")


def config_name(config: Mapping[str, Any]) -> str:
    """Return a stable, human-readable identifier for a chunking config."""
    strategy = str(config["strategy"])
    if strategy == "semantic":
        return f"semantic_{config.get('breakpoint_threshold_amount', 'auto')}"
    return f"{strategy}_{config.get('chunk_size')}_{config.get('chunk_overlap')}"
