"""Isolated vector indexes and embedding providers for RAG experiments.

Chroma + Google mirrors production. Memory + deterministic hash embeddings is
a credential-free smoke-test path and must not be used for headline quality claims.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Protocol, Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

EVAL_CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_eval"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"


class VectorIndex(Protocol):
    def similarity_search(self, query: str, k: int) -> list[Document]: ...


class HashingEmbeddings(Embeddings):
    """Stable signed feature hashing for local tests (not a semantic model)."""

    def __init__(self, dimensions: int = 768):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9]+", text.casefold()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            vector[value % self.dimensions] += -1.0 if value & 1 else 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class SentenceTransformerEmbeddings(Embeddings):
    """Local semantic embeddings; document text never leaves the machine."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "Local embeddings require sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load_model().encode(texts, normalize_embeddings=True)
        return [[float(value) for value in vector] for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._load_model().encode(text, normalize_embeddings=True)
        return [float(value) for value in vector]


class MemoryVectorIndex:
    """Exact cosine-search index for deterministic local smoke tests."""

    def __init__(self, chunks: Sequence[Document], embeddings: Embeddings):
        self._chunks = list(chunks)
        self._embeddings = embeddings
        self._vectors = embeddings.embed_documents([doc.page_content for doc in self._chunks])

    def similarity_search(self, query: str, k: int) -> list[Document]:
        if k <= 0:
            return []
        query_vector = self._embeddings.embed_query(query)
        scored = []
        for index, (document, vector) in enumerate(zip(self._chunks, self._vectors)):
            score = sum(a * b for a, b in zip(query_vector, vector))
            scored.append((score, index, document))
        scored.sort(key=lambda item: (-item[0], item[1]))
        output = []
        for rank, (score, _, document) in enumerate(scored[:k], start=1):
            metadata = {**document.metadata, "retrieval_score": float(score), "retrieval_rank": rank}
            output.append(Document(page_content=document.page_content, metadata=metadata))
        return output


class ChromaVectorIndex:
    """Adapter that retains Chroma relevance scores in result metadata."""

    def __init__(self, store: object):
        self._store = store

    def similarity_search(self, query: str, k: int) -> list[Document]:
        # Chroma exposes raw distances without enforcing a relevance-score
        # transform. Convert distance monotonically for diagnostics while
        # preserving Chroma's native ranking and avoiding out-of-range warnings.
        pairs = self._store.similarity_search_with_score(query, k=k)
        output = []
        for rank, (document, distance) in enumerate(pairs, start=1):
            score = 1.0 / (1.0 + max(0.0, float(distance)))
            output.append(
                Document(
                    page_content=document.page_content,
                    metadata={
                        **document.metadata,
                        "retrieval_score": score,
                        "retrieval_distance": float(distance),
                        "retrieval_rank": rank,
                    },
                )
            )
        return output


_embedding_cache: dict[tuple[str, str, int], Embeddings] = {}


def get_embeddings(
    provider: str = "google",
    model_name: str | None = None,
    hash_dimensions: int = 768,
) -> Embeddings:
    """Construct or reuse an embedding provider without eager optional imports."""
    model_name = model_name or DEFAULT_EMBEDDING_MODEL
    key = (provider, model_name, hash_dimensions)
    if key in _embedding_cache:
        return _embedding_cache[key]
    if provider == "hash":
        embeddings: Embeddings = HashingEmbeddings(hash_dimensions)
    elif provider == "sentence-transformers":
        embeddings = SentenceTransformerEmbeddings(model_name)
    elif provider == "google":
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY is required for Google embeddings")
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install langchain-google-genai for Google embeddings") from exc
        embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
    else:
        raise ValueError(f"Unknown embedding provider: {provider!r}")
    _embedding_cache[key] = embeddings
    return embeddings


def _collection_name(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", f"eval_{name}")[:63]
    return sanitized if len(sanitized) >= 3 else f"eval_{sanitized}"


def _chroma_safe_document(document: Document) -> Document:
    """Normalize loader metadata to Chroma's scalar-only value contract."""
    metadata = {}
    for key, value in document.metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[str(key)] = value
        else:
            metadata[str(key)] = json.dumps(value, sort_keys=True, default=str)
    return Document(page_content=document.page_content, metadata=metadata)


def build_index(
    chunks: Sequence[Document],
    config_name: str,
    embeddings: Embeddings,
    backend: str = "chroma",
) -> VectorIndex:
    """Build a completely isolated index for one chunking configuration."""
    if not chunks:
        raise ValueError("Cannot build an index with no chunks")
    if backend == "memory":
        return MemoryVectorIndex(chunks, embeddings)
    if backend != "chroma":
        raise ValueError(f"Unknown vector backend: {backend!r}")
    try:
        from langchain_chroma import Chroma
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Chroma requires chromadb and langchain-chroma; install rag_eval/requirements.txt"
        ) from exc

    collection = _collection_name(config_name)
    persist_dir = EVAL_CHROMA_DIR / collection
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    store = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    safe_chunks = [_chroma_safe_document(document) for document in chunks]
    store.add_documents(
        safe_chunks, ids=[str(document.metadata["chunk_id"]) for document in safe_chunks]
    )
    return ChromaVectorIndex(store)


def cleanup(config_name: str, backend: str = "chroma") -> None:
    if backend != "chroma":
        return
    persist_dir = EVAL_CHROMA_DIR / _collection_name(config_name)
    if persist_dir.exists():
        shutil.rmtree(persist_dir)


def cleanup_all() -> None:
    if EVAL_CHROMA_DIR.exists():
        shutil.rmtree(EVAL_CHROMA_DIR)
