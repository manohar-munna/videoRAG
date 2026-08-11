"""
vector_store.py
---------------
FAISS-backed vector store with cosine similarity search, persistence,
and chunk metadata retrieval.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise *vectors* row-wise (in-place safe copy).

    Args:
        vectors: 2-D float32 array of shape ``(n, dim)``.

    Returns:
        A normalised copy.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype(np.float32)


class FAISSVectorStore:
    """In-memory FAISS index with JSON-serialisable metadata sidecar.

    Uses ``faiss.IndexFlatIP`` (inner product) on L2-normalised vectors
    so that the inner product equals cosine similarity.

    Args:
        dim: Embedding dimensionality.
        index_type: Reserved for future index variants; currently only
            ``'flat'`` (``IndexFlatIP``) is supported.
    """

    def __init__(self, dim: int, index_type: str = "flat") -> None:
        self.dim = dim
        self.index_type = index_type
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self._metadata: List[dict] = []
        logger.info(
            "Initialised FAISSVectorStore (dim=%d, type='%s')", dim, index_type
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(self, embeddings: np.ndarray, metadata: List[dict]) -> None:
        """Add vectors and their associated metadata to the store.

        Vectors are L2-normalised before insertion so that inner-product
        search yields cosine similarities in [−1, 1].

        Args:
            embeddings: Float32 array of shape ``(n, dim)``.
            metadata: List of *n* metadata dicts (one per vector).
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Expected embeddings of shape (n, {self.dim}), "
                f"got {embeddings.shape}"
            )
        if len(embeddings) != len(metadata):
            raise ValueError(
                "embeddings and metadata must have the same length"
            )

        normed = _l2_normalize(embeddings)
        self._index.add(normed)
        self._metadata.extend(metadata)
        logger.info(
            "Added %d vectors; store now contains %d vectors",
            len(embeddings),
            self._index.ntotal,
        )

    def search(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[dict]:
        """Search for the *top_k* most similar vectors.

        Args:
            query_embedding: 1-D float32 array of shape ``(dim,)`` **or**
                2-D array of shape ``(1, dim)``.
            top_k: Number of results to return.

        Returns:
            A list of dicts, each with keys:
            ``score`` (float cosine similarity) and
            ``metadata`` (the dict stored alongside the vector).
        """
        if self._index.ntotal == 0:
            logger.warning("Search called on empty index — returning []")
            return []

        q = query_embedding.reshape(1, -1).astype(np.float32)
        q = _l2_normalize(q)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)

        results: List[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                {
                    "score": float(score),
                    "metadata": self._metadata[idx],
                }
            )

        logger.debug("Search returned %d results (top_k=%d)", len(results), top_k)
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the FAISS index and metadata to *path*.

        Two files are written:

        * ``<path>.faiss`` — binary FAISS index.
        * ``<path>.meta.json`` — JSON array of metadata dicts.

        Args:
            path: Base path (without extension).
        """
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)

        index_path = str(base) + ".faiss"
        meta_path = str(base) + ".meta.json"

        faiss.write_index(self._index, index_path)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(self._metadata, fh, ensure_ascii=False, indent=2)

        logger.info(
            "Saved index (%d vectors) → '%s' | metadata → '%s'",
            self._index.ntotal,
            index_path,
            meta_path,
        )

    def load(self, path: str) -> None:
        """Load a previously saved index from *path*.

        Args:
            path: Base path used when calling :meth:`save` (no extension).

        Raises:
            FileNotFoundError: If either expected file is missing.
        """
        base = Path(path)
        index_path = str(base) + ".faiss"
        meta_path = str(base) + ".meta.json"

        for fp in (index_path, meta_path):
            if not Path(fp).exists():
                raise FileNotFoundError(f"Vector store file not found: {fp}")

        self._index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as fh:
            self._metadata = json.load(fh)

        logger.info(
            "Loaded index with %d vectors from '%s'",
            self._index.ntotal,
            index_path,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of vectors currently stored."""
        return self._index.ntotal
