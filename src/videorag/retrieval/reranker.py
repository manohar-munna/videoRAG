"""
reranker.py
-----------
Re-ranking utilities for CCTV RAG results.  Provides a cross-encoder
reranker (``sentence-transformers``) and a lightweight score-based
passthrough reranker.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Re-ranks retrieval results using a cross-encoder relevance model.

    The cross-encoder scores each (query, passage) pair jointly, producing
    more accurate relevance estimates than bi-encoder retrieval alone.

    Args:
        model_name: HuggingFace model identifier for the cross-encoder.
            Defaults to ``'cross-encoder/ms-marco-MiniLM-L-6-v2'``.
    """

    def __init__(self, model_name: str = _CROSS_ENCODER_MODEL) -> None:
        from sentence_transformers import CrossEncoder  # lazy import

        self.model_name = model_name
        logger.info("Loading cross-encoder model '%s'", model_name)
        self._model = CrossEncoder(model_name)
        logger.info("Cross-encoder loaded successfully")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        results: List[dict],
        top_k: int = 5,
    ) -> List[dict]:
        """Re-rank *results* by cross-encoder score.

        Each result dict must contain a ``text`` key (or a ``metadata``
        dict with a ``description`` key as fallback).

        Args:
            query: The original natural-language query.
            results: List of retrieval result dicts (e.g. from
                :class:`~videorag.retrieval.retriever.CCTVRetriever`).
            top_k: Maximum number of results to return.

        Returns:
            A new list of result dicts sorted by ``rerank_score`` in
            descending order, with the ``rerank_score`` key added.
        """
        if not results:
            return []

        passages: List[str] = []
        for r in results:
            text = (
                r.get("text")
                or r.get("description")
                or r.get("metadata", {}).get("text")
                or r.get("metadata", {}).get("description", "")
            )
            # If this is an expanded episode, aggregate descriptions across all storyboard frames
            if r.get("frames"):
                frame_descs = [
                    f.get("description") or f.get("text") or ""
                    for f in r.get("frames", [])
                    if f.get("description") or f.get("text")
                ]
                if frame_descs:
                    extra = " | ".join(frame_descs)
                    text = f"{text} | {extra}" if text else extra

            passages.append(str(text or ""))

        pairs = [(query, passage) for passage in passages]
        logger.info(
            "Cross-encoder scoring %d pairs for query='%s'",
            len(pairs),
            query[:80],
        )
        scores = self._model.predict(pairs)

        reranked = []
        for result, score in zip(results, scores):
            enriched = dict(result)
            enriched["rerank_score"] = float(score)
            reranked.append(enriched)

        reranked.sort(key=lambda r: r["rerank_score"], reverse=True)
        reranked = reranked[:top_k]

        logger.info(
            "Reranked to top-%d; best score=%.4f",
            len(reranked),
            reranked[0]["rerank_score"] if reranked else float("nan"),
        )
        return reranked


class ScoreReranker:
    """Lightweight passthrough reranker that sorts by existing retrieval score.

    No external model is required.  Results are simply sorted by their
    ``score`` field (descending) and truncated to *top_k*.
    """

    def rerank(
        self,
        query: str,
        results: List[dict],
        top_k: int = 5,
    ) -> List[dict]:
        """Sort *results* by descending ``score`` and return the top *top_k*.

        Args:
            query: Not used directly but kept for API consistency.
            results: List of retrieval result dicts.
            top_k: Maximum number of results to return.

        Returns:
            Top-*top_k* results sorted by ``score`` descending.
        """
        if not results:
            return []

        sorted_results = sorted(
            results, key=lambda r: r.get("score", 0.0), reverse=True
        )
        top = sorted_results[:top_k]

        logger.info(
            "ScoreReranker: returning top-%d from %d results",
            len(top),
            len(results),
        )
        return top
