"""
retriever.py
------------
High-level retrieval interface over a FAISSVectorStore + TextEmbedder,
with optional camera filtering and context-window expansion.
"""

import logging
from typing import List, Optional

from videorag.indexing.embedder import TextEmbedder
from videorag.indexing.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class CCTVRetriever:
    """Retrieves relevant CCTV chunks for a natural-language query.

    Args:
        vector_store: A populated :class:`~videorag.indexing.vector_store.FAISSVectorStore`.
        embedder: A :class:`~videorag.indexing.embedder.TextEmbedder` instance
            compatible with the embeddings stored in *vector_store*.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: TextEmbedder,
    ) -> None:
        self._store = vector_store
        self._embedder = embedder

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        camera_filter: Optional[str] = None,
    ) -> List[dict]:
        """Retrieve the top-k chunks most relevant to *query*.

        Args:
            query: Natural-language question or search string.
            top_k: Maximum number of results to return after filtering.
            camera_filter: If provided, only results whose metadata
                ``camera`` field matches this string are returned.

        Returns:
            A list of result dicts from the vector store, each containing
            ``score`` and ``metadata`` keys.  Results are sorted by
            descending cosine similarity.
        """
        logger.info(
            "Retrieving top_k=%d for query='%s' camera_filter=%s",
            top_k,
            query[:80],
            camera_filter,
        )

        query_embedding = self._embedder.embed_query(query)

        # Over-fetch so we still have top_k results after filtering.
        fetch_k = top_k * 3 if camera_filter else top_k
        raw_results = self._store.search(query_embedding, top_k=fetch_k)

        if camera_filter:
            raw_results = [
                r
                for r in raw_results
                if r["metadata"].get("camera") == camera_filter
            ]
            logger.debug(
                "After camera filter '%s': %d results remain",
                camera_filter,
                len(raw_results),
            )

        results = raw_results[:top_k]
        logger.info("Returning %d results", len(results))
        return results

    # ------------------------------------------------------------------
    # Context-window retrieval
    # ------------------------------------------------------------------

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5,
        context_window: int = 1,
    ) -> List[dict]:
        """Retrieve chunks and expand each hit with neighbouring timestamps.

        For each retrieved chunk, the method fetches additional chunks from
        the same camera whose ``start_timestamp`` is immediately before or
        after (within *context_window* positions in the sorted order of all
        stored metadata entries).

        Args:
            query: Natural-language query.
            top_k: Number of primary results to retrieve.
            context_window: Number of neighbouring entries (on each side)
                to include for each primary hit.

        Returns:
            Deduplicated list of result dicts (primary hits + neighbours),
            preserving primary hit ordering and appending neighbours.
        """
        primary_results = self.retrieve(query, top_k=top_k)

        # Build a sorted per-camera index from all stored metadata.
        all_meta: List[dict] = self._store._metadata  # type: ignore[attr-defined]
        camera_index: dict[str, List[dict]] = {}
        for entry in all_meta:
            cam = entry.get("camera", "")
            camera_index.setdefault(cam, []).append(entry)
        for cam in camera_index:
            camera_index[cam].sort(key=lambda m: m.get("start_timestamp", ""))

        seen_ids: set = set()
        expanded: List[dict] = []

        def _add(result: dict) -> None:
            key = (
                result["metadata"].get("camera"),
                result["metadata"].get("start_timestamp"),
            )
            if key not in seen_ids:
                seen_ids.add(key)
                expanded.append(result)

        for result in primary_results:
            _add(result)
            cam = result["metadata"].get("camera", "")
            ts = result["metadata"].get("start_timestamp", "")

            cam_entries = camera_index.get(cam, [])
            positions = [
                i for i, m in enumerate(cam_entries)
                if m.get("start_timestamp") == ts
            ]
            if not positions:
                continue

            pos = positions[0]
            for offset in range(-context_window, context_window + 1):
                if offset == 0:
                    continue
                neighbour_pos = pos + offset
                if 0 <= neighbour_pos < len(cam_entries):
                    neighbour_meta = cam_entries[neighbour_pos]
                    _add(
                        {
                            "score": 0.0,  # context neighbour, no direct score
                            "metadata": neighbour_meta,
                        }
                    )

        logger.info(
            "retrieve_with_context: %d primary → %d total (context_window=%d)",
            len(primary_results),
            len(expanded),
            context_window,
        )
        return expanded
