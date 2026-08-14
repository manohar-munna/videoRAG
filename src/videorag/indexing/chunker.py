"""
chunker.py
----------
Creates sliding-window temporal chunks from CCTV documents grouped by
camera, or treats each document as its own individual chunk.
"""

import logging
from itertools import groupby
from typing import Any

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Chunks CCTV documents for indexing.

    Two chunking strategies are provided:

    * **Sliding-window** (``chunk``): groups documents by camera, sorts
      them by timestamp, then applies a sliding window of configurable
      size and overlap.
    * **Individual** (``chunk_individual``): each document becomes its
      own chunk with no windowing.
    """

    # ------------------------------------------------------------------
    # Sliding-window chunking
    # ------------------------------------------------------------------

    def chunk(
        self,
        documents: list[dict],
        window_size: int = 3,
        overlap: int = 1,
    ) -> list[dict]:
        """Create sliding-window chunks grouped by camera.

        Documents within each camera group are sorted lexicographically
        by their ``timestamp`` field before windowing is applied.

        Args:
            documents: List of document dicts produced by
                :class:`~videorag.ingestion.loader.CCTVDataLoader`.
            window_size: Number of documents per chunk window.
            overlap: Number of documents that consecutive windows share.

        Returns:
            A flat list of chunk dicts.  Each dict contains:

            ``chunk_id`` (str), ``camera`` (str),
            ``start_timestamp`` (str), ``end_timestamp`` (str),
            ``text`` (str), ``doc_ids`` (list[int]),
            ``metadata`` (dict).
        """
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if overlap < 0 or overlap >= window_size:
            raise ValueError("overlap must be >= 0 and < window_size")

        step = window_size - overlap

        # Group by camera
        by_camera: dict[str, list[dict]] = {}
        for doc in documents:
            by_camera.setdefault(doc["camera"], []).append(doc)

        chunks: list[dict] = []
        chunk_counter = 0

        for camera, cam_docs in sorted(by_camera.items()):
            sorted_docs = sorted(cam_docs, key=lambda d: d["timestamp"])
            n = len(sorted_docs)

            start = 0
            while start < n:
                window = sorted_docs[start : start + window_size]
                if not window:
                    break

                text = "\n".join(d["text"] for d in window)
                doc_ids = [d["id"] for d in window]
                start_ts = window[0]["timestamp"]
                end_ts = window[-1]["timestamp"]

                chunk: dict[str, Any] = {
                    "chunk_id": f"chunk_{chunk_counter:05d}",
                    "camera": camera,
                    "start_timestamp": start_ts,
                    "end_timestamp": end_ts,
                    "text": text,
                    "doc_ids": doc_ids,
                    "metadata": {
                        "camera": camera,
                        "start_timestamp": start_ts,
                        "end_timestamp": end_ts,
                        "doc_ids": doc_ids,
                    },
                }
                chunks.append(chunk)
                chunk_counter += 1
                start += step

        logger.info(
            "Created %d sliding-window chunks (window=%d, overlap=%d)",
            len(chunks),
            window_size,
            overlap,
        )
        return chunks

    # ------------------------------------------------------------------
    # Individual chunking
    # ------------------------------------------------------------------

    def chunk_individual(self, documents: list[dict]) -> list[dict]:
        """Treat each document as its own chunk.

        Args:
            documents: List of document dicts.

        Returns:
            A list of chunk dicts — one per document — with the same
            schema as :meth:`chunk`.
        """
        chunks: list[dict] = []
        for doc in documents:
            camera: str = doc["camera"]
            ts: str = doc["timestamp"]
            chunk: dict[str, Any] = {
                "chunk_id": f"chunk_{doc['id']:05d}",
                "camera": camera,
                "start_timestamp": ts,
                "end_timestamp": ts,
                "text": doc["text"],
                "doc_ids": [doc["id"]],
                "metadata": {
                    "camera": camera,
                    "start_timestamp": ts,
                    "end_timestamp": ts,
                    "doc_ids": [doc["id"]],
                    "description": doc.get("description", ""),
                    "image_path": doc.get("metadata", {}).get("image_path", ""),
                },
            }
            chunks.append(chunk)

        logger.info("Created %d individual chunks", len(chunks))
        return chunks
