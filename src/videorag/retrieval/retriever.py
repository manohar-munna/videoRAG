"""
retriever.py
------------
Multimodal retrieval interface over FAISSVectorStore + MultimodalEmbedder,
with chronological temporal context window expansion (Episode grouping).
"""

import logging
from typing import List, Optional, Dict, Any

from videorag.indexing.embedder import MultimodalEmbedder
from videorag.indexing.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


def _parse_ts_to_seconds(val: Any) -> float:
    """Parse 'HH:MM:SS', 'MM:SS', or numeric value into float seconds."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    try:
        parts = str(val).strip().split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60.0 + float(parts[1])
        return float(val)
    except (ValueError, TypeError):
        return 0.0


class CCTVRetriever:
    """Retrieves relevant CCTV keyframe visual moments and expands them into temporal episodes.

    Args:
        vector_store: Populated :class:`~videorag.indexing.vector_store.FAISSVectorStore`.
        embedder: Multimodal CLIP embedder :class:`~videorag.indexing.embedder.MultimodalEmbedder`.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: MultimodalEmbedder,
    ) -> None:
        self._store = vector_store
        self._embedder = embedder

    # ------------------------------------------------------------------
    # Core vector search
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        camera_filter: Optional[str] = None,
    ) -> List[dict]:
        """Retrieve the top-k anchor frames most visually/semantically similar to *query*.

        Args:
            query: Natural-language question or description.
            top_k: Maximum number of candidate hits to return.
            camera_filter: If provided, only results from this camera are returned.

        Returns:
            A list of result dicts sorted by descending similarity score.
        """
        logger.info(
            "Retrieving top_k=%d for query='%s' (camera_filter=%s)...",
            top_k,
            query[:80],
            camera_filter,
        )

        query_embedding = self._embedder.embed_query(query)

        # Over-fetch when filtering by camera to ensure top_k items
        fetch_k = top_k * 4 if camera_filter else top_k
        raw_results = self._store.search(query_embedding, top_k=fetch_k)

        if camera_filter:
            raw_results = [
                r
                for r in raw_results
                if r["metadata"].get("camera") == camera_filter
            ]

        results = raw_results[:top_k]
        logger.info("Retrieved %d anchor frame results", len(results))
        return results

    # ------------------------------------------------------------------
    # Temporal Context Window & Episode Expansion
    # ------------------------------------------------------------------

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 3,
        context_window: int = 2,
        camera_filter: Optional[str] = None,
    ) -> List[dict]:
        """Retrieve top candidate anchor frames and expand each into a chronological episode.

        For each matched anchor frame, this method retrieves the surrounding sequence
        of keyframes from the same camera feed (e.g. 2 frames before, the anchor frame,
        and 2 frames after) to give the downstream VLM full chronological context.

        Args:
            query: Natural-language search query.
            top_k: Number of primary anchor hits to retrieve.
            context_window: Number of neighbouring frames (on each side) to bundle.
            camera_filter: Optional camera ID to restrict retrieval.

        Returns:
            A list of Episode dictionaries with ordered storyboard frames.
        """
        primary_results = self.retrieve(query, top_k=top_k, camera_filter=camera_filter)
        if not primary_results:
            return []

        # Build chronologically sorted index per camera from store metadata
        with self._store._lock:
            all_meta: List[dict] = [dict(m) for m in self._store._metadata]

        camera_index: Dict[str, List[dict]] = {}
        for entry in all_meta:
            cam = entry.get("camera", "UNKNOWN")
            camera_index.setdefault(cam, []).append(entry)

        # Sort each camera's records chronologically by epoch_time -> seconds -> timestamp
        for cam, entries in camera_index.items():
            entries.sort(key=lambda m: (
                float(m.get("epoch_time") or 0.0),
                _parse_ts_to_seconds(m.get("seconds") or m.get("timestamp") or m.get("start_timestamp")),
                str(m.get("timestamp") or m.get("start_timestamp") or "")
            ))

        episodes: List[dict] = []
        seen_anchor_keys = set()

        for rank, anchor_result in enumerate(primary_results, start=1):
            anchor_meta = anchor_result.get("metadata", {})
            cam = anchor_meta.get("camera", "CAM_01")
            anchor_ts = anchor_meta.get("timestamp") or anchor_meta.get("start_timestamp", "")
            anchor_img = anchor_meta.get("image_path", "")
            anchor_score = float(anchor_result.get("score", 0.0))

            anchor_key = (cam, anchor_ts, anchor_img)
            if anchor_key in seen_anchor_keys:
                continue
            seen_anchor_keys.add(anchor_key)

            cam_entries = camera_index.get(cam, [])
            # Find the position of the anchor in the sorted camera timeline
            anchor_pos = -1
            for i, item in enumerate(cam_entries):
                item_ts = item.get("timestamp") or item.get("start_timestamp", "")
                if (anchor_img and item.get("image_path") == anchor_img) or (anchor_ts and item_ts == anchor_ts):
                    anchor_pos = i
                    break

            # Fallback if exact match not found
            if anchor_pos == -1:
                anchor_secs = _parse_ts_to_seconds(anchor_meta.get("seconds") or anchor_ts)
                storyboard_frames = [{
                    "image_path": anchor_img,
                    "timestamp": anchor_ts,
                    "seconds": anchor_secs,
                    "epoch_time": anchor_meta.get("epoch_time"),
                    "score": anchor_score,
                    "is_anchor": True,
                    "description": anchor_meta.get("description", ""),
                }]
                start_ts = anchor_ts
                end_ts = anchor_ts
            else:
                # Slice the temporal context window
                start_idx = max(0, anchor_pos - context_window)
                end_idx = min(len(cam_entries), anchor_pos + context_window + 1)
                window_slice = cam_entries[start_idx:end_idx]

                storyboard_frames = []
                for idx_in_cam, frame_meta in enumerate(window_slice, start=start_idx):
                    is_anchor = (idx_in_cam == anchor_pos)
                    frame_ts = frame_meta.get("timestamp") or frame_meta.get("start_timestamp", "00:00:00")
                    frame_secs = _parse_ts_to_seconds(frame_meta.get("seconds") or frame_ts)
                    storyboard_frames.append({
                        "image_path": frame_meta.get("image_path", ""),
                        "timestamp": frame_ts,
                        "seconds": frame_secs,
                        "epoch_time": frame_meta.get("epoch_time"),
                        "score": anchor_score if is_anchor else 0.0,
                        "is_anchor": is_anchor,
                        "description": frame_meta.get("description", ""),
                    })

                start_ts = storyboard_frames[0]["timestamp"] if storyboard_frames else anchor_ts
                end_ts = storyboard_frames[-1]["timestamp"] if storyboard_frames else anchor_ts

            time_range = f"{start_ts} → {end_ts}" if start_ts != end_ts else start_ts

            episode = {
                "rank": rank,
                "score": anchor_score,
                "camera": cam,
                "time_range": time_range,
                "anchor_timestamp": anchor_ts,
                "anchor_image": anchor_img,
                "frames": storyboard_frames,
                "frame_count": len(storyboard_frames),
                "metadata": anchor_meta,
                "text": f"Camera {cam} Episode ({time_range}): {len(storyboard_frames)} chronological frames.",
            }
            episodes.append(episode)

        logger.info(
            "retrieve_with_context: Produced %d chronological episodes (context_window=±%d frames)",
            len(episodes),
            context_window,
        )
        return episodes
