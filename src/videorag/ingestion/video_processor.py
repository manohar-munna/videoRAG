"""
src/videorag/ingestion/video_processor.py
-------------------------------------------
Frame extraction, sampling, and smart perceptual hash filtering for CCTV video.

Uses OpenCV to sample frames at fixed or adaptive intervals (e.g. every 5s or 15s),
format precise timestamps (HH:MM:SS), and optionally apply EdgeFrameFilter (dHash/pHash)
to drop duplicate static frames before VLM captioning.
"""

import os
import cv2
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from videorag.ingestion.hash_filter import EdgeFrameFilter

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """Format total seconds into HH:MM:SS format."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


class VideoFrameExtractor:
    """Extracts timestamped frame images from video files with optional dHash/pHash filtering."""

    def __init__(self, output_dir: str = "data/extracted_frames") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_frames(
        self,
        video_path: str,
        camera_id: str = "CAM_01",
        sample_interval: float = 5.0,
        max_frames: int = 500,
        hash_filter: Optional[EdgeFrameFilter] = None,
    ) -> Dict[str, Any]:
        """
        Extract frames from *video_path* at every *sample_interval* seconds.

        If *hash_filter* (EdgeFrameFilter) is provided, evaluates dHash/pHash Hamming distance
        and skips static/duplicate frames.

        Returns:
            {
                "extracted_frames": [list of keyframe metadata dicts],
                "skipped_count": int,
                "total_sampled": int,
                "audit_trail": [full audit logs per sampled frame for UI dev verification],
                "filter_stats": dict
            }
        """
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path_obj))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Fallback

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps
        step_frames = int(fps * sample_interval)

        logger.info(
            "Extracting frames from '%s' (FPS: %.2f, Duration: %.1fs, Interval: %.1fs, Smart Filter: %s)",
            video_path_obj.name,
            fps,
            duration_sec,
            sample_interval,
            "ENABLED (" + hash_filter.method + ")" if hash_filter else "DISABLED",
        )

        extracted: List[Dict[str, Any]] = []
        audit_trail: List[Dict[str, Any]] = []
        frame_idx = 0
        extracted_count = 0
        total_sampled = 0
        skipped_count = 0

        if hash_filter:
            hash_filter.reset_stats()

        while cap.isOpened() and extracted_count < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            seconds = frame_idx / fps
            timestamp_str = format_timestamp(seconds)
            total_sampled += 1

            is_keyframe = True
            audit_item = None

            if hash_filter:
                audit_item = hash_filter.evaluate_frame(frame, frame_idx=frame_idx, timestamp=timestamp_str)
                is_keyframe = audit_item["is_keyframe"]
                audit_trail.append(audit_item)

            if is_keyframe:
                clean_ts = timestamp_str.replace(":", "_")
                out_filename = f"{camera_id}_{clean_ts}_{frame_idx}.jpg"
                out_path = self.output_dir / out_filename

                cv2.imwrite(str(out_path), frame)

                item = {
                    "camera": camera_id,
                    "timestamp": timestamp_str,
                    "seconds": round(seconds, 2),
                    "frame_idx": frame_idx,
                    "image_path": str(out_path),
                }
                if audit_item:
                    item["hash_hex"] = audit_item["hash_hex"]
                    item["hamming_distance"] = audit_item["hamming_distance"]
                    item["motion_pct"] = audit_item["motion_pct"]

                extracted.append(item)
                extracted_count += 1
            else:
                skipped_count += 1

            frame_idx += step_frames
            if frame_idx >= total_frames:
                break

        cap.release()
        logger.info(
            "Frame extraction complete: %d keyframes kept, %d frames skipped (%.1f%% LLM compute saved)",
            len(extracted),
            skipped_count,
            (skipped_count / total_sampled * 100.0) if total_sampled > 0 else 0.0,
        )

        filter_stats = hash_filter.get_summary() if hash_filter else {
            "total_frames": total_sampled,
            "keyframes_kept": len(extracted),
            "frames_skipped": skipped_count,
            "llm_compute_saved_pct": 0.0,
        }

        return {
            "extracted_frames": extracted,
            "skipped_count": skipped_count,
            "total_sampled": total_sampled,
            "audit_trail": audit_trail,
            "filter_stats": filter_stats,
        }
