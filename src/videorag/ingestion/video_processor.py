"""
src/videorag/ingestion/video_processor.py
-------------------------------------------
Frame extraction and sampling module for CCTV video footage.

Uses OpenCV to sample frames at fixed time intervals (e.g. every 5 or 10 seconds),
format precise timestamps (HH:MM:SS), and save frame images for VLM captioning.
"""

import os
import cv2
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """Format total seconds into HH:MM:SS format."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


class VideoFrameExtractor:
    """Extracts timestamped frame images from video files."""

    def __init__(self, output_dir: str = "data/extracted_frames") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_frames(
        self,
        video_path: str,
        camera_id: str = "CAM_01",
        sample_interval: float = 5.0,
        max_frames: int = 500,
    ) -> List[Dict[str, Any]]:
        """Extract frames from *video_path* at every *sample_interval* seconds.

        Args:
            video_path: Path to the MP4/video file.
            camera_id: Camera identifier (e.g., 'CAM_01').
            sample_interval: Sampling interval in seconds (default: 5.0s).
            max_frames: Maximum number of frames to extract (default: 500).

        Returns:
            List of metadata dicts for extracted frames:
            [
                {
                    "camera": "CAM_01",
                    "timestamp": "00:01:15",
                    "seconds": 75.0,
                    "frame_idx": 2250,
                    "image_path": "data/extracted_frames/CAM_01_00_01_15.jpg"
                }, ...
            ]
        """
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path_obj))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Fallback assumption

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps
        step_frames = int(fps * sample_interval)

        logger.info(
            "Extracting frames from '%s' (FPS: %.2f, Duration: %.1fs, Interval: %.1fs)",
            video_path_obj.name,
            fps,
            duration_sec,
            sample_interval,
        )

        extracted: List[Dict[str, Any]] = []
        frame_idx = 0
        extracted_count = 0

        while cap.isOpened() and extracted_count < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            seconds = frame_idx / fps
            timestamp_str = format_timestamp(seconds)
            clean_ts = timestamp_str.replace(":", "_")
            out_filename = f"{camera_id}_{clean_ts}_{frame_idx}.jpg"
            out_path = self.output_dir / out_filename

            cv2.imwrite(str(out_path), frame)

            extracted.append({
                "camera": camera_id,
                "timestamp": timestamp_str,
                "seconds": round(seconds, 2),
                "frame_idx": frame_idx,
                "image_path": str(out_path),
            })

            extracted_count += 1
            frame_idx += step_frames
            if frame_idx >= total_frames:
                break

        cap.release()
        logger.info("Successfully extracted %d frames to '%s'", len(extracted), self.output_dir)
        return extracted
