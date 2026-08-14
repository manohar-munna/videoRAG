"""
src/videorag/ingestion/stream_capture.py
-----------------------------------------
Async Multi-Threaded RTSP Stream Capture & Producer-Consumer Ring Buffer Queue.

Provides high-performance, non-blocking video stream ingestion for live CCTV network feeds (RTSP/RTMP)
and video files. Features:
- Dedicated thread-based frame producer reading into a size-1 Ring Buffer (zero latency).
- Automatic RTSP socket watchdog with exponential backoff reconnection.
- Integrated EdgeFrameFilter (dHash/pHash) keyframe extraction.
- Multi-camera stream manager for concurrent surveillance feed monitoring.
"""

import os
import cv2
import time
import queue
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional

from videorag.ingestion.hash_filter import EdgeFrameFilter

logger = logging.getLogger("videorag.ingestion.stream_capture")


class RTSPStreamCapture:
    """
    Thread-safe, non-blocking RTSP/RTMP/File stream reader with Ring Buffer & Watchdog.
    """

    def __init__(
        self,
        camera_id: str,
        stream_url: str,
        output_dir: str = "data/extracted_frames",
        sample_interval: float = 5.0,
        hash_filter: Optional[EdgeFrameFilter] = None,
        max_reconnect_attempts: int = 10,
    ):
        """
        :param camera_id: Unique camera identifier (e.g. 'CAM_01')
        :param stream_url: RTSP/RTMP stream URL (e.g. 'rtsp://192.168.1.100/live') or file path
        :param output_dir: Directory to save extracted keyframe images
        :param sample_interval: Seconds between keyframe evaluations (default: 5.0s)
        :param hash_filter: EdgeFrameFilter (dHash/pHash) instance
        """
        self.camera_id = camera_id
        self.stream_url = str(stream_url)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sample_interval = sample_interval
        self.hash_filter = hash_filter or EdgeFrameFilter(method="dhash", threshold=10)
        self.max_reconnect_attempts = max_reconnect_attempts

        # Threading state
        self._running = False
        self._producer_thread: Optional[threading.Thread] = None
        self._consumer_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Ring Buffer Queue of size 1 (Always drops old unread frames to guarantee 0-latency live view)
        self._ring_buffer = queue.Queue(maxsize=1)

        # Connection & Metric Counters
        self.is_connected = False
        self.fps = 0.0
        self.total_frames_read = 0
        self.total_frames_dropped = 0
        self.reconnect_count = 0
        
        self.extracted_keyframes: List[Dict[str, Any]] = []
        self.audit_trail: List[Dict[str, Any]] = []

    def start(self) -> None:
        """Start producer and consumer background worker threads."""
        with self._lock:
            if self._running:
                logger.warning("[%s] Stream capture already running.", self.camera_id)
                return

            self._running = True
            self._producer_thread = threading.Thread(
                target=self._producer_loop, name=f"RTSP-Producer-{self.camera_id}", daemon=True
            )
            self._consumer_thread = threading.Thread(
                target=self._consumer_loop, name=f"RTSP-Consumer-{self.camera_id}", daemon=True
            )

            self._producer_thread.start()
            self._consumer_thread.start()
            logger.info("[%s] Started multi-threaded RTSP capture pipeline.", self.camera_id)

    def stop(self) -> None:
        """Gracefully stop background threads and release resources."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        if self._producer_thread and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=3.0)

        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=3.0)

        self.is_connected = False
        logger.info("[%s] Stopped stream capture. Total keyframes extracted: %d", self.camera_id, len(self.extracted_keyframes))

    def _connect_capture(self) -> Optional[cv2.VideoCapture]:
        """Attempt to open VideoCapture with RTSP/RTMP/YouTube parameters."""
        logger.info("[%s] Opening video source: %s", self.camera_id, self.stream_url)
        
        url_to_open = self.stream_url
        if "youtube.com" in url_to_open.lower() or "youtu.be" in url_to_open.lower():
            try:
                import subprocess
                res = subprocess.run(["yt-dlp", "-g", url_to_open], capture_output=True, text=True, timeout=15)
                if res.returncode == 0 and res.stdout.strip():
                    url_to_open = res.stdout.strip().split("\n")[0]
                    logger.info("[%s] Resolved YouTube live stream link via yt-dlp.", self.camera_id)
            except Exception as e:
                logger.warning("[%s] Could not resolve YouTube link (%s). Using original URL.", self.camera_id, e)

        # Enable FFmpeg TCP transport for RTSP streams
        if url_to_open.lower().startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        cap = cv2.VideoCapture(url_to_open, cv2.CAP_FFMPEG)
        if cap.isOpened():
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.is_connected = True
            logger.info("[%s] Connected successfully. FPS: %.2f", self.camera_id, self.fps)
            return cap
        else:
            self.is_connected = False
            logger.error("[%s] Failed to open video source.", self.camera_id)
            return None

    def _producer_loop(self) -> None:
        """
        Producer Thread Loop: Reads frames continuously into size-1 Ring Buffer.
        If ring buffer is full, discards oldest frame (ensuring zero live latency).
        Includes automatic reconnection watchdog.
        """
        cap = self._connect_capture()
        attempts = 0

        while self._running:
            if cap is None or not cap.isOpened():
                self.is_connected = False
                attempts += 1
                if attempts > self.max_reconnect_attempts:
                    logger.critical("[%s] Max reconnection attempts reached (%d). Exiting producer.", self.camera_id, attempts)
                    break

                backoff_sec = min(2 * attempts, 10)
                logger.warning("[%s] Stream disconnected. Reconnecting in %ds (Attempt %d/%d)...", 
                               self.camera_id, backoff_sec, attempts, self.max_reconnect_attempts)
                time.sleep(backoff_sec)
                cap = self._connect_capture()
                if cap and cap.isOpened():
                    self.reconnect_count += 1
                    attempts = 0
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("[%s] Failed to read frame from stream. Reconnecting...", self.camera_id)
                cap.release()
                cap = None
                continue

            self.total_frames_read += 1

            # Push frame into Ring Buffer (Size = 1)
            # If buffer is full, pop old frame first to avoid live streaming latency lag
            if self._ring_buffer.full():
                try:
                    self._ring_buffer.get_nowait()
                    self.total_frames_dropped += 1
                except queue.Empty:
                    pass

            try:
                self._ring_buffer.put_nowait((time.time(), self.total_frames_read, frame))
            except queue.Full:
                pass

        if cap:
            cap.release()
        self.is_connected = False

    def _consumer_loop(self) -> None:
        """
        Consumer Thread Loop: Samples frames from Ring Buffer at `sample_interval` rate,
        applies EdgeFrameFilter (dHash/pHash), and saves keyframes to disk without blocking server GIL.
        """
        last_sample_time = 0.0

        while self._running:
            try:
                # Fetch latest frame from Ring Buffer
                capture_time, frame_idx, frame = self._ring_buffer.get(timeout=0.5)
            except queue.Empty:
                continue

            # Enforce sampling interval
            if (capture_time - last_sample_time) < self.sample_interval:
                continue

            last_sample_time = capture_time

            # Format timestamp string (HH:MM:SS)
            hrs = int(capture_time // 3600) % 24
            mins = int((capture_time % 3600) // 60)
            secs = int(capture_time % 60)
            ts_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"

            # Evaluate dHash / pHash
            audit_item = self.hash_filter.evaluate_frame(frame, frame_idx=frame_idx, timestamp=ts_str)
            self.audit_trail.append(audit_item)

            if audit_item["is_keyframe"]:
                clean_ts = ts_str.replace(":", "_")
                out_filename = f"{self.camera_id}_{clean_ts}_{frame_idx}.jpg"
                out_path = self.output_dir / out_filename

                cv2.imwrite(str(out_path), frame)

                keyframe_meta = {
                    "camera": self.camera_id,
                    "timestamp": ts_str,
                    "seconds": round(capture_time, 2),
                    "frame_idx": frame_idx,
                    "image_path": str(out_path),
                    "hash_hex": audit_item["hash_hex"],
                    "hamming_distance": audit_item["hamming_distance"],
                    "motion_pct": audit_item["motion_pct"],
                }
                with self._lock:
                    self.extracted_keyframes.append(keyframe_meta)

    def get_status(self) -> Dict[str, Any]:
        """Return real-time stream capture health status and metrics."""
        filter_summary = self.hash_filter.get_summary()
        with self._lock:
            keyframe_count = len(self.extracted_keyframes)

        return {
            "camera_id": self.camera_id,
            "stream_url": self.stream_url,
            "is_connected": self.is_connected,
            "is_running": self._running,
            "fps": round(self.fps, 2),
            "total_frames_read": self.total_frames_read,
            "total_frames_dropped": self.total_frames_dropped,
            "reconnect_count": self.reconnect_count,
            "keyframes_kept": keyframe_count,
            "frames_skipped": filter_summary.get("frames_skipped", 0),
            "llm_compute_saved_pct": filter_summary.get("llm_compute_saved_pct", 0.0),
            "hash_method": self.hash_filter.method,
            "hamming_threshold": self.hash_filter.threshold,
        }


class MultiCameraStreamManager:
    """
    Manages concurrent multi-camera RTSP/RTMP stream captures.
    """

    def __init__(self, output_dir: str = "data/extracted_frames"):
        self.output_dir = output_dir
        self.streams: Dict[str, RTSPStreamCapture] = {}
        self._lock = threading.Lock()

    def add_camera(
        self,
        camera_id: str,
        stream_url: str,
        sample_interval: float = 5.0,
        hash_method: str = "dhash",
        threshold: int = 10,
    ) -> RTSPStreamCapture:
        """Add and start a new camera stream capture."""
        with self._lock:
            if camera_id in self.streams:
                logger.info("Camera %s already active. Stopping existing stream...", camera_id)
                self.streams[camera_id].stop()

            hash_filter = EdgeFrameFilter(method=hash_method, threshold=threshold)
            stream = RTSPStreamCapture(
                camera_id=camera_id,
                stream_url=stream_url,
                output_dir=self.output_dir,
                sample_interval=sample_interval,
                hash_filter=hash_filter,
            )
            self.streams[camera_id] = stream
            stream.start()
            return stream

    def remove_camera(self, camera_id: str) -> bool:
        """Stop and remove a camera stream capture."""
        with self._lock:
            if camera_id in self.streams:
                self.streams[camera_id].stop()
                del self.streams[camera_id]
                return True
            return False

    def get_all_statuses(self) -> List[Dict[str, Any]]:
        """Return health status list for all active camera streams."""
        with self._lock:
            return [stream.get_status() for stream in self.streams.values()]

    def stop_all(self) -> None:
        """Stop all camera stream captures."""
        with self._lock:
            for stream in self.streams.values():
                stream.stop()
            self.streams.clear()
        logger.info("Stopped all multi-camera RTSP stream captures.")
