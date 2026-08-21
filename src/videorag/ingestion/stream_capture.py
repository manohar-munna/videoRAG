"""
src/videorag/ingestion/stream_capture.py
-----------------------------------------
Async Multi-Threaded RTSP Stream Capture & Producer-Consumer Ring Buffer Queue.

Provides high-performance, non-blocking video stream ingestion for live CCTV network feeds (RTSP/RTMP),
YouTube live streams, and local video files.
Features:
- Dedicated thread-based frame producer reading into a size-1 Ring Buffer (zero latency).
- Pause/Resume and Stop/Start stream lifecycle controls.
- Dynamic per-camera folder creation under data/cameras/<camera_id>/extracted_frames/.
- Automatic RTSP socket watchdog with reconnection backoff.
- Integrated EdgeFrameFilter (dHash/pHash) keyframe extraction.
- MultiCameraStreamManager integrated with persistent CameraRegistry.
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
from videorag.ingestion.camera_registry import CameraRegistry

logger = logging.getLogger("videorag.ingestion.stream_capture")


class RTSPStreamCapture:
    """
    Thread-safe, non-blocking RTSP/RTMP/YouTube/File stream reader with Ring Buffer & Watchdog.
    """

    def __init__(
        self,
        camera_id: str,
        stream_url: str,
        output_dir: Optional[str] = None,
        sample_interval: float = 5.0,
        hash_filter: Optional[EdgeFrameFilter] = None,
        max_reconnect_attempts: int = 10,
        on_keyframe_callback: Optional[Any] = None,
    ):
        """
        :param camera_id: Unique camera identifier (e.g. 'CAM_01', 'CAM_3000')
        :param stream_url: RTSP/RTMP/YouTube stream URL or local video file path
        :param output_dir: Base output directory (defaults dynamically to data/cameras/<camera_id>/extracted_frames)
        :param sample_interval: Seconds between keyframe evaluations (default: 5.0s)
        :param hash_filter: EdgeFrameFilter (dHash/pHash) instance
        """
        self.camera_id = camera_id.strip()
        self.stream_url = str(stream_url).strip()

        # Dynamic per-camera directory creation (Zero hardcoding)
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path("data/cameras") / self.camera_id / "extracted_frames"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sample_interval = sample_interval
        self.hash_filter = hash_filter or EdgeFrameFilter(method="dhash", threshold=10)
        self.max_reconnect_attempts = max_reconnect_attempts
        self.on_keyframe_callback = on_keyframe_callback

        # Threading state
        self._running = False
        self._paused = False
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
        self.start_time = time.time()
        self.current_position_sec = 0.0
        self.total_duration_sec: Optional[float] = None
        self.total_frames_count: Optional[int] = None
        self.progress_pct: Optional[float] = None
        
        # Camera source type detection
        url_lower = self.stream_url.lower()
        is_local = Path(self.stream_url).is_file() or any(url_lower.endswith(ext) for ext in [".mp4", ".avi", ".mkv", ".mov", ".webm"])
        if is_local:
            self.camera_type = "video_file"
            self.is_live = False
        elif "youtube.com" in url_lower or "youtu.be" in url_lower:
            if "/live" in url_lower or "live=1" in url_lower or "1eic9bv" in url_lower or "eo_1lwqs" in url_lower:
                self.camera_type = "youtube_stream"
                self.is_live = True
            else:
                self.camera_type = "youtube_video"
                self.is_live = False
        else:
            self.camera_type = "rtsp_stream"
            self.is_live = True
        
        self.extracted_keyframes: List[Dict[str, Any]] = []
        self.audit_trail: List[Dict[str, Any]] = []

    def start(self) -> None:
        """Start producer and consumer background worker threads."""
        with self._lock:
            if self._running:
                if self._paused:
                    self._paused = False
                    logger.info("[%s] Resumed stream capture threads.", self.camera_id)
                return

            self._running = True
            self._paused = False
            self.start_time = time.time()
            self._producer_thread = threading.Thread(
                target=self._producer_loop, name=f"RTSP-Producer-{self.camera_id}", daemon=True
            )
            self._consumer_thread = threading.Thread(
                target=self._consumer_loop, name=f"RTSP-Consumer-{self.camera_id}", daemon=True
            )

            self._producer_thread.start()
            self._consumer_thread.start()
            logger.info("[%s] Started multi-threaded capture pipeline (%s).", self.camera_id, self.camera_type)

    def pause(self) -> None:
        """Pause extraction without destroying the stream object or removing the camera."""
        with self._lock:
            self._paused = True
            self._running = False

        if self._producer_thread and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=2.0)

        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=2.0)

        self.is_connected = False
        logger.info("[%s] Stream extraction paused (idle).", self.camera_id)

    def resume(self) -> None:
        """Resume stream extraction threads."""
        self.start()

    def stop(self) -> None:
        """Gracefully stop background threads and release resources."""
        with self._lock:
            if not self._running and not self._paused:
                return
            self._running = False
            self._paused = False

        if self._producer_thread and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=2.0)

        if self._consumer_thread and self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=2.0)

        self.is_connected = False
        logger.info("[%s] Stopped stream capture. Total keyframes extracted: %d", self.camera_id, len(self.extracted_keyframes))

    def _connect_capture(self) -> Optional[cv2.VideoCapture]:
        """Attempt to open VideoCapture with RTSP/RTMP/YouTube parameters."""
        logger.info("[%s] Opening video source (%s): %s", self.camera_id, self.camera_type, self.stream_url)
        
        url_to_open = self.stream_url
        if "youtube.com" in url_to_open.lower() or "youtu.be" in url_to_open.lower():
            try:
                import subprocess
                res = subprocess.run(["yt-dlp", "-g", url_to_open], capture_output=True, text=True, timeout=15)
                if res.returncode == 0 and res.stdout.strip():
                    url_to_open = res.stdout.strip().split("\n")[0]
                    logger.info("[%s] Resolved YouTube stream link via yt-dlp.", self.camera_id)
            except Exception as e:
                logger.warning("[%s] Could not resolve YouTube link (%s). Using original URL.", self.camera_id, e)

        # Enable FFmpeg TCP transport for RTSP streams
        if url_to_open.lower().startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        cap = cv2.VideoCapture(url_to_open, cv2.CAP_FFMPEG)
        if cap.isOpened():
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            # Query duration for static video files if available
            frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_cnt > 0:
                self.total_frames_count = frame_cnt
                self.total_duration_sec = round(frame_cnt / self.fps, 2)
            else:
                self.total_frames_count = None
                self.total_duration_sec = None

            self.is_connected = True
            logger.info("[%s] Connected successfully. FPS: %.2f | Duration: %s", 
                        self.camera_id, self.fps, f"{self.total_duration_sec}s" if self.total_duration_sec else "LIVE")
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
                # If reading a local MP4 file, stop cleanly at EOF
                if Path(self.stream_url).exists() and Path(self.stream_url).is_file():
                    logger.info("[%s] Reached end of local video file (%s). Stream capture completed.", self.camera_id, self.stream_url)
                    self.is_connected = False
                    break

                logger.warning("[%s] Failed to read frame from live stream. Reconnecting...", self.camera_id)
                cap.release()
                cap = None
                continue

            self.total_frames_read += 1

            # For video files, use actual video playback position; for live streams, use current epoch time
            is_video_file = Path(self.stream_url).exists() and Path(self.stream_url).is_file()
            if is_video_file:
                pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                frame_sec = (pos_msec / 1000.0) if pos_msec > 0 else (self.total_frames_read / 30.0)
                self.current_position_sec = frame_sec
                if self.total_duration_sec and self.total_duration_sec > 0:
                    self.progress_pct = min(100.0, round((self.current_position_sec / self.total_duration_sec) * 100.0, 1))
            else:
                frame_sec = time.time()
                self.current_position_sec = time.time() - self.start_time
                self.progress_pct = None

            # Push frame into Ring Buffer (Size = 1)
            # If buffer is full, pop old frame first to avoid live streaming latency lag
            if self._ring_buffer.full():
                try:
                    self._ring_buffer.get_nowait()
                    self.total_frames_dropped += 1
                except queue.Empty:
                    pass

            try:
                self._ring_buffer.put_nowait((frame_sec, self.total_frames_read, frame))
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
        is_video_file = Path(self.stream_url).exists() and Path(self.stream_url).is_file()

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
            if is_video_file:
                hrs = int(capture_time // 3600)
                mins = int((capture_time % 3600) // 60)
                secs = int(capture_time % 60)
                ts_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"
                display_sec = round(capture_time, 2)
            else:
                ts_str = time.strftime("%H:%M:%S", time.localtime(capture_time))
                display_sec = round(capture_time, 2)

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
                    "seconds": display_sec,
                    "epoch_time": round(capture_time, 3) if not is_video_file else None,
                    "is_live": not is_video_file,
                    "frame_idx": frame_idx,
                    "image_path": str(out_path),
                    "hash_hex": audit_item["hash_hex"],
                    "hamming_distance": audit_item["hamming_distance"],
                    "motion_pct": audit_item["motion_pct"],
                }
                with self._lock:
                    self.extracted_keyframes.append(keyframe_meta)

                if self.on_keyframe_callback:
                    try:
                        self.on_keyframe_callback(keyframe_meta)
                    except Exception as exc:
                        logger.error("[%s] Error in on_keyframe_callback: %s", self.camera_id, exc)

    def get_status(self) -> Dict[str, Any]:
        """Return real-time stream capture health status and metrics."""
        filter_summary = self.hash_filter.get_summary()
        with self._lock:
            keyframe_count = len(self.extracted_keyframes)

        state_str = "RUNNING" if self._running else ("PAUSED" if self._paused else "STOPPED")

        return {
            "camera_id": self.camera_id,
            "stream_url": self.stream_url,
            "camera_type": self.camera_type,
            "is_live": self.is_live,
            "is_connected": self.is_connected,
            "is_running": self._running,
            "is_paused": self._paused,
            "state": state_str,
            "fps": round(self.fps, 2),
            "total_frames_read": self.total_frames_read,
            "total_frames_dropped": self.total_frames_dropped,
            "total_frames_count": self.total_frames_count,
            "total_duration_sec": self.total_duration_sec,
            "current_position_sec": round(self.current_position_sec, 2),
            "progress_pct": self.progress_pct,
            "reconnect_count": self.reconnect_count,
            "keyframes_kept": keyframe_count,
            "frames_skipped": filter_summary.get("frames_skipped", 0),
            "llm_compute_saved_pct": filter_summary.get("llm_compute_saved_pct", 0.0),
            "hash_method": self.hash_filter.method,
            "hamming_threshold": self.hash_filter.threshold,
        }


class MultiCameraStreamManager:
    """
    Manages concurrent multi-camera RTSP/RTMP stream captures and syncs with persistent CameraRegistry.
    """

    def __init__(self, registry: Optional[CameraRegistry] = None, on_keyframe_callback: Optional[Any] = None):
        self.registry = registry or CameraRegistry()
        self.on_keyframe_callback = on_keyframe_callback
        self.streams: Dict[str, RTSPStreamCapture] = {}
        self._lock = threading.Lock()

    def initialize_from_registry(self) -> None:
        """Boot up streams that have status 'running' in persistent registry."""
        configs = self.registry.get_all()
        for cfg in configs:
            cam_id = cfg["camera_id"]
            stream_url = cfg["stream_url"]
            status = cfg.get("status", "stopped")
            interval = cfg.get("sample_interval", 5.0)
            method = cfg.get("hash_method", "dhash")
            thresh = cfg.get("threshold", 10)

            stream = RTSPStreamCapture(
                camera_id=cam_id,
                stream_url=stream_url,
                sample_interval=interval,
                hash_filter=EdgeFrameFilter(method=method, threshold=thresh),
                on_keyframe_callback=self.on_keyframe_callback,
            )
            self.streams[cam_id] = stream

            if status == "running":
                stream.start()
            elif status == "paused":
                stream._paused = True

        logger.info("MultiCameraStreamManager initialized with %d persistent camera streams.", len(self.streams))

    def add_camera(
        self,
        camera_id: str,
        stream_url: str,
        name: Optional[str] = None,
        sample_interval: float = 5.0,
        hash_method: str = "dhash",
        threshold: int = 10,
        start_immediately: bool = True,
    ) -> RTSPStreamCapture:
        """Register camera dynamically, persist to registry, and start capture."""
        with self._lock:
            # 1. Update Persistent Registry
            status = "running" if start_immediately else "stopped"
            self.registry.register_camera(
                camera_id=camera_id,
                stream_url=stream_url,
                name=name,
                sample_interval=sample_interval,
                hash_method=hash_method,
                threshold=threshold,
                status=status,
            )

            # 2. Start StreamCapture
            if camera_id in self.streams:
                self.streams[camera_id].stop()

            stream = RTSPStreamCapture(
                camera_id=camera_id,
                stream_url=stream_url,
                sample_interval=sample_interval,
                hash_filter=EdgeFrameFilter(method=hash_method, threshold=threshold),
                on_keyframe_callback=self.on_keyframe_callback,
            )
            self.streams[camera_id] = stream
            if start_immediately:
                stream.start()

            return stream

    def pause_camera(self, camera_id: str) -> bool:
        """Pause stream extraction without removing the camera from system."""
        with self._lock:
            if camera_id in self.streams:
                self.streams[camera_id].pause()
                self.registry.update_status(camera_id, "paused")
                return True
            return False

    def resume_camera(self, camera_id: str) -> bool:
        """Resume paused stream extraction."""
        with self._lock:
            if camera_id in self.streams:
                self.streams[camera_id].resume()
                self.registry.update_status(camera_id, "running")
                return True
            return False

    def remove_camera(self, camera_id: str) -> bool:
        """Stop and remove a camera from both stream manager and persistent registry."""
        with self._lock:
            if camera_id in self.streams:
                self.streams[camera_id].stop()
                del self.streams[camera_id]
            self.registry.remove_camera(camera_id)
            return True

    def reindex_camera(self, camera_id: str, hash_method: str = "dhash", threshold: int = 10) -> Optional[RTSPStreamCapture]:
        """Stop extraction, purge previous frame cache and events, and restart extraction from scratch."""
        import shutil
        with self._lock:
            stream_url = ""
            interval = 5.0
            if camera_id in self.streams:
                old_stream = self.streams[camera_id]
                stream_url = old_stream.stream_url
                interval = old_stream.sample_interval
                old_stream.stop()
            else:
                cfg = self.registry.get(camera_id)
                if cfg:
                    stream_url = cfg.get("stream_url", "")
                    interval = cfg.get("sample_interval", 5.0)

            if not stream_url:
                return None

            # 1. Purge extracted frame images
            cam_frame_dir = Path("data/cameras") / camera_id / "extracted_frames"
            if cam_frame_dir.exists():
                shutil.rmtree(cam_frame_dir, ignore_errors=True)
            cam_frame_dir.mkdir(parents=True, exist_ok=True)

            # 2. Reset camera events.json
            cam_events = Path("data/cameras") / camera_id / "events.json"
            if cam_events.exists():
                with open(cam_events, "w", encoding="utf-8") as fh:
                    import json
                    json.dump([], fh)

            # 3. Create fresh stream capture instance
            new_stream = RTSPStreamCapture(
                camera_id=camera_id,
                stream_url=stream_url,
                sample_interval=interval,
                hash_filter=EdgeFrameFilter(method=hash_method, threshold=threshold),
                on_keyframe_callback=self.on_keyframe_callback,
            )
            self.streams[camera_id] = new_stream
            self.registry.update_status(camera_id, "running")
            new_stream.start()
            logger.info("[%s] Re-indexed camera from scratch with %s (threshold=%d).", camera_id, hash_method, threshold)
            return new_stream

    def get_all_statuses(self) -> List[Dict[str, Any]]:
        """Return status for all registered camera streams."""
        with self._lock:
            statuses = []
            for cam_id, stream in self.streams.items():
                st = stream.get_status()
                cfg = self.registry.get(cam_id) or {}
                st["name"] = cfg.get("name", cam_id)
                st["type"] = cfg.get("type", "rtsp_stream")
                statuses.append(st)
            return statuses

    def stop_all(self) -> None:
        """Stop all camera stream captures."""
        with self._lock:
            for stream in self.streams.values():
                stream.stop()
        logger.info("Stopped all multi-camera RTSP stream captures.")
