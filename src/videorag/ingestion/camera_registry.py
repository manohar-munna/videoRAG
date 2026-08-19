"""
src/videorag/ingestion/camera_registry.py
-----------------------------------------
Persistent Camera Registry for VideoRAG.
Saves and loads registered cameras to/from data/cameras_registry.json.
Ensures camera definitions and states (running, paused, stopped) persist across restarts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger("videorag.ingestion.camera_registry")

_DEFAULT_REGISTRY_PATH = Path("data/cameras_registry.json")


class CameraRegistry:
    """Manages persistent camera definitions on disk."""

    def __init__(self, registry_file: Path = _DEFAULT_REGISTRY_PATH):
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self._cameras: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """Load registry from JSON file or initialize with default actual cameras."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as fh:
                    self._cameras = json.load(fh)
                logger.info("Loaded %d cameras from %s", len(self._cameras), self.registry_file)
                return
            except Exception as e:
                logger.warning("Could not read %s (%s). Re-initializing.", self.registry_file, e)

        # Default initial actual camera configurations (Real footage & live stream)
        self._cameras = {
            "CAM_01": {
                "camera_id": "CAM_01",
                "name": "Main Entrance (Video Footage)",
                "stream_url": "Video Footage/sample_cctv.mp4",
                "type": "video_file",
                "sample_interval": 15.0,
                "hash_method": "dhash",
                "threshold": 10,
                "status": "stopped",
            },
            "CAM_3000": {
                "camera_id": "CAM_3000",
                "name": "YouTube Live Surveillance Feed",
                "stream_url": "https://www.youtube.com/watch?v=1EiC9bvVGnk",
                "type": "youtube_stream",
                "sample_interval": 5.0,
                "hash_method": "dhash",
                "threshold": 10,
                "status": "running",
            },
        }
        self.save()

    def save(self) -> None:
        """Persist current registry to JSON file."""
        try:
            with open(self.registry_file, "w", encoding="utf-8") as fh:
                json.dump(self._cameras, fh, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("Failed to save camera registry: %s", e)

    def get_all(self) -> List[Dict[str, Any]]:
        """Return list of all registered cameras."""
        return list(self._cameras.values())

    def get(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Get specific camera configuration."""
        return self._cameras.get(camera_id)

    def register_camera(
        self,
        camera_id: str,
        stream_url: str,
        name: Optional[str] = None,
        sample_interval: float = 5.0,
        hash_method: str = "dhash",
        threshold: int = 10,
        status: str = "running",
    ) -> Dict[str, Any]:
        """Register or update a camera configuration and create its folder dynamically."""
        clean_id = camera_id.strip()
        
        # Determine camera type
        url_lower = stream_url.lower()
        if "youtube.com" in url_lower or "youtu.be" in url_lower:
            cam_type = "youtube_stream"
        elif url_lower.endswith(".mp4") or url_lower.endswith(".avi") or url_lower.endswith(".mkv") or Path(stream_url).exists():
            cam_type = "video_file"
        else:
            cam_type = "rtsp_stream"

        cam_name = name or f"Camera {clean_id}"

        # Dynamically create isolated folder for this camera
        cam_dir = Path("data/cameras") / clean_id / "extracted_frames"
        cam_dir.mkdir(parents=True, exist_ok=True)

        entry = {
            "camera_id": clean_id,
            "name": cam_name,
            "stream_url": stream_url,
            "type": cam_type,
            "sample_interval": sample_interval,
            "hash_method": hash_method,
            "threshold": threshold,
            "status": status,
        }

        self._cameras[clean_id] = entry
        self.save()
        logger.info("Registered camera '%s' in persistent registry.", clean_id)
        return entry

    def update_status(self, camera_id: str, status: str) -> bool:
        """Update camera state ('running', 'paused', 'stopped')."""
        if camera_id in self._cameras:
            self._cameras[camera_id]["status"] = status
            self.save()
            return True
        return False

    def remove_camera(self, camera_id: str) -> bool:
        """Remove a camera from the registry."""
        if camera_id in self._cameras:
            del self._cameras[camera_id]
            self.save()
            logger.info("Removed camera '%s' from registry.", camera_id)
            return True
        return False
