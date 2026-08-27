"""
src/videorag/llm/vlm_process_manager.py
---------------------------------------
Runtime Profile Manager for VideoRAG VLM Inference.

Defines and manages explicit runtime profiles:
- Desktop: Qwen3-VL 4B, GPU (-ngl 99), 4096 ctx, 4 slots, FP16 KV cache, 5-frame storyboard
- Mobile:  Qwen2-VL 2B, CPU (-ngl 0), 2048 ctx, 1 slot, Q8_0 KV cache, 3-frame storyboard (512px max dim)

Guarantees clean unload/reload switching so both VLMs never reside concurrently in RAM/VRAM.
"""

import json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LLAMA_SERVER_EXE = _PROJECT_ROOT / "tools" / "llama" / "llama-server.exe"

RUNTIME_PROFILES: Dict[str, Dict[str, Any]] = {
    "desktop": {
        "id": "desktop",
        "name": "Desktop 4B GPU",
        "description": "High-throughput CUDA offloaded 4B VLM with 4 slots and full 5-frame temporal context",
        "model_file": "models/qwen3_vl/Qwen3VL-4B-Instruct-Q4_K_M.gguf",
        "mmproj_file": "models/qwen3_vl/mmproj-Qwen3VL-4B-Instruct-F16.gguf",
        "device": "gpu",
        "ngl": 99,
        "ctx": 4096,
        "slots": 1,
        "kv_cache_type": "f16",
        "ctk": "f16",
        "ctv": "f16",
        "threads": 8,
        "context_window": 2,      # ±2 neighbouring frames (5 frames total)
        "max_img_dim": 768,
        "port": 8080,
    },
    "mobile": {
        "id": "mobile",
        "name": "Mobile 2B CPU",
        "description": "Low-footprint CPU quantized 2B VLM with Q8_0 KV cache, 3-frame storyboard & 512px inference scaling",
        "model_file": "models/qwen2_vl_2b/Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
        "mmproj_file": "models/qwen2_vl_2b/mmproj-Qwen2-VL-2B-Instruct-f16.gguf",
        "device": "cpu",
        "ngl": 0,
        "ctx": 2048,
        "slots": 1,
        "kv_cache_type": "q8_0",
        "ctk": "q8_0",
        "ctv": "q8_0",
        "threads": 6,
        "context_window": 1,      # ±1 neighbouring frame (3 frames total: Pre, Anchor, Post)
        "max_img_dim": 512,       # max dimension 512px for low visual token count
        "port": 8080,
    },
}


class VLMProcessManager:
    """Manages the lifecycle of the active llama-server process with profile switching."""

    def __init__(self, default_profile: str = "desktop"):
        self.lock = threading.Lock()
        self.current_profile_id: str = default_profile
        self.process: Optional[subprocess.Popen] = None
        self.port: int = 8080

    def get_profile_info(self, profile_id: Optional[str] = None) -> Dict[str, Any]:
        pid = profile_id or self.current_profile_id
        return RUNTIME_PROFILES.get(pid, RUNTIME_PROFILES["desktop"])

    def is_server_healthy(self, port: int = 8080, timeout: float = 1.5) -> bool:
        url = f"http://127.0.0.1:{port}/health"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def stop_current_server(self) -> None:
        """Unload and terminate current running llama-server to free all RAM/VRAM."""
        with self.lock:
            if self.process is not None:
                logger.info("Unloading current VLM server (PID: %d)...", self.process.pid)
                try:
                    self.process.terminate()
                    self.process.wait(timeout=4.0)
                except Exception:
                    try:
                        self.process.kill()
                    except Exception:
                        pass
                self.process = None

            # Clean up any orphan llama-server processes on Windows
            if sys.platform == "win32":
                try:
                    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe", "/T"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass

            # Free PyTorch CUDA context if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            time.sleep(0.5)
            logger.info("Previous VLM server unloaded successfully.")

    def start_profile(self, profile_id: str = "desktop") -> bool:
        """Start a specific runtime profile after fully unloading previous one."""
        with self.lock:
            profile = RUNTIME_PROFILES.get(profile_id, RUNTIME_PROFILES["desktop"])
            self.current_profile_id = profile_id

            model_path = _PROJECT_ROOT / profile["model_file"]
            mmproj_path = _PROJECT_ROOT / profile["mmproj_file"]

            if not model_path.exists():
                logger.error("Model file not found: %s", model_path)
                return False
            if not mmproj_path.exists():
                logger.error("mmproj file not found: %s", mmproj_path)
                return False

            cmd = [
                str(LLAMA_SERVER_EXE),
                "-m", str(model_path),
                "--mmproj", str(mmproj_path),
                "-ngl", str(profile["ngl"]),
                "-c", str(profile["ctx"]),
                "--parallel", str(profile["slots"]),
                "-ctk", profile["ctk"],
                "-ctv", profile["ctv"],
                "--threads", str(profile["threads"]),
                "--port", str(profile["port"]),
                "--host", "127.0.0.1",
            ]

            logger.info(
                "Launching profile '%s' (%s) on port %d...",
                profile["name"], profile["device"].upper(), profile["port"]
            )
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Wait for server health
            start_t = time.time()
            for attempt in range(1, 40):
                if self.is_server_healthy(profile["port"]):
                    elapsed = round((time.time() - start_t) * 1000, 1)
                    logger.info("Profile '%s' is ready in %s ms!", profile["name"], elapsed)
                    return True
                time.sleep(0.5)

            logger.error("Profile '%s' failed to become healthy within 20s.", profile["name"])
            return False

    def switch_profile(self, target_profile_id: str) -> Dict[str, Any]:
        """Switch from current profile to target profile with clean unload and reload."""
        if target_profile_id not in RUNTIME_PROFILES:
            return {"success": False, "error": f"Unknown profile: {target_profile_id}"}

        logger.info("=== SWITCHING PROFILE: %s -> %s ===", self.current_profile_id, target_profile_id)
        t0 = time.time()
        
        # 1. Fully unload previous VLM
        self.stop_current_server()

        # 2. Launch new VLM profile
        ok = self.start_profile(target_profile_id)
        elapsed_ms = round((time.time() - t0) * 1000, 1)

        if ok:
            return {
                "success": True,
                "profile": RUNTIME_PROFILES[target_profile_id],
                "switch_time_ms": elapsed_ms,
                "message": f"Successfully activated {RUNTIME_PROFILES[target_profile_id]['name']}",
            }
        else:
            return {
                "success": False,
                "error": f"Failed to start {target_profile_id} profile",
                "switch_time_ms": elapsed_ms,
            }


# Global singleton instance
VLM_MANAGER = VLMProcessManager(default_profile="desktop")
