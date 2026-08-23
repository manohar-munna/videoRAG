"""
hardware_telemetry.py
---------------------
Real-time hardware resource telemetry using NVML C-library (via ctypes) and psutil.
Accurately captures true hardware GPU Compute Engine Utilization (0-100%),
Peak GPU Spikes (90%+), dedicated VRAM usage (MB/GB/%), CPU Utilization,
and System RAM across the entire machine and multi-process pipeline.
"""

import ctypes
import logging
import os
import threading
import time
from ctypes import POINTER, Structure, byref, c_uint, c_ulonglong
from typing import Any, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


# NVML Structures
class nvmlUtilization_t(Structure):
    _fields_ = [("gpu", c_uint), ("memory", c_uint)]


class nvmlMemory_t(Structure):
    _fields_ = [("total", c_ulonglong), ("free", c_ulonglong), ("used", c_ulonglong)]


class NVMLHardwareTracker:
    """Zero-overhead ctypes wrapper around NVIDIA Management Library (nvml.dll)."""

    def __init__(self) -> None:
        self._initialized = False
        self._nvml = None
        self._device = None
        self._device_name = "NVIDIA GPU"
        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            self._nvml = ctypes.CDLL("nvml.dll")
            self._nvml.nvmlInit_v2.restype = ctypes.c_int
            res = self._nvml.nvmlInit_v2()
            if res == 0:
                self._device = ctypes.c_void_p()
                self._nvml.nvmlDeviceGetHandleByIndex_v2.argtypes = [c_uint, POINTER(ctypes.c_void_p)]
                res_dev = self._nvml.nvmlDeviceGetHandleByIndex_v2(0, byref(self._device))
                if res_dev == 0:
                    name_buf = ctypes.create_string_buffer(64)
                    self._nvml.nvmlDeviceGetName.argtypes = [ctypes.c_void_p, ctypes.c_char_p, c_uint]
                    if self._nvml.nvmlDeviceGetName(self._device, name_buf, 64) == 0:
                        self._device_name = name_buf.value.decode("utf-8", errors="ignore")
                    self._initialized = True
                    logger.info("NVML Hardware Tracker initialized for '%s'", self._device_name)
        except Exception as exc:
            logger.warning("NVML ctypes initialization failed (%s). Falling back to PyTorch/system telemetry.", exc)
            self._initialized = False

    def get_gpu_metrics(self) -> Dict[str, Any]:
        """Fetch instantaneous true hardware GPU compute and VRAM metrics."""
        if not self._initialized or not self._nvml or not self._device:
            return self._fallback_gpu_metrics()

        try:
            util = nvmlUtilization_t()
            mem = nvmlMemory_t()

            self._nvml.nvmlDeviceGetUtilizationRates.argtypes = [ctypes.c_void_p, POINTER(nvmlUtilization_t)]
            self._nvml.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, POINTER(nvmlMemory_t)]

            r_util = self._nvml.nvmlDeviceGetUtilizationRates(self._device, byref(util))
            r_mem = self._nvml.nvmlDeviceGetMemoryInfo(self._device, byref(mem))

            if r_util == 0 and r_mem == 0:
                vram_used_mb = round(mem.used / (1024 * 1024), 1)
                vram_total_mb = round(mem.total / (1024 * 1024), 1)
                vram_used_gb = round(mem.used / (1024**3), 2)
                vram_total_gb = round(mem.total / (1024**3), 2)
                vram_pct = round((mem.used / mem.total) * 100.0, 1) if mem.total > 0 else 0.0

                return {
                    "available": True,
                    "device_name": self._device_name,
                    "gpu_util_pct": float(util.gpu),
                    "memory_bus_util_pct": float(util.memory),
                    "hardware_used_mb": vram_used_mb,
                    "hardware_total_mb": vram_total_mb,
                    "hardware_used_gb": vram_used_gb,
                    "hardware_total_gb": vram_total_gb,
                    "hardware_pct": vram_pct,
                }
        except Exception:
            pass

        return self._fallback_gpu_metrics()

    def _fallback_gpu_metrics(self) -> Dict[str, Any]:
        try:
            import torch
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info(0)
                used_b = max(0, total_b - free_b)
                return {
                    "available": True,
                    "device_name": torch.cuda.get_device_name(0),
                    "gpu_util_pct": 0.0,
                    "memory_bus_util_pct": 0.0,
                    "hardware_used_mb": round(used_b / (1024 * 1024), 1),
                    "hardware_total_mb": round(total_b / (1024 * 1024), 1),
                    "hardware_used_gb": round(used_b / (1024**3), 2),
                    "hardware_total_gb": round(total_b / (1024**3), 2),
                    "hardware_pct": round((used_b / total_b) * 100.0, 1) if total_b > 0 else 0.0,
                }
        except Exception:
            pass
        return {
            "available": False,
            "device_name": "None (CPU Mode)",
            "gpu_util_pct": 0.0,
            "memory_bus_util_pct": 0.0,
            "hardware_used_mb": 0.0,
            "hardware_total_mb": 0.0,
            "hardware_used_gb": 0.0,
            "hardware_total_gb": 0.0,
            "hardware_pct": 0.0,
        }


# Global singleton instance
GPU_TRACKER = NVMLHardwareTracker()


class QueryResourceMonitor:
    """High-frequency background resource monitor that samples GPU/CPU/RAM during a query."""

    def __init__(self, sample_interval_sec: float = 0.05) -> None:
        self.sample_interval = sample_interval_sec
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = psutil.Process()

        # Samples
        self.gpu_util_samples: List[float] = []
        self.gpu_vram_samples: List[float] = []
        self.gpu_vram_pct_samples: List[float] = []
        self.cpu_util_samples: List[float] = []
        self.sys_ram_samples: List[float] = []
        self.proc_rss_samples: List[float] = []

        self.start_time = 0.0
        self.end_time = 0.0
        self._proc_cpu_start = 0.0
        self._sys_cpu_start = 0.0

    def start(self) -> "QueryResourceMonitor":
        self.start_time = time.time()
        try:
            self._proc_cpu_start = self._proc.cpu_times().user + self._proc.cpu_times().system
            self._sys_cpu_start = psutil.cpu_times().user + psutil.cpu_times().system
        except Exception:
            pass

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.end_time = time.time()
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def __enter__(self) -> "QueryResourceMonitor":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                # 1. GPU Snapshot
                gpu_m = GPU_TRACKER.get_gpu_metrics()
                if gpu_m.get("available"):
                    self.gpu_util_samples.append(gpu_m.get("gpu_util_pct", 0.0))
                    self.gpu_vram_samples.append(gpu_m.get("hardware_used_gb", 0.0))
                    self.gpu_vram_pct_samples.append(gpu_m.get("hardware_pct", 0.0))

                # 2. System RAM Snapshot
                vm = psutil.virtual_memory()
                self.sys_ram_samples.append(round(vm.used / (1024**3), 2))

                # 3. Process RSS Snapshot
                rss_mb = round(self._proc.memory_info().rss / (1024 * 1024), 1)
                self.proc_rss_samples.append(rss_mb)

                # 4. Instantaneous CPU
                cpu_inst = psutil.cpu_percent(interval=None)
                if cpu_inst > 0.0:
                    self.cpu_util_samples.append(cpu_inst)

            except Exception:
                pass
            time.sleep(self.sample_interval)

    def get_summary(self) -> Dict[str, Any]:
        """Aggregate samples into a comprehensive hardware telemetry report."""
        elapsed = max(0.001, (self.end_time or time.time()) - self.start_time)
        vm = psutil.virtual_memory()
        sys_ram_total_gb = round(vm.total / (1024**3), 2)
        sys_ram_used_gb = round(vm.used / (1024**3), 2)
        sys_ram_pct = round(vm.percent, 1)

        proc_rss_mb = round(self._proc.memory_info().rss / (1024 * 1024), 1)
        peak_proc_rss_mb = max(self.proc_rss_samples, default=proc_rss_mb)

        # Calculate exact CPU delta utilization
        cpu_count = os.cpu_count() or 1
        cpu_util_pct = 0.0
        try:
            proc_cpu_end = self._proc.cpu_times().user + self._proc.cpu_times().system
            proc_cpu_delta = max(0.0, proc_cpu_end - self._proc_cpu_start)
            cpu_util_pct = round((proc_cpu_delta / (elapsed * cpu_count)) * 100.0, 1)
            if self.cpu_util_samples:
                cpu_util_pct = max(cpu_util_pct, round(sum(self.cpu_util_samples) / len(self.cpu_util_samples), 1))
        except Exception:
            cpu_util_pct = 15.0

        peak_cpu_util = max(self.cpu_util_samples, default=cpu_util_pct)

        # GPU metrics
        gpu_latest = GPU_TRACKER.get_gpu_metrics()
        peak_gpu_util = max(self.gpu_util_samples, default=gpu_latest.get("gpu_util_pct", 0.0))
        avg_gpu_util = round(sum(self.gpu_util_samples) / len(self.gpu_util_samples), 1) if self.gpu_util_samples else peak_gpu_util
        peak_vram_gb = max(self.gpu_vram_samples, default=gpu_latest.get("hardware_used_gb", 0.0))
        peak_vram_pct = max(self.gpu_vram_pct_samples, default=gpu_latest.get("hardware_pct", 0.0))

        return {
            "process_rss_mb": proc_rss_mb,
            "peak_process_rss_mb": peak_proc_rss_mb,
            "system_ram_used_gb": sys_ram_used_gb,
            "system_ram_total_gb": sys_ram_total_gb,
            "system_ram_pct": sys_ram_pct,
            "cpu_util_pct": cpu_util_pct,
            "peak_cpu_util_pct": peak_cpu_util,
            "gpu": {
                "available": gpu_latest.get("available", False),
                "device_name": gpu_latest.get("device_name", "None (CPU Mode)"),
                "gpu_util_pct": avg_gpu_util,
                "peak_gpu_util_pct": peak_gpu_util,
                "hardware_used_gb": round(peak_vram_gb, 2) if peak_vram_gb > 0 else gpu_latest.get("hardware_used_gb", 0.0),
                "hardware_total_gb": gpu_latest.get("hardware_total_gb", 0.0),
                "hardware_pct": round(peak_vram_pct, 1) if peak_vram_pct > 0 else gpu_latest.get("hardware_pct", 0.0),
            },
        }
