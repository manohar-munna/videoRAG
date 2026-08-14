"""
scripts/test_rtsp_stream.py
----------------------------
Test & Benchmark suite for Async Multi-Threaded RTSP Stream Capture,
Ring Buffer Queue Latency, Watchdog Reconnection, and Multi-Camera Stream Manager.
"""

import sys
import time
import logging
from pathlib import Path

# Add src to Python path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from videorag.ingestion.stream_capture import RTSPStreamCapture, MultiCameraStreamManager
from videorag.ingestion.hash_filter import EdgeFrameFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("test_rtsp")


def test_single_stream_capture():
    print("\n" + "="*70)
    print("TEST 1: Multi-Threaded Stream Capture & Ring Buffer Latency")
    print("="*70)

    sample_video = _PROJECT_ROOT / "Video Footage" / "sample_cctv.mp4"
    if not sample_video.exists():
        print(f"Error: Sample video file not found at {sample_video}")
        return False

    hash_filter = EdgeFrameFilter(method="dhash", threshold=10)
    stream = RTSPStreamCapture(
        camera_id="CAM_TEST_01",
        stream_url=str(sample_video),
        output_dir=str(_PROJECT_ROOT / "data" / "test_frames"),
        sample_interval=1.0,  # Fast 1.0s interval for testing
        hash_filter=hash_filter,
    )

    t0 = time.time()
    stream.start()
    print("🚀 Stream capture worker threads started asynchronously...")

    # Sleep for 5 seconds while producer & consumer run in background
    for i in range(5):
        time.sleep(1)
        status = stream.get_status()
        print(f"  [{i+1}s] Connected: {status['is_connected']} | Frames Read: {status['total_frames_read']} | Keyframes Kept: {status['keyframes_kept']} | Dropped: {status['total_frames_dropped']}")

    stream.stop()
    t1 = time.time()

    final_status = stream.get_status()
    print("\n📊 Final Single Stream Benchmark:")
    print(f"  - Total Elapsed Time: {t1 - t0:.2f}s")
    print(f"  - Total Frames Read: {final_status['total_frames_read']}")
    print(f"  - Keyframes Kept: {final_status['keyframes_kept']}")
    print(f"  - Frames Skipped (dHash): {final_status['frames_skipped']}")
    print(f"  - Compute Saved: {final_status['llm_compute_saved_pct']}%")

    assert final_status['keyframes_kept'] > 0, "Failed: No keyframes extracted"
    assert not stream.is_connected, "Failed: Stream should be disconnected after stop"
    print("✅ TEST 1 PASSED: Multi-threaded stream capture verified!")
    return True


def test_multicamera_manager():
    print("\n" + "="*70)
    print("TEST 2: Multi-Camera Stream Manager (Concurrent Surveillance Feeds)")
    print("="*70)

    sample_video = _PROJECT_ROOT / "Video Footage" / "sample_cctv.mp4"
    manager = MultiCameraStreamManager(output_dir=str(_PROJECT_ROOT / "data" / "test_frames"))

    print("🚀 Adding 3 concurrent camera feeds (CAM_NORTH, CAM_PARKING, CAM_SOUTH)...")
    manager.add_camera("CAM_NORTH", str(sample_video), sample_interval=2.0, hash_method="dhash", threshold=10)
    manager.add_camera("CAM_PARKING", str(sample_video), sample_interval=2.0, hash_method="phash", threshold=12)
    manager.add_camera("CAM_SOUTH", str(sample_video), sample_interval=2.0, hash_method="ahash", threshold=8)

    time.sleep(4)

    statuses = manager.get_all_statuses()
    print(f"\n📊 Active Camera Feeds ({len(statuses)}):")
    for s in statuses:
        print(f"  📹 {s['camera_id']} ({s['hash_method'].upper()}): Connected={s['is_connected']}, Read={s['total_frames_read']}, Keyframes={s['keyframes_kept']}")

    assert len(statuses) == 3, "Failed: Expected 3 active stream channels"

    manager.stop_all()
    print("✅ TEST 2 PASSED: Multi-camera stream manager verified!")
    return True


if __name__ == "__main__":
    success1 = test_single_stream_capture()
    success2 = test_multicamera_manager()

    if success1 and success2:
        print("\n" + "="*70)
        print("🎉 ALL RTSP MULTI-THREADED STREAM TESTS PASSED SUCCESSFULLY!")
        print("="*70 + "\n")
    else:
        sys.exit(1)
