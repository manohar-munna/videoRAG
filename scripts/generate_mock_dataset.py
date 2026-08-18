"""
scripts/generate_mock_dataset.py
---------------------------------
Generates a comprehensive multi-camera mock CCTV surveillance dataset,
extracts physical video frames, creates rich 24-hour security events,
and computes 512-D multimodal CLIP photo vectors into the FAISS index.
"""

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image
import io

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from videorag.indexing.embedder import MultimodalEmbedder
from videorag.indexing.vector_store import FAISSVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("generate_mock_dataset")


def setup_camera_data() -> List[Dict[str, Any]]:
    """Build multi-camera keyframe images and timestamped surveillance records."""
    cameras_dir = _PROJECT_ROOT / "data" / "cameras"
    cameras_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, Any]] = []

    # ----------------------------------------------------------------------
    # 1. Extract frames from sample_cctv.mp4 for CAM_01 (Main Entrance Gate)
    # ----------------------------------------------------------------------
    cam01_dir = cameras_dir / "CAM_01" / "extracted_frames"
    cam01_dir.mkdir(parents=True, exist_ok=True)
    video_file = _PROJECT_ROOT / "Video Footage" / "sample_cctv.mp4"

    cam01_events = [
        {"time": "08:14:10", "desc": "Security officer opens the main vehicle barrier gate for morning staff arrival. Clear weather conditions."},
        {"time": "08:22:45", "desc": "White delivery van (Plate partial: 7X8) pauses at the checkpoint for driver ID credential scan."},
        {"time": "09:05:12", "desc": "Pedestrian staff group wearing high-visibility safety vests enter through turnstile 1."},
        {"time": "10:30:20", "desc": "Courier driver drops off priority parcel container at security guard booth."},
        {"time": "12:15:35", "desc": "Lunchtime pedestrian traffic peak. Multiple employees exiting towards perimeter cafeteria."},
        {"time": "14:02:10", "desc": "Black SUV approaches entrance gate, security guard inspects visitor pass and grants entry."},
        {"time": "15:45:00", "desc": "Facilities maintenance team transporting boxed equipment on hand-trolley through gate."},
        {"time": "17:30:15", "desc": "Evening departure rush: multiple employee vehicles and pedestrians clearing through exit lane."},
        {"time": "19:10:40", "desc": "Evening shift security guard exchange and equipment handover log verified."},
        {"time": "21:40:00", "desc": "Nighttime illumination active. Single security guard conducting perimeter flashlight check."},
        {"time": "23:15:30", "desc": "Gate locked in overnight secure mode. Exterior floodlights operating normally."},
        {"time": "02:45:10", "desc": "Overnight perimeter patrol: security mobile unit pauses at main entrance for clock-in."}
    ]

    if video_file.exists():
        print(f"[CAM_01] Extracting {len(cam01_events)} keyframes from sample_cctv.mp4...", flush=True)
        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps

        step_sec = max(10.0, duration_sec / max(1, len(cam01_events)))
        frame_idx = 0
        cur_sec = 0.0

        for ev in cam01_events:
            cap.set(cv2.CAP_PROP_POS_MSEC, cur_sec * 1000.0)
            ret, frame = cap.read()
            if ret:
                ts_clean = ev["time"].replace(":", "_")
                img_name = f"CAM_01_{ts_clean}_{frame_idx}.jpg"
                img_path = cam01_dir / img_name
                cv2.imwrite(str(img_path), frame)

                rec = {
                    "camera": "CAM_01",
                    "timestamp": ev["time"],
                    "seconds": round(cur_sec, 2),
                    "epoch_time": round(time.time() - (86400 - cur_sec), 3),
                    "description": ev["desc"],
                    "image_path": f"data/cameras/CAM_01/extracted_frames/{img_name}",
                    "hash_hex": "0xa1b2c3d4e5f60718",
                    "motion_pct": 18.5,
                }
                all_records.append(rec)
            cur_sec += step_sec
            frame_idx += 1
        cap.release()

    # ----------------------------------------------------------------------
    # 2. Setup CAM_02 (Parking Lot & Perimeter Bay)
    # ----------------------------------------------------------------------
    cam02_dir = cameras_dir / "CAM_02" / "extracted_frames"
    cam02_dir.mkdir(parents=True, exist_ok=True)
    art_parking = Path(r"C:\Users\manoh\.gemini\antigravity-cli\brain\dc75e67d-acdc-46bd-b1ed-50a96c23f6da\cctv_parking_area_1786550812346.jpg")
    
    cam02_events = [
        {"time": "08:30:00", "desc": "Silver hatchback maneuvers into reserved parking space slot P-14."},
        {"time": "10:15:20", "desc": "Contractor pickup truck parked across diagonal loading bay stripes for short drop-off."},
        {"time": "11:45:10", "desc": "Two individuals observed conversing near rear row of parked vehicles; regular staff."},
        {"time": "13:20:45", "desc": "Red compact vehicle backing out of space P-08, clear visibility across aisle."},
        {"time": "15:10:00", "desc": "Delivery courier parked in temporary loading stall with hazard blinkers on."},
        {"time": "16:45:30", "desc": "Unattended duffel bag spotted resting against light pole LP-3; security guard dispatched."},
        {"time": "18:00:15", "desc": "Commuter vehicle row emptying out during evening shift changeover."},
        {"time": "20:30:00", "desc": "Overhead parking lot floodlights active; low ambient activity."},
        {"time": "22:15:50", "desc": "Individual wearing dark hooded sweatshirt observed lingering between vehicle rows; perimeter alert flagged."},
        {"time": "01:20:00", "desc": "Night illumination check: empty parking lanes, clear line of sight to perimeter fence."}
    ]

    print(f"[CAM_02] Setting up {len(cam02_events)} parking lot keyframes...", flush=True)
    for idx, ev in enumerate(cam02_events, 1):
        ts_clean = ev["time"].replace(":", "_")
        img_name = f"CAM_02_{ts_clean}_{idx}.jpg"
        target_img = cam02_dir / img_name

        if art_parking.exists():
            shutil.copy2(art_parking, target_img)
        else:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            dummy[:] = (40, 45, 50)
            cv2.putText(dummy, f"CAM_02 PARKING | {ev['time']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
            cv2.imwrite(str(target_img), dummy)

        rec = {
            "camera": "CAM_02",
            "timestamp": ev["time"],
            "seconds": idx * 600.0,
            "epoch_time": round(time.time() - (86400 - idx * 600), 3),
            "description": ev["desc"],
            "image_path": f"data/cameras/CAM_02/extracted_frames/{img_name}",
            "hash_hex": "0xb2c3d4e5f6071829",
            "motion_pct": 24.2,
        }
        all_records.append(rec)

    # ----------------------------------------------------------------------
    # 3. Setup CAM_03 (Lobby Reception & Secure Turnstiles)
    # ----------------------------------------------------------------------
    cam03_dir = cameras_dir / "CAM_03" / "extracted_frames"
    cam03_dir.mkdir(parents=True, exist_ok=True)
    art_lobby = Path(r"C:\Users\manoh\.gemini\antigravity-cli\brain\dc75e67d-acdc-46bd-b1ed-50a96c23f6da\cctv_lobby_1786550835392.jpg")

    cam03_events = [
        {"time": "08:45:00", "desc": "Morning visitor registration desk busy with incoming client badge issuance."},
        {"time": "10:00:15", "desc": "Visitor in blue blazer badges through electronic speed gate 2 to elevator bank."},
        {"time": "11:30:40", "desc": "Janitorial staff mopping central lobby marble floor; safety cone deployed."},
        {"time": "13:00:00", "desc": "Executive guest greeted at reception desk and escorted by department host."},
        {"time": "14:35:20", "desc": "Guest seated in waiting lounge area reading document; normal lobby activity."},
        {"time": "16:10:50", "desc": "Package delivery handoff to front reception coordinator."},
        {"time": "17:50:00", "desc": "Evening departure through automated badge turnstiles."},
        {"time": "19:30:10", "desc": "Lobby overhead lights dimmed to night energy conservation mode."},
        {"time": "21:00:00", "desc": "Overnight guard check-in at front reception desk terminal."}
    ]

    print(f"[CAM_03] Setting up {len(cam03_events)} lobby reception keyframes...", flush=True)
    for idx, ev in enumerate(cam03_events, 1):
        ts_clean = ev["time"].replace(":", "_")
        img_name = f"CAM_03_{ts_clean}_{idx}.jpg"
        target_img = cam03_dir / img_name

        if art_lobby.exists():
            shutil.copy2(art_lobby, target_img)
        else:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            dummy[:] = (55, 60, 65)
            cv2.putText(dummy, f"CAM_03 LOBBY | {ev['time']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
            cv2.imwrite(str(target_img), dummy)

        rec = {
            "camera": "CAM_03",
            "timestamp": ev["time"],
            "seconds": idx * 750.0,
            "epoch_time": round(time.time() - (86400 - idx * 750), 3),
            "description": ev["desc"],
            "image_path": f"data/cameras/CAM_03/extracted_frames/{img_name}",
            "hash_hex": "0xc3d4e5f60718293a",
            "motion_pct": 14.8,
        }
        all_records.append(rec)

    # ----------------------------------------------------------------------
    # 4. Setup CAM_04 (Warehouse & Loading Dock)
    # ----------------------------------------------------------------------
    cam04_dir = cameras_dir / "CAM_04" / "extracted_frames"
    cam04_dir.mkdir(parents=True, exist_ok=True)
    art_warehouse = Path(r"C:\Users\manoh\.gemini\antigravity-cli\brain\dc75e67d-acdc-46bd-b1ed-50a96c23f6da\cctv_warehouse_1786550858969.jpg")

    cam04_events = [
        {"time": "07:30:00", "desc": "Forklift operator staging timber shipping pallets in aisle 4."},
        {"time": "09:15:30", "desc": "Articulated freight lorry docks at bay door 3; roll-up door opened."},
        {"time": "11:00:00", "desc": "Warehouse inventory scan with handheld barcode reader along pallet racks."},
        {"time": "13:45:20", "desc": "Two warehouse technicians inspecting hydraulic dock leveler mechanism."},
        {"time": "15:30:10", "desc": "Loading cargo cartons onto outgoing distribution transport vehicle."},
        {"time": "18:00:00", "desc": "Roll-up shutter door 3 lowered and locked at end of logistics shift."},
        {"time": "20:45:00", "desc": "Interior motion sensors active; high-bay LED lighting reduced to 30% intensity."},
        {"time": "23:30:00", "desc": "Night infrared camera view: clear unobstructed storage aisles, all dock doors secured."}
    ]

    print(f"[CAM_04] Setting up {len(cam04_events)} warehouse keyframes...", flush=True)
    for idx, ev in enumerate(cam04_events, 1):
        ts_clean = ev["time"].replace(":", "_")
        img_name = f"CAM_04_{ts_clean}_{idx}.jpg"
        target_img = cam04_dir / img_name

        if art_warehouse.exists():
            shutil.copy2(art_warehouse, target_img)
        else:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            dummy[:] = (35, 40, 45)
            cv2.putText(dummy, f"CAM_04 WAREHOUSE | {ev['time']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
            cv2.imwrite(str(target_img), dummy)

        rec = {
            "camera": "CAM_04",
            "timestamp": ev["time"],
            "seconds": idx * 900.0,
            "epoch_time": round(time.time() - (86400 - idx * 900), 3),
            "description": ev["desc"],
            "image_path": f"data/cameras/CAM_04/extracted_frames/{img_name}",
            "hash_hex": "0xd4e5f60718293a4b",
            "motion_pct": 31.0,
        }
        all_records.append(rec)

    # ----------------------------------------------------------------------
    # 5. Include Existing Real Events from CAM_3000 & CAM_4000
    # ----------------------------------------------------------------------
    for cam_name in ["CAM_3000", "CAM_4000"]:
        ev_file = cameras_dir / cam_name / "events.json"
        if ev_file.exists():
            try:
                with open(ev_file, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
                    if isinstance(existing, list):
                        all_records.extend(existing)
            except Exception:
                pass

    # Save per-camera events.json
    for cam_id in ["CAM_01", "CAM_02", "CAM_03", "CAM_04"]:
        c_events = [r for r in all_records if r.get("camera") == cam_id]
        with open(cameras_dir / cam_id / "events.json", "w", encoding="utf-8") as fh:
            json.dump(c_events, fh, indent=2, ensure_ascii=False)

    # Save master real_cctv_events.json
    master_file = _PROJECT_ROOT / "data" / "real_cctv_events.json"
    with open(master_file, "w", encoding="utf-8") as fh:
        json.dump(all_records, fh, indent=2, ensure_ascii=False)

    # Update cameras_registry.json
    reg_file = _PROJECT_ROOT / "data" / "cameras_registry.json"
    registry = {
        "CAM_01": {
            "camera_id": "CAM_01",
            "name": "Main Entrance Barrier Gate",
            "location": "North Access Perimeter",
            "type": "video_file",
            "stream_url": "/video/sample_cctv.mp4",
            "sample_interval": 10.0,
            "hash_method": "dhash",
            "threshold": 10,
            "status": "active"
        },
        "CAM_02": {
            "camera_id": "CAM_02",
            "name": "Visitor & Staff Parking Lot",
            "location": "West Parking Bays",
            "type": "snapshot",
            "stream_url": "",
            "sample_interval": 15.0,
            "hash_method": "dhash",
            "threshold": 10,
            "status": "active"
        },
        "CAM_03": {
            "camera_id": "CAM_03",
            "name": "Central Lobby Reception",
            "location": "Main Building Foyer",
            "type": "snapshot",
            "stream_url": "",
            "sample_interval": 15.0,
            "hash_method": "dhash",
            "threshold": 10,
            "status": "active"
        },
        "CAM_04": {
            "camera_id": "CAM_04",
            "name": "Warehouse Loading Dock 3",
            "location": "Logistics & Freight Bay",
            "type": "snapshot",
            "stream_url": "",
            "sample_interval": 15.0,
            "hash_method": "dhash",
            "threshold": 10,
            "status": "active"
        },
        "CAM_3000": {
            "camera_id": "CAM_3000",
            "name": "Downtown Commercial Intersection",
            "location": "Main Street & 5th Ave",
            "type": "youtube_stream",
            "stream_url": "https://www.youtube.com/watch?v=1EiC9bvVGnk",
            "sample_interval": 5.0,
            "hash_method": "dhash",
            "threshold": 10,
            "status": "paused"
        },
        "CAM_4000": {
            "camera_id": "CAM_4000",
            "name": "Coastal Promenade Walkway",
            "location": "Beachfront Public Zone",
            "type": "youtube_stream",
            "stream_url": "https://www.youtube.com/live/EO_1LWqsCNE?si=zGu_teV2HCn5EeWi",
            "sample_interval": 5.0,
            "hash_method": "dhash",
            "threshold": 10,
            "status": "paused"
        }
    }
    with open(reg_file, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2, ensure_ascii=False)

    print(f"  [OK] Total CCTV surveillance records: {len(all_records)} across 6 cameras.", flush=True)
    return all_records


def build_and_index_vectors(records: List[Dict[str, Any]]) -> None:
    """Compute 512-D multimodal CLIP embeddings directly from photo frames and build FAISS index."""
    print("\n--- Computing Multimodal CLIP Photo Vectors (512-D) ---", flush=True)

    embedder = MultimodalEmbedder(model_name="clip-ViT-B-32")
    store = FAISSVectorStore(dim=512)

    embeddings_list = []
    metadata_list = []

    for idx, r in enumerate(records, start=1):
        img_p = r.get("image_path", "")
        local_img = Path(img_p)
        if not local_img.is_absolute():
            cand1 = _PROJECT_ROOT / img_p.lstrip("/")
            cand2 = _PROJECT_ROOT / "data" / img_p.lstrip("/")
            local_img = cand1 if cand1.exists() else cand2

        img_embedded = False
        if local_img.exists():
            try:
                vec = embedder.embed_image(local_img)
                embeddings_list.append(vec)
                img_embedded = True
            except Exception:
                pass

        if not img_embedded:
            doc_text = f"Camera: {r.get('camera')} | Time: {r.get('timestamp')} | Event: {r.get('description')}"
            vec = embedder.embed_query(doc_text)
            embeddings_list.append(vec)

        meta = {
            "camera": r.get("camera"),
            "timestamp": r.get("timestamp"),
            "seconds": r.get("seconds", 0.0),
            "epoch_time": r.get("epoch_time"),
            "description": r.get("description", ""),
            "image_path": str(r.get("image_path", "")).replace("\\", "/"),
            "chunk_id": f"{r.get('camera')}_{r.get('timestamp', '00_00_00').replace(':', '_')}",
            "text": f"Camera: {r.get('camera')} | Time: {r.get('timestamp')} | Event: {r.get('description')}",
        }
        metadata_list.append(meta)

        if idx % 50 == 0 or idx == len(records):
            print(f"  Encoded {idx} / {len(records)} records...", flush=True)

    embeddings_2d = np.vstack(embeddings_list).astype(np.float32)
    print(f"  [OK] Generated dense photo vectors: shape {embeddings_2d.shape}", flush=True)

    print("\n--- Populating FAISS Index & Saving to Disk ---", flush=True)
    store.add(embeddings_2d, metadata_list)
    index_save_path = _PROJECT_ROOT / "index" / "cctv_index"
    store.save(str(index_save_path))
    print(f"  [OK] Saved FAISS index ({store.size} vectors) -> {index_save_path}", flush=True)


def main():
    print("==================================================", flush=True)
    print(" VideoRAG -- High-Fidelity Mock & Real Dataset Generator", flush=True)
    print("==================================================", flush=True)
    records = setup_camera_data()
    build_and_index_vectors(records)
    print("\n[COMPLETE] All mock photo vectors and metadata successfully indexed into FAISS!", flush=True)


if __name__ == "__main__":
    main()
