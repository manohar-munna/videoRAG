"""
src/videorag/server.py
----------------------
FastAPI Web Application backend for VideoRAG.
Serves REST API endpoints for semantic search, video processing, system health,
dHash/pHash Developer Inspector, and hosts the Classic Light Web UI.
"""

import argparse
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv
load_dotenv()

# Add src to Python path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from videorag.ingestion.loader import CCTVDataLoader
from videorag.indexing.embedder import TextEmbedder
from videorag.indexing.vector_store import FAISSVectorStore
from videorag.retrieval.retriever import CCTVRetriever
from videorag.retrieval.reranker import CrossEncoderReranker, ScoreReranker
from videorag.llm.prompter import RAGPrompter, LLMClient
from videorag.evaluation.evaluator import RAGEvaluator
from videorag.ingestion.video_processor import VideoFrameExtractor
from videorag.ingestion.hash_filter import EdgeFrameFilter
from videorag.ingestion.stream_capture import MultiCameraStreamManager
from videorag.captioning.vlm_captioner import VLMCaptioner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("videorag.server")

app = FastAPI(title="VideoRAG Intelligence Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import queue
import threading
import time

import collections
import queue
import threading
import time
from videorag.ingestion.camera_registry import CameraRegistry

# Global pipeline & persistent stream manager instances
PIPELINE: Dict[str, Any] = {}
CAMERA_REGISTRY = CameraRegistry(_PROJECT_ROOT / "data" / "cameras_registry.json")

# Fair Round-Robin Auto-Indexing Queue for Real-Time Multi-Camera Ingestion
CAMERA_QUEUES: Dict[str, collections.deque] = collections.defaultdict(lambda: collections.deque(maxlen=10))
QUEUE_LOCK = threading.Lock()
QUEUE_NOTIFY = threading.Condition(QUEUE_LOCK)


def _queue_auto_index_keyframe(keyframe_meta: Dict[str, Any]) -> None:
    """Callback invoked by stream workers when a new keyframe is kept by dHash/pHash."""
    cam = keyframe_meta.get("camera")
    if not cam:
        return
    with QUEUE_NOTIFY:
        CAMERA_QUEUES[cam].append(keyframe_meta)
        QUEUE_NOTIFY.notify_all()
    logger.info("[Auto-Indexer] Queued new keyframe for %s @ %s (cam pending: %d).", 
                cam, keyframe_meta.get("timestamp"), len(CAMERA_QUEUES[cam]))


def _clear_camera_queue(cam_id: str) -> None:
    """Instantly clears pending auto-index backlog for a paused/removed camera."""
    with QUEUE_NOTIFY:
        if cam_id in CAMERA_QUEUES:
            CAMERA_QUEUES[cam_id].clear()
            logger.info("[Auto-Indexer] Cleared pending auto-index queue for %s.", cam_id)


STREAM_MANAGER = MultiCameraStreamManager(
    registry=CAMERA_REGISTRY, 
    on_keyframe_callback=_queue_auto_index_keyframe
)


def _auto_index_worker_loop() -> None:
    """
    Background worker thread: round-robins across active camera queues so no single camera
    can starve other cameras, captions keyframes with Qwen3-VL, and updates per-camera JSON.
    """
    captioner = None
    current_cam_idx = 0

    while True:
        try:
            keyframe = None
            with QUEUE_NOTIFY:
                # Wait until at least one camera has a pending keyframe
                while not any(len(q) > 0 for q in CAMERA_QUEUES.values()):
                    QUEUE_NOTIFY.wait(timeout=1.0)

                # Fair round-robin across cameras that have pending frames
                available_cams = [c for c, q in CAMERA_QUEUES.items() if len(q) > 0]
                if available_cams:
                    current_cam_idx = current_cam_idx % len(available_cams)
                    chosen_cam = available_cams[current_cam_idx]
                    keyframe = CAMERA_QUEUES[chosen_cam].popleft()
                    current_cam_idx = (current_cam_idx + 1) % len(available_cams)

            if not keyframe:
                continue

            cam_id = keyframe.get("camera")
            ts_str = keyframe.get("timestamp")
            img_path = keyframe.get("image_path")

            if not img_path or not Path(img_path).exists():
                continue            # Direct multimodal visual embedding (~25ms, no slow VLM text generation in background loop)
            clean_p = str(img_path).replace("\\", "/")
            emb = None
            if PIPELINE.get("embedder"):
                try:
                    emb = PIPELINE["embedder"].embed_image(img_path)
                except Exception as exc:
                    logger.warning("[Auto-Indexer] Error embedding image %s: %s", img_path, exc)

            desc = f"Surveillance keyframe captured by {cam_id} at {ts_str}."

            # 1. Update per-camera isolated events JSON
            cam_dir = _PROJECT_ROOT / "data" / "cameras" / cam_id
            cam_dir.mkdir(parents=True, exist_ok=True)
            cam_json_file = cam_dir / "events.json"

            existing_cam = []
            if cam_json_file.exists():
                try:
                    with open(cam_json_file, "r", encoding="utf-8") as fh:
                        existing_cam = json.load(fh)
                except Exception:
                    existing_cam = []

            # Check if this exact frame is already in per-camera json
            if not any(str(r.get("image_path", "")).replace("\\", "/") == clean_p for r in existing_cam):
                new_rec = {
                    "camera": cam_id,
                    "timestamp": ts_str,
                    "seconds": keyframe.get("seconds", 0.0),
                    "epoch_time": keyframe.get("epoch_time", round(time.time(), 3)),
                    "description": desc,
                    "image_path": clean_p,
                    "hash_hex": keyframe.get("hash_hex"),
                    "motion_pct": keyframe.get("motion_pct"),
                }
                existing_cam.append(new_rec)
                with open(cam_json_file, "w", encoding="utf-8") as fh:
                    json.dump(existing_cam, fh, indent=2, ensure_ascii=False)

                # 2. Update master real_cctv_events.json
                out_file = _PROJECT_ROOT / "data" / "real_cctv_events.json"
                all_events = []
                for c_dir in (_PROJECT_ROOT / "data" / "cameras").glob("*"):
                    e_f = c_dir / "events.json"
                    if e_f.exists():
                        try:
                            with open(e_f, "r", encoding="utf-8") as fh:
                                all_events.extend(json.load(fh))
                        except Exception:
                            pass

                seen_k = set()
                deduped = []
                for r in all_events:
                    k = (r.get("camera"), r.get("timestamp"), r.get("image_path", ""))
                    if k not in seen_k:
                        seen_k.add(k)
                        deduped.append(r)

                with open(out_file, "w", encoding="utf-8") as fh:
                    json.dump(deduped, fh, indent=2, ensure_ascii=False)

                # 3. Dynamic in-memory FAISS vector indexing (Instant Image Vector)
                if PIPELINE.get("vector_store") and emb is not None:
                    import numpy as np
                    emb_2d = np.ascontiguousarray(emb.reshape(1, -1), dtype=np.float32)
                    meta = {
                        "camera": cam_id,
                        "timestamp": ts_str,
                        "seconds": keyframe.get("seconds", 0.0),
                        "epoch_time": keyframe.get("epoch_time", round(time.time(), 3)),
                        "description": desc,
                        "text": f"Camera: {cam_id} | Time: {ts_str}",
                        "image_path": clean_p,
                        "chunk_id": f"{cam_id}_{ts_str.replace(':', '_')}",
                    }
                    PIPELINE["vector_store"].add(emb_2d, [meta])
                    idx_path = _PROJECT_ROOT / "index" / "cctv_index"
                    PIPELINE["vector_store"].save(str(idx_path))
                    logger.info("[Auto-Indexer] [%s] Instantly indexed visual keyframe @ %s in ~25ms! Vector count: %d", 
                                cam_id, ts_str, PIPELINE["vector_store"].size)

        except Exception as exc:
            logger.error("[Auto-Indexer] Error during keyframe auto-indexing: %s", exc, exc_info=True)
            time.sleep(1)


# Start background auto-indexer worker thread
_auto_index_thread = threading.Thread(target=_auto_index_worker_loop, daemon=True, name="AutoIndexWorker")
_auto_index_thread.start()


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    rerank_top_k: Optional[int] = 5
    camera_filter: Optional[str] = None


class SmartProcessRequest(BaseModel):
    video_path: Optional[str] = None
    camera_id: Optional[str] = "CAM_01"
    sample_interval: Optional[float] = 15.0
    enable_hash_filter: Optional[bool] = True
    hash_method: Optional[str] = "dhash"
    threshold: Optional[int] = 10
    run_vlm_captioning: Optional[bool] = True


class AddStreamRequest(BaseModel):
    camera_id: str
    feed_type: str = "youtube_stream"  # 'youtube_stream' | 'rtsp_stream' | 'video_file'
    stream_url: str
    name: Optional[str] = None
    location: Optional[str] = "Perimeter Gate"
    sample_interval: Optional[float] = 5.0
    hash_method: Optional[str] = "dhash"
    threshold: Optional[int] = 10
    enable_hash_filter: Optional[bool] = True


class StreamControlRequest(BaseModel):
    camera_id: str


# In-memory pipeline objects
PIPELINE: Dict[str, Any] = {}

# In-memory real-time Edge Frame Inspector metrics buffer
DEV_INSPECTOR_STATE = {
    "kpis": {
        "total_frames": 55,
        "keyframes_kept": 48,
        "frames_skipped": 7,
        "llm_compute_saved_pct": 12.7,
        "method": "dhash",
        "threshold": 10,
    },
    "audit_trail": [],
}


def init_pipeline(config_path: str = "config/config.yaml") -> None:
    """Initialize multimodal retriever, reranker, vector store, VLM reasoner, and LLM clients."""
    import yaml
    cfg_file = Path(config_path)
    if not cfg_file.is_absolute():
        cfg_file = _PROJECT_ROOT / cfg_file

    with open(cfg_file, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    cfg_idx = config.get("indexing", {})
    cfg_ret = config.get("retrieval", {})
    cfg_llm = config.get("llm", {})

    model_name = cfg_idx.get("model_name", "MobileCLIP-S2")
    model_path = cfg_idx.get("model_path")
    index_path = cfg_idx.get("index_save_path", "index/cctv_index")

    from videorag.indexing.embedder import MultimodalEmbedder
    embedder = MultimodalEmbedder(model_name=model_name, model_path=model_path)
    store = FAISSVectorStore(dim=embedder.dimension)

    idx_path = Path(index_path)
    if not idx_path.is_absolute():
        idx_path = _PROJECT_ROOT / idx_path

    if idx_path.with_suffix(".faiss").exists():
        try:
            store.load(str(idx_path))
            if store._index.d != embedder.dimension:
                logger.warning(
                    "Existing FAISS index dimension (%d) does not match embedder dimension (%d). Initializing fresh %d-D store.",
                    store._index.d, embedder.dimension, embedder.dimension
                )
                store = FAISSVectorStore(dim=embedder.dimension)
            else:
                logger.info("Loaded FAISS index with %d vectors", store.size)
        except Exception as exc:
            logger.warning("Failed to load index (%s). Creating fresh %d-D store.", exc, embedder.dimension)
            store = FAISSVectorStore(dim=embedder.dimension)
    else:
        logger.warning("FAISS index not found at %s. Creating empty %d-D store.", idx_path, embedder.dimension)

    retriever = CCTVRetriever(vector_store=store, embedder=embedder)

    try:
        reranker = CrossEncoderReranker()
        logger.info("Cross-encoder reranker loaded.")
    except Exception as exc:
        logger.warning("Cross-encoder failed (%s), using score fallback.", exc)
        reranker = ScoreReranker()

    captioner = VLMCaptioner(
        backend=cfg_llm.get("backend", "local"),
        model=cfg_llm.get("model", "Local LLM 3VL 4Q/Qwen3VL-4B-Instruct-Q4_K_M.gguf"),
        api_key=cfg_llm.get("api_key"),
        base_url=cfg_llm.get("base_url"),
    )

    llm_client = LLMClient(
        backend=cfg_llm.get("backend", "local"),
        model=cfg_llm.get("model", "Local LLM 3VL 4Q/Qwen3VL-4B-Instruct-Q4_K_M.gguf"),
        api_key=cfg_llm.get("api_key"),
        base_url=cfg_llm.get("base_url"),
    )

    prompter = RAGPrompter()
    evaluator = RAGEvaluator()

    PIPELINE["config"] = config
    PIPELINE["embedder"] = embedder
    PIPELINE["vector_store"] = store
    PIPELINE["retriever"] = retriever
    PIPELINE["reranker"] = reranker
    PIPELINE["captioner"] = captioner
    PIPELINE["llm_client"] = llm_client
    PIPELINE["prompter"] = prompter
    PIPELINE["evaluator"] = evaluator
    PIPELINE["config_path"] = str(cfg_file)

    # Initialize active camera streams from persistent registry
    STREAM_MANAGER.initialize_from_registry()


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def get_health():
    """Return backend health and system info."""
    import time
    store = PIPELINE.get("vector_store")
    llm = PIPELINE.get("llm_client")
    embedder = PIPELINE.get("embedder")
    return {
        "status": "online",
        "server_time": round(time.time(), 3),
        "vector_count": store.size if store else 0,
        "vector_dimension": embedder.dimension if embedder else 512,
        "embedder_model": embedder.model_name if embedder else "clip-ViT-B-32",
        "llm_backend": llm.backend if llm else "unknown",
        "llm_model": llm.model if llm else "unknown",
        "reranker": PIPELINE.get("reranker").__class__.__name__ if PIPELINE.get("reranker") else "none",
    }


@app.get("/api/events")
def get_events(camera: Optional[str] = None, detailed: bool = False):
    """Return combined CCTV events dataset with dynamic camera discovery and optional filtering."""
    data_path = _PROJECT_ROOT / "data" / "real_cctv_events.json"
    records = []
    
    # 1. Read from master real_cctv_events.json if available
    if data_path.exists():
        try:
            with open(data_path, "r", encoding="utf-8") as fh:
                records = json.load(fh)
        except Exception as exc:
            logger.warning("Could not load real_cctv_events.json: %s", exc)
            records = []

    # 2. Dynamic discovery: Scan data/cameras/*/events.json for camera events
    discovered_records = []
    cameras_dir = _PROJECT_ROOT / "data" / "cameras"
    if cameras_dir.exists():
        for cam_folder in cameras_dir.glob("*"):
            if cam_folder.is_dir():
                events_file = cam_folder / "events.json"
                if events_file.exists():
                    try:
                        with open(events_file, "r", encoding="utf-8") as ef:
                            cam_events = json.load(ef)
                            if isinstance(cam_events, list):
                                discovered_records.extend(cam_events)
                    except Exception as ef_exc:
                        logger.warning("Error reading %s: %s", events_file, ef_exc)

    # 3. Deduplicate by (camera, timestamp, image_path)
    combined = records + discovered_records
    seen = set()
    deduped = []
    for r in combined:
        k = (r.get("camera"), r.get("timestamp"), r.get("image_path", ""))
        if k not in seen:
            seen.add(k)
            deduped.append(r)

    # 4. Filter by camera if requested
    if camera:
        filtered = [r for r in deduped if r.get("camera", "").upper() == camera.upper()]
    else:
        filtered = deduped

    # 5. Return detailed dataset telemetry if requested
    if detailed:
        all_cams = sorted(list(set(r.get("camera", "") for r in deduped if r.get("camera"))))
        file_size = data_path.stat().st_size if data_path.exists() else 0
        return {
            "total_count": len(records),
            "filtered_count": len(filtered),
            "cameras": all_cams,
            "file_size_bytes": file_size,
            "dataset_path": "data/real_cctv_events.json",
        }

    return filtered


@app.post("/api/search")
def search_cctv(req: SearchRequest):
    """Execute semantic multimodal query against CCTV video index and perform multi-frame forensic reasoning."""
    if not PIPELINE.get("retriever"):
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    retriever: CCTVRetriever = PIPELINE["retriever"]
    captioner: Optional[VLMCaptioner] = PIPELINE.get("captioner")
    evaluator: RAGEvaluator = PIPELINE["evaluator"]

    top_k = req.top_k or 5
    context_window = 2  # ±2 neighbouring frames (5 frames total per episode)

    import time
    t0 = time.time()
    query_vec = PIPELINE["embedder"].embed_query(req.query)
    t1 = time.time()

    # 1. Retrieve episodes with temporal context window expansion
    episodes = retriever.retrieve_with_context(
        req.query,
        top_k=top_k,
        context_window=context_window,
        camera_filter=req.camera_filter
    )
    t2 = time.time()

    # 2. Multi-Frame Forensic Reasoning via VLM
    answer = ""
    storyboard = []
    if episodes:
        top_episode = episodes[0]
        if captioner is None:
            cfg_llm = PIPELINE.get("config", {}).get("llm", {})
            captioner = VLMCaptioner(
                backend=cfg_llm.get("backend", "local"),
                model=cfg_llm.get("model", "Local LLM 3VL 4Q/Qwen3VL-4B-Instruct-Q4_K_M.gguf"),
                api_key=cfg_llm.get("api_key"),
                base_url=cfg_llm.get("base_url"),
            )
            PIPELINE["captioner"] = captioner

        logger.info("[Search] Running multi-frame forensic reasoning on top episode (%s, %s)...",
                    top_episode.get("camera"), top_episode.get("time_range"))
        answer = captioner.reason_over_episode(req.query, top_episode)

        # Build chronological storyboard for the top episode
        for f in top_episode.get("frames", []):
            img_p = f.get("image_path", "")
            if img_p:
                clean_img = img_p.replace("\\", "/")
                if "data/" in clean_img:
                    img_p = "/data/" + clean_img.split("data/", 1)[-1].lstrip("/")
            storyboard.append({
                "camera": top_episode.get("camera", "CAM_01"),
                "image_path": img_p,
                "timestamp": f.get("timestamp", "00:00:00"),
                "seconds": f.get("seconds", 0.0),
                "epoch_time": f.get("epoch_time"),
                "is_anchor": f.get("is_anchor", False),
                "score": round(float(f.get("score", 0.0)), 4),
            })
    else:
        answer = "No matching CCTV surveillance moments found for this query in the vector index."
    t3 = time.time()

    # 3. Format primary result items for evidence list and map
    items = []
    for rank, ep in enumerate(episodes, start=1):
        meta = ep.get("metadata", {})
        cam = ep.get("camera", "CAM_01")
        ts = ep.get("anchor_timestamp", meta.get("timestamp", "00:00:00"))
        time_range = ep.get("time_range", ts)

        secs = 0
        try:
            parts = [int(p) for p in ts.split(":")]
            if len(parts) == 3:
                secs = parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2:
                secs = parts[0] * 60 + parts[1]
        except Exception:
            secs = 0

        img_p = ep.get("anchor_image", meta.get("image_path", ""))
        if img_p:
            clean_img = img_p.replace("\\", "/")
            if "data/" in clean_img:
                img_p = "/data/" + clean_img.split("data/", 1)[-1].lstrip("/")
        elif cam:
            cam_dir = _PROJECT_ROOT / "data" / "cameras" / cam / "extracted_frames"
            if cam_dir.exists():
                jpgs = sorted(cam_dir.glob("*.jpg"), key=os.path.getmtime, reverse=True)
                if jpgs:
                    img_p = f"/data/cameras/{cam}/extracted_frames/{jpgs[0].name}"

        cam_info = CAMERA_REGISTRY.get(cam)
        feed_type = cam_info.get("type", "snapshot") if cam_info else "snapshot"
        feed_url = ""
        embed_url = ""
        if cam_info and cam_info.get("type") == "video_file":
            feed_type = "video_file"
            feed_url = "/video/sample_cctv.mp4"
        elif cam_info and cam_info.get("type") == "youtube_stream":
            feed_type = "youtube_stream"
            feed_url = cam_info.get("stream_url", "")
            import re
            yt_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", feed_url)
            vid_id = yt_match.group(1) if yt_match else "1EiC9bvVGnk"
            embed_url = f"https://www.youtube-nocookie.com/embed/{vid_id}?autoplay=1&mute=1"
        epoch_time = meta.get("epoch_time")

        items.append({
            "rank": rank,
            "camera": cam,
            "timestamp": ts,
            "time_range": time_range,
            "seconds": secs,
            "epoch_time": epoch_time,
            "description": f"Episode from {cam} ({time_range}) matching '{req.query}' [Anchor: {ts}]",
            "image_path": img_p,
            "feed_type": feed_type,
            "feed_url": feed_url,
            "embed_url": embed_url,
            "faiss_score": round(float(ep.get("score", 0.0)), 4),
            "rerank_score": round(float(ep.get("score", 0.0)), 4),
            "frame_count": ep.get("frame_count", 1),
        })

    # 4. Evaluation
    stop_words = {"the", "a", "an", "is", "was", "were", "are", "in", "at", "on", "of", "to", "any", "did", "do", "what", "when", "where", "who", "how", "there"}
    keywords = [w.strip("?.,!").lower() for w in req.query.split() if w.lower() not in stop_words and len(w) > 2]
    eval_result = evaluator.full_evaluation(req.query, episodes, answer, keywords)

    # Detailed Vector & Pipeline Debugging Trace
    vec_sample = [round(float(val), 4) for val in query_vec[:12]]
    vec_norm = round(float(sum(v*v for v in query_vec)**0.5), 4)

    debug_trace = {
        "query_vector_dim": len(query_vec),
        "query_vector_norm": vec_norm,
        "query_vector_sample": vec_sample,
        "faiss_indexed_vectors": PIPELINE["vector_store"].size if PIPELINE.get("vector_store") else 0,
        "episodes_retrieved": len(episodes),
        "timings_ms": {
            "query_embedding_ms": round((t1 - t0) * 1000, 2),
            "temporal_retrieval_ms": round((t2 - t1) * 1000, 2),
            "vlm_reasoning_ms": round((t3 - t2) * 1000, 2),
            "total_ms": round((t3 - t0) * 1000, 2),
        }
    }

    return {
        "query": req.query,
        "answer": answer,
        "storyboard": storyboard,
        "episodes": episodes,
        "results": items,
        "evaluation": eval_result,
        "total_retrieved": len(episodes),
        "debug_trace": debug_trace,
    }


@app.post("/api/process_video_smart")
def process_video_smart(req: SmartProcessRequest):
    """
    Run smart video processing with dHash/pHash frame filtering,
    optional VLM keyframe captioning, FAISS index rebuilding, and in-memory pipeline reloading.
    """
    video_p = req.video_path or str(_PROJECT_ROOT / "Video Footage" / "sample_cctv.mp4")
    video_file = Path(video_p)
    if not video_file.is_absolute():
        video_file = _PROJECT_ROOT / video_file

    if not video_file.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {video_p}")

    # 1. Extract frames with EdgeFrameFilter
    extractor = VideoFrameExtractor(output_dir=str(_PROJECT_ROOT / "data" / "extracted_frames"))
    hash_filter = EdgeFrameFilter(method=req.hash_method or "dhash", threshold=req.threshold or 10) if req.enable_hash_filter else None

    result = extractor.extract_frames(
        video_path=str(video_file),
        camera_id=req.camera_id or "CAM_01",
        sample_interval=req.sample_interval or 15.0,
        hash_filter=hash_filter,
    )

    LATEST_HASH_AUDIT["stats"] = result["filter_stats"]
    LATEST_HASH_AUDIT["audit_trail"] = result["audit_trail"]

    extracted_frames = result["extracted_frames"]

    # 2. VLM Captioning & Indexing if requested
    if req.run_vlm_captioning and extracted_frames:
        logger.info("Dev Mode: Running VLM captioning on %d keyframes...", len(extracted_frames))
        captioner = VLMCaptioner(backend="local")
        records = captioner.caption_batch(extracted_frames, show_progress=False)

        out_file = _PROJECT_ROOT / "data" / "real_cctv_events.json"
        with open(out_file, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, ensure_ascii=False)

        # Rebuild FAISS Index
        from scripts.index import run_indexing
        cfg_file = PIPELINE.get("config_path", str(_PROJECT_ROOT / "config" / "config.yaml"))
        run_indexing(config_path=cfg_file, data_path=str(out_file))

        # Reload in-memory PIPELINE FAISS index
        init_pipeline(config_path=cfg_file)
        logger.info("Dev Mode: In-memory pipeline reloaded with %d new keyframe vectors.", PIPELINE["vector_store"].size)

    return {
        "status": "success",
        "extracted_count": len(extracted_frames),
        "skipped_count": result["skipped_count"],
        "total_sampled": result["total_sampled"],
        "filter_stats": result["filter_stats"],
        "audit_trail": result["audit_trail"],
        "new_vector_count": PIPELINE["vector_store"].size if PIPELINE.get("vector_store") else 0,
    }


@app.get("/api/hash_audit")
def get_hash_audit():
    """Return the latest frame hashing audit log for Developer Mode UI inspection."""
    return LATEST_HASH_AUDIT


@app.post("/api/streams/add")
def add_camera_stream(req: AddStreamRequest):
    """Add and start an async multi-threaded RTSP camera stream channel."""
    stream = STREAM_MANAGER.add_camera(
        camera_id=req.camera_id,
        stream_url=req.stream_url,
        sample_interval=req.sample_interval or 5.0,
        hash_method=req.hash_method or "dhash",
        threshold=req.threshold or 10,
    )
    return {"status": "started", "camera_id": req.camera_id, "stream_info": stream.get_status()}


@app.get("/api/streams/status")
def get_streams_status():
    """Return health metrics and stats for all registered multi-camera streams."""
    return {"active_streams": STREAM_MANAGER.get_all_statuses()}


@app.post("/api/streams/pause")
def pause_camera_stream(req: RemoveStreamRequest):
    """Pause stream extraction while keeping camera registered in system and searchable."""
    _clear_camera_queue(req.camera_id)
    success = STREAM_MANAGER.pause_camera(req.camera_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Camera '{req.camera_id}' not found.")
    return {"status": "paused", "camera_id": req.camera_id}


@app.post("/api/streams/resume")
def resume_camera_stream(req: RemoveStreamRequest):
    """Resume paused stream extraction worker threads."""
    success = STREAM_MANAGER.resume_camera(req.camera_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Camera '{req.camera_id}' not found.")
    return {"status": "resumed", "camera_id": req.camera_id}


@app.post("/api/streams/remove")
def remove_camera_stream(req: RemoveStreamRequest):
    """Stop and completely remove camera stream from registry."""
    _clear_camera_queue(req.camera_id)
    success = STREAM_MANAGER.remove_camera(req.camera_id)
    return {"status": "removed", "camera_id": req.camera_id}


@app.post("/api/streams/index_now")
def index_stream_keyframes(req: RemoveStreamRequest):
    """
    Run VLM visual captioning and FAISS vector indexing on all captured keyframes
    for the specified camera stream (e.g. 'CAM_3000').
    """
    cam_id = req.camera_id
    if cam_id not in STREAM_MANAGER.streams:
        raise HTTPException(status_code=404, detail=f"Camera '{cam_id}' not found.")

    stream = STREAM_MANAGER.streams[cam_id]
    keyframes = list(stream.extracted_keyframes)

    if not keyframes:
        raise HTTPException(status_code=400, detail=f"No keyframes extracted yet for camera '{cam_id}'.")

    logger.info("Indexing %d keyframes for camera '%s' using Qwen3-VL VLM...", len(keyframes), cam_id)

    # 1. VLM Captioning
    captioner = VLMCaptioner(backend="local")
    new_records = captioner.caption_batch(keyframes, show_progress=False)

    # 2. Save isolated per-camera JSON data at data/cameras/<camera_id>/events.json
    cam_dir = _PROJECT_ROOT / "data" / "cameras" / cam_id
    cam_dir.mkdir(parents=True, exist_ok=True)
    cam_json_file = cam_dir / "events.json"

    existing_cam_records = []
    if cam_json_file.exists():
        try:
            with open(cam_json_file, "r", encoding="utf-8") as fh:
                existing_cam_records = json.load(fh)
        except Exception:
            existing_cam_records = []

    cam_paths = {str(r.get("image_path", "")).replace("\\", "/") for r in existing_cam_records if "image_path" in r}
    unique_cam_new = [r for r in new_records if str(r.get("image_path", "")).replace("\\", "/") not in cam_paths]
    updated_cam_records = existing_cam_records + unique_cam_new

    with open(cam_json_file, "w", encoding="utf-8") as fh:
        json.dump(updated_cam_records, fh, indent=2, ensure_ascii=False)

    # 3. Update master combined real_cctv_events.json by aggregating all per-camera event files
    out_file = _PROJECT_ROOT / "data" / "real_cctv_events.json"
    all_events = []
    for cam_dir in (_PROJECT_ROOT / "data" / "cameras").glob("*"):
        events_f = cam_dir / "events.json"
        if events_f.exists():
            try:
                with open(events_f, "r", encoding="utf-8") as fh:
                    all_events.extend(json.load(fh))
            except Exception:
                pass

    seen_keys = set()
    deduped_master = []
    for r in all_events:
        key = (r.get("camera"), r.get("timestamp"), r.get("description", "")[:30])
        if key not in seen_keys:
            seen_keys.add(key)
            deduped_master.append(r)

    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(deduped_master, fh, indent=2, ensure_ascii=False)

    # 4. Rebuild FAISS Vector Index
    from scripts.index import run_indexing
    cfg_file = PIPELINE.get("config_path", str(_PROJECT_ROOT / "config" / "config.yaml"))
    run_indexing(config_path=cfg_file, data_path=str(out_file))

    # 5. Reload in-memory FAISS store
    init_pipeline(config_path=cfg_file)
    logger.info("Successfully indexed %d new keyframes for %s. Total vectors: %d", len(unique_cam_new), cam_id, PIPELINE["vector_store"].size)

    return {
        "status": "success",
        "camera_id": cam_id,
        "indexed_count": len(unique_cam_new),
        "total_vectors": PIPELINE["vector_store"].size if PIPELINE.get("vector_store") else 0,
        "camera_json_path": str(cam_json_file),
    }


@app.get("/api/cameras")
def get_camera_list():
    """Return list of all dynamically registered cameras."""
    registered = [c["camera_id"] for c in CAMERA_REGISTRY.get_all()]
    # Add any cameras that have folders with events
    for cam_dir in (_PROJECT_ROOT / "data" / "cameras").glob("*"):
        if cam_dir.is_dir() and (cam_dir / "events.json").exists():
            registered.append(cam_dir.name)
    return {"cameras": sorted(list(set(registered)))}


@app.get("/api/cameras/feeds")
def get_camera_feeds():
    """Return live and recorded surveillance feeds from persistent CameraRegistry."""
    feeds = []
    registered_cams = CAMERA_REGISTRY.get_all()

    for cfg in registered_cams:
        cam_id = cfg["camera_id"]
        cam_type = cfg.get("type", "rtsp_stream")
        stream_url = cfg.get("stream_url", "")
        status = cfg.get("status", "stopped")

        # Find latest frame snapshot if available
        cam_frames_dir = _PROJECT_ROOT / "data" / "cameras" / cam_id / "extracted_frames"
        preview_img = ""
        if cam_frames_dir.exists():
            jpgs = sorted(cam_frames_dir.glob("*.jpg"), key=os.path.getmtime, reverse=True)
            if jpgs:
                preview_img = f"/data/cameras/{cam_id}/extracted_frames/{jpgs[0].name}"

        is_connected = False
        fps_val = 30.0
        progress_pct = None
        total_duration_sec = None
        current_pos_sec = None
        keyframes_count = 0
        frames_read = 0

        stream = STREAM_MANAGER.streams.get(cam_id)
        if stream:
            st = stream.get_status()
            is_connected = st.get("is_connected", False)
            fps_val = st.get("fps", 30.0)
            progress_pct = st.get("progress_pct")
            total_duration_sec = st.get("total_duration_sec")
            current_pos_sec = st.get("current_position_sec")
            keyframes_count = st.get("keyframes_kept", 0)
            frames_read = st.get("total_frames_read", 0)
            cam_type = st.get("camera_type", cam_type)

            if st.get("is_running"):
                status = "LIVE (TCP)" if is_connected else "RECONNECTING"
            elif st.get("is_paused"):
                status = "PAUSED"
            else:
                status = "STOPPED"
        elif cam_type == "video_file":
            status = "RECORDED MP4"
            total_duration_sec = 811.0 # 13m 31s
        elif cam_type == "youtube_video":
            status = "RECORDED YT"

        embed_url = ""
        if "youtube.com" in stream_url.lower() or "youtu.be" in stream_url.lower():
            # Extract video ID for clean embed
            import re
            yt_match = re.search(r"(?:v=|\/|embed\/|live\/)([0-9A-Za-z_-]{11})", stream_url)
            vid_id = yt_match.group(1) if yt_match else "1EiC9bvVGnk"
            embed_url = f"https://www.youtube-nocookie.com/embed/{vid_id}?autoplay=1&mute=1&enablejsapi=1"

        feed_item = {
            "camera_id": cam_id,
            "name": cfg.get("name", f"Camera {cam_id}"),
            "type": cam_type,
            "is_live": (cam_type in ["youtube_stream", "rtsp_stream"]),
            "src": stream_url if cam_type != "video_file" else "/video/sample_cctv.mp4",
            "embed_url": embed_url,
            "status": status,
            "fps": fps_val,
            "duration": f"{int(total_duration_sec // 60)}m {int(total_duration_sec % 60)}s" if total_duration_sec else "24/7 LIVE",
            "total_duration_sec": total_duration_sec,
            "current_position_sec": current_pos_sec,
            "progress_pct": progress_pct,
            "keyframes_count": keyframes_count,
            "frames_read": frames_read,
            "preview_image": preview_img,
        }
        feeds.append(feed_item)

    return {"feeds": feeds}


@app.get("/video/sample_cctv.mp4")
def get_sample_video():
    """Stream the sample CCTV video file."""
    video_path = _PROJECT_ROOT / "Video Footage" / "sample_cctv.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(str(video_path), media_type="video/mp4")


@app.post("/api/shutdown")
def shutdown_server():
    """Gracefully shut down the VideoRAG web server process from UI button."""
    logger.info("Shutdown requested from Web UI button.")
    
    def kill_process():
        import time
        import signal
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)

    import threading
    threading.Thread(target=kill_process, daemon=True).start()
    return {"status": "shutting_down", "message": "Server is shutting down..."}


# Mount data static files for frame snapshot images
data_dir = _PROJECT_ROOT / "data"
data_dir.mkdir(exist_ok=True)
app.mount("/data", StaticFiles(directory=str(data_dir)), name="data")

# Mount UI static files
ui_dir = _PROJECT_ROOT / "ui"
ui_dir.mkdir(exist_ok=True)
app.mount("/ui", StaticFiles(directory=str(ui_dir)), name="ui")


@app.get("/")
def read_root():
    """Serve main UI HTML page."""
    index_file = ui_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"message": "VideoRAG UI loading..."})


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Empty favicon response to prevent 404 in browser console."""
    from starlette.responses import Response
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VideoRAG Web API & UI Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    init_pipeline(args.config)
    logger.info("VideoRAG UI running at: http://%s:%d/", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
