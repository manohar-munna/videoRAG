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

# Global pipeline & stream manager instances
PIPELINE: Dict[str, Any] = {}
STREAM_MANAGER = MultiCameraStreamManager(output_dir=str(_PROJECT_ROOT / "data" / "extracted_frames"))


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
    stream_url: str
    sample_interval: Optional[float] = 5.0
    hash_method: Optional[str] = "dhash"
    threshold: Optional[int] = 10


class RemoveStreamRequest(BaseModel):
    camera_id: str


# Latest Hash Filter Audit Trail store
LATEST_HASH_AUDIT: Dict[str, Any] = {
    "stats": {
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
    """Initialize retriever, reranker, vector store, and LLM clients."""
    import yaml
    cfg_file = Path(config_path)
    if not cfg_file.is_absolute():
        cfg_file = _PROJECT_ROOT / cfg_file

    with open(cfg_file, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    cfg_idx = config.get("indexing", {})
    cfg_ret = config.get("retrieval", {})
    cfg_llm = config.get("llm", {})

    model_name = cfg_idx.get("model_name", "all-MiniLM-L6-v2")
    index_path = cfg_idx.get("index_save_path", "index/cctv_index")

    embedder = TextEmbedder(model_name=model_name)
    store = FAISSVectorStore(dim=embedder.dimension)

    idx_path = Path(index_path)
    if not idx_path.is_absolute():
        idx_path = _PROJECT_ROOT / idx_path

    if idx_path.with_suffix(".faiss").exists():
        store.load(str(idx_path))
        logger.info("Loaded FAISS index with %d vectors", store.size)
    else:
        logger.warning("FAISS index not found at %s. Creating empty store.", idx_path)

    retriever = CCTVRetriever(vector_store=store, embedder=embedder)

    try:
        reranker = CrossEncoderReranker()
        logger.info("Cross-encoder reranker loaded.")
    except Exception as exc:
        logger.warning("Cross-encoder failed (%s), using score fallback.", exc)
        reranker = ScoreReranker()

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
    PIPELINE["llm_client"] = llm_client
    PIPELINE["prompter"] = prompter
    PIPELINE["evaluator"] = evaluator
    PIPELINE["config_path"] = str(cfg_file)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def get_health():
    """Return backend health and system info."""
    store = PIPELINE.get("vector_store")
    llm = PIPELINE.get("llm_client")
    return {
        "status": "online",
        "vector_count": store.size if store else 0,
        "llm_backend": llm.backend if llm else "unknown",
        "llm_model": llm.model if llm else "unknown",
        "reranker": PIPELINE.get("reranker").__class__.__name__ if PIPELINE.get("reranker") else "none",
    }


@app.get("/api/events")
def get_events():
    """Return raw CCTV events dataset if available."""
    data_path = _PROJECT_ROOT / "data" / "real_cctv_events.json"
    if not data_path.exists():
        data_path = _PROJECT_ROOT / "data" / "mock_cctv.json"
    
    if data_path.exists():
        with open(data_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return []


@app.post("/api/search")
def search_cctv(req: SearchRequest):
    """Execute semantic query against CCTV video index."""
    if not PIPELINE.get("retriever"):
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    retriever: CCTVRetriever = PIPELINE["retriever"]
    reranker = PIPELINE["reranker"]
    prompter: RAGPrompter = PIPELINE["prompter"]
    llm_client: LLMClient = PIPELINE["llm_client"]
    evaluator: RAGEvaluator = PIPELINE["evaluator"]

    top_k = req.top_k or 10
    rerank_top_k = req.rerank_top_k or 5

    # 1. Retrieve & measure time
    import time
    t0 = time.time()
    query_vec = PIPELINE["embedder"].embed_query(req.query)
    t1 = time.time()
    
    raw_results = retriever.retrieve(req.query, top_k=top_k, camera_filter=req.camera_filter)
    t2 = time.time()

    # 2. Rerank
    for r in raw_results:
        if "text" not in r:
            r["text"] = r.get("metadata", {}).get("text", "")
    reranked = reranker.rerank(req.query, raw_results, top_k=rerank_top_k)
    t3 = time.time()

    # 3. Prompt & LLM
    prompt = prompter.build_prompt(req.query, reranked)
    answer = llm_client.generate(prompt)
    t4 = time.time()

    # 4. Evaluation
    stop_words = {"the", "a", "an", "is", "was", "were", "are", "in", "at", "on", "of", "to", "any", "did", "do", "what", "when", "where", "who", "how", "there"}
    keywords = [w.strip("?.,!").lower() for w in req.query.split() if w.lower() not in stop_words and len(w) > 2]
    eval_result = evaluator.full_evaluation(req.query, reranked, answer, keywords)

    # Format output items
    items = []
    for rank, r in enumerate(reranked, start=1):
        meta = r.get("metadata", {})
        cam = meta.get("camera", "CAM_01")
        ts = meta.get("start_timestamp", meta.get("timestamp", "00:00:00"))
        desc = meta.get("description", r.get("text", ""))

        secs = 0
        try:
            parts = [int(p) for p in ts.split(":")]
            if len(parts) == 3:
                secs = parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2:
                secs = parts[0] * 60 + parts[1]
        except Exception:
            secs = 0

        img_p = meta.get("image_path", "")
        if img_p:
            clean_img = img_p.replace("\\", "/")
            if "data/" in clean_img:
                img_p = "/data/" + clean_img.split("data/", 1)[-1].lstrip("/")

        items.append({
            "rank": rank,
            "camera": cam,
            "timestamp": ts,
            "seconds": secs,
            "description": desc,
            "image_path": img_p,
            "faiss_score": round(float(r.get("score", 0.0)), 4),
            "rerank_score": round(float(r.get("rerank_score", 0.0)), 4),
        })

    # Detailed Vector & Pipeline Debugging Trace
    vec_sample = [round(float(val), 4) for val in query_vec[:12]]
    vec_norm = round(float(sum(v*v for v in query_vec)**0.5), 4)

    debug_trace = {
        "query_vector_dim": len(query_vec),
        "query_vector_norm": vec_norm,
        "query_vector_sample": vec_sample,
        "faiss_indexed_vectors": PIPELINE["vector_store"].size if PIPELINE.get("vector_store") else 0,
        "prompt_constructed": prompt,
        "timings_ms": {
            "query_embedding_ms": round((t1 - t0) * 1000, 2),
            "faiss_retrieval_ms": round((t2 - t1) * 1000, 2),
            "cross_encoder_rerank_ms": round((t3 - t2) * 1000, 2),
            "llm_generation_ms": round((t4 - t3) * 1000, 2),
        }
    }

    return {
        "query": req.query,
        "answer": answer,
        "results": items,
        "evaluation": eval_result,
        "total_retrieved": len(raw_results),
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
    """Return health metrics and stats for all active multi-camera streams."""
    return {"active_streams": STREAM_MANAGER.get_all_statuses()}


@app.post("/api/streams/remove")
def remove_camera_stream(req: RemoveStreamRequest):
    """Stop and remove an active camera stream capture channel."""
    success = STREAM_MANAGER.remove_camera(req.camera_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Camera {req.camera_id} not found")
    return {"status": "stopped", "camera_id": req.camera_id}


@app.post("/api/streams/index_now")
def index_stream_keyframes(req: RemoveStreamRequest):
    """
    Run VLM visual captioning and FAISS vector indexing on all captured keyframes
    for the specified camera stream (e.g. 'CAM_3000').
    """
    cam_id = req.camera_id
    if cam_id not in STREAM_MANAGER.streams:
        raise HTTPException(status_code=404, detail=f"Active stream camera '{cam_id}' not found.")

    stream = STREAM_MANAGER.streams[cam_id]
    keyframes = list(stream.extracted_keyframes)

    if not keyframes:
        raise HTTPException(status_code=400, detail=f"No keyframes extracted yet for camera '{cam_id}'.")

    logger.info("Indexing %d keyframes for camera '%s' using Qwen3-VL VLM...", len(keyframes), cam_id)

    # 1. VLM Captioning
    captioner = VLMCaptioner(backend="local")
    new_records = captioner.caption_batch(keyframes, show_progress=False)

    # 2. Update real_cctv_events.json
    out_file = _PROJECT_ROOT / "data" / "real_cctv_events.json"
    existing_records = []
    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as fh:
                existing_records = json.load(fh)
        except Exception:
            existing_records = []

    # Filter out duplicates by image_path
    existing_paths = {r.get("image_path") for r in existing_records if "image_path" in r}
    unique_new = [r for r in new_records if r.get("image_path") not in existing_paths]
    combined_records = existing_records + unique_new

    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(combined_records, fh, indent=2, ensure_ascii=False)

    # 3. Rebuild FAISS Vector Index
    from scripts.index import run_indexing
    cfg_file = PIPELINE.get("config_path", str(_PROJECT_ROOT / "config" / "config.yaml"))
    run_indexing(config_path=cfg_file, data_path=str(out_file))

    # 4. Reload in-memory FAISS store
    init_pipeline(config_path=cfg_file)
    logger.info("Successfully indexed %d new keyframes for %s. Total vectors: %d", len(unique_new), cam_id, PIPELINE["vector_store"].size)

    return {
        "status": "success",
        "camera_id": cam_id,
        "indexed_count": len(unique_new),
        "total_vectors": PIPELINE["vector_store"].size if PIPELINE.get("vector_store") else 0,
    }


@app.get("/api/cameras")
def get_camera_list():
    """Return list of all unique camera IDs indexed in the system."""
    data_path = _PROJECT_ROOT / "data" / "real_cctv_events.json"
    cams = set()
    if data_path.exists():
        try:
            with open(data_path, "r", encoding="utf-8") as fh:
                records = json.load(fh)
                for r in records:
                    c = r.get("camera")
                    if c:
                        cams.add(c)
        except Exception:
            pass

    for stream_id in STREAM_MANAGER.streams:
        cams.add(stream_id)

    if not cams:
        cams = {"CAM_01", "CAM_02", "CAM_03"}

    return {"cameras": sorted(list(cams))}


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
    print(f"\n🚀 VideoRAG UI running at: http://{args.host}:{args.port}/\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
