"""
src/videorag/server.py
----------------------
FastAPI Web Application backend for VideoRAG.
Serves REST API endpoints for semantic search, video processing, system health,
and hosts the Glassmorphic web UI.
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

# Global pipeline instances
PIPELINE: Dict[str, Any] = {}


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    rerank_top_k: Optional[int] = 5
    camera_filter: Optional[str] = None


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

    # 1. Retrieve
    raw_results = retriever.retrieve(req.query, top_k=top_k, camera_filter=req.camera_filter)

    # 2. Rerank
    for r in raw_results:
        if "text" not in r:
            r["text"] = r.get("metadata", {}).get("text", "")
    reranked = reranker.rerank(req.query, raw_results, top_k=rerank_top_k)

    # 3. Prompt & LLM
    prompt = prompter.build_prompt(req.query, reranked)
    answer = llm_client.generate(prompt)

    # 4. Evaluation
    stop_words = {"the", "a", "an", "is", "was", "were", "are", "in", "at", "on", "of", "to", "any", "did", "do", "what", "when", "where", "who", "how", "there"}
    keywords = [w.strip("?.,!").lower() for w in req.query.split() if w.lower() not in stop_words and len(w) > 2]
    eval_result = evaluator.full_evaluation(req.query, reranked, answer, keywords)

    # Format output items with frame image links if present
    items = []
    for rank, r in enumerate(reranked, start=1):
        meta = r.get("metadata", {})
        cam = meta.get("camera", "CAM_01")
        ts = meta.get("start_timestamp", meta.get("timestamp", "00:00:00"))
        desc = meta.get("description", r.get("text", ""))

        # Convert HH:MM:SS to total seconds for video seek
        secs = 0
        try:
            parts = [int(p) for p in ts.split(":")]
            if len(parts) == 3:
                secs = parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2:
                secs = parts[0] * 60 + parts[1]
        except Exception:
            secs = 0

        items.append({
            "rank": rank,
            "camera": cam,
            "timestamp": ts,
            "seconds": secs,
            "description": desc,
            "faiss_score": round(float(r.get("score", 0.0)), 4),
            "rerank_score": round(float(r.get("rerank_score", 0.0)), 4),
        })

    return {
        "query": req.query,
        "answer": answer,
        "results": items,
        "evaluation": eval_result,
        "total_retrieved": len(raw_results),
    }


@app.get("/video/sample_cctv.mp4")
def get_sample_video():
    """Stream the sample CCTV video file."""
    video_path = _PROJECT_ROOT / "Video Footage" / "sample_cctv.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(str(video_path), media_type="video/mp4")


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
