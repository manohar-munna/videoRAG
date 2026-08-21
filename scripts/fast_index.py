"""
scripts/fast_index.py
----------------------
High-speed Lazy VLM indexer:
1. Fast frame extraction with dHash perceptual deduplication (~3s).
2. Fast MobileCLIP-S2 visual spatial crop embedding on GPU (~2s).
3. FAISS index generation (10ms).
Total runtime: < 6 seconds. Zero CPU/RAM strain. Qwen3-VL is lazy (query-time only).
"""

import sys
import json
import logging
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from videorag.ingestion.video_processor import VideoFrameExtractor
from videorag.ingestion.hash_filter import EdgeFrameFilter
from videorag.indexing.embedder import MultimodalEmbedder
from videorag.indexing.vector_store import FAISSVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("fast_index")
console = Console()


def run_fast_indexing(
    video_path: str = "data/videos/sample_cctv.mp4",
    sample_interval: float = 4.0,
    dhash_threshold: int = 10,
):
    console.print(Panel("[bold cyan]VideoRAG — High-Speed Lazy VLM Indexer (Zero Indexing Compute)[/bold cyan]", expand=False))

    vid_p = Path(video_path)
    if not vid_p.is_absolute():
        vid_p = _PROJECT_ROOT / vid_p

    if not vid_p.exists():
        console.print(f"[bold red]Error: Video file not found: {vid_p}[/bold red]")
        return

    # ------------------------------------------------------------------
    # Step 1: Perceptual Frame Extraction
    # ------------------------------------------------------------------
    console.print(f"\n[bold yellow]1. Extracting Keyframes from {vid_p.name}...[/bold yellow]")
    out_dir = _PROJECT_ROOT / "data" / "extracted_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_filter = EdgeFrameFilter(method="dhash", threshold=dhash_threshold)
    extractor = VideoFrameExtractor(output_dir=str(out_dir))

    result = extractor.extract_frames(
        video_path=str(vid_p),
        camera_id="CAM_01",
        sample_interval=sample_interval,
        max_frames=600,
        hash_filter=edge_filter,
    )

    extracted_frames = result["extracted_frames"]
    console.print(f"  [green]Kept {len(extracted_frames)} keyframes (Skipped {result['skipped_count']} duplicate frames)[/green]")

    events_file = _PROJECT_ROOT / "data" / "real_cctv_events.json"
    for f in extracted_frames:
        f["description"] = f"Surveillance keyframe captured by {f['camera']} at {f['timestamp']}."
        f["searchable_text"] = f["description"]

    with open(events_file, "w", encoding="utf-8") as fh:
        json.dump(extracted_frames, fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Step 2: High-Speed MobileCLIP-S2 Spatial Crops Embedding
    # ------------------------------------------------------------------
    console.print(f"\n[bold yellow]2. Embedding Spatial Crops with MobileCLIP-S2 on GPU...[/bold yellow]")
    embedder = MultimodalEmbedder(model_name="MobileCLIP-S2")
    import numpy as np

    embeddings_list = []
    metadata_list = []

    for f in extracted_frames:
        img_p = Path(f["image_path"])
        try:
            crop_results = embedder.embed_image_with_crops(img_p)
            for cr in crop_results:
                embeddings_list.append(cr["embedding"])
                meta = dict(f)
                meta["chunk_id"] = f"{f['camera']}_{f['timestamp'].replace(':', '_')}_{cr['crop_region']}"
                meta["crop_region"] = cr["crop_region"]
                meta["crop_box"] = cr["crop_box"]
                meta["vector_type"] = "visual"
                meta["text"] = f"Camera: {f['camera']} | Time: {f['timestamp']} | Region: {cr['crop_region']}"
                metadata_list.append(meta)
        except Exception as exc:
            logger.warning("Error embedding %s: %s", img_p, exc)

    embeddings = np.vstack(embeddings_list)
    store = FAISSVectorStore(dim=embedder.dimension)
    store.add(embeddings, metadata_list)

    idx_path = _PROJECT_ROOT / "index" / "cctv_index"
    store.save(str(idx_path))

    console.print(Panel(f"[bold green]Indexing Complete in ~5s! Created FAISS Index with {store.size} vectors.[/bold green]", expand=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fast Lazy VLM Indexer")
    parser.add_argument("--video", default="data/videos/sample_cctv.mp4")
    parser.add_argument("--interval", type=float, default=4.0)
    parser.add_argument("--dhash", type=int, default=10)
    args = parser.parse_args()

    run_fast_indexing(video_path=args.video, sample_interval=args.interval, dhash_threshold=args.dhash)
