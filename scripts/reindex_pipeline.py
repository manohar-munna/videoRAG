"""
scripts/reindex_pipeline.py
----------------------------
Clean, automated end-to-end ingestion and indexing pipeline:
1. Frame extraction with dHash perceptual deduplication from video(s).
2. Parallel forensic attribute extraction via Local Qwen3-VL.
3. Two-tier multimodal FAISS vector indexing (MobileCLIP-S2 visual crops + VLM semantic vectors).
"""

import os
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
from videorag.captioning.vlm_captioner import VLMCaptioner
from videorag.indexing.embedder import MultimodalEmbedder
from videorag.indexing.vector_store import FAISSVectorStore
from videorag.indexing.chunker import DocumentChunker
from videorag.ingestion.loader import CCTVDataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("reindex_pipeline")
console = Console()


def run_full_pipeline(
    video_path: str = "Video Footage/sample_cctv.mp4",
    sample_interval: float = 4.0,
    dhash_threshold: int = 10,
    workers: int = 3,
    skip_vlm: bool = False,
):
    console.print(Panel("[bold cyan]VideoRAG — Full Clean Re-Indexing Pipeline[/bold cyan]", expand=False))

    vid_p = Path(video_path)
    if not vid_p.is_absolute():
        vid_p = _PROJECT_ROOT / vid_p

    if not vid_p.exists():
        console.print(f"[bold red]Error: Video file not found: {vid_p}[/bold red]")
        return

    # ------------------------------------------------------------------
    # Step 1: Video Frame Extraction with dHash Deduplication
    # ------------------------------------------------------------------
    console.print(f"\n[bold yellow]Stage 1: Extracting Frames from {vid_p.name} (Interval: {sample_interval}s)...[/bold yellow]")
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
    cache_file = _PROJECT_ROOT / "data" / "vlm_forensic_cache.json"

    # ------------------------------------------------------------------
    # Step 2: Parallel VLM Forensic Attribute Enrichment
    # ------------------------------------------------------------------
    if not skip_vlm:
        console.print(f"\n[bold yellow]Stage 2: Parallel Qwen3-VL Forensic Attribute Extraction ({workers} workers)...[/bold yellow]")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from threading import Lock
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

        captioner = VLMCaptioner(backend="local")
        cache = {}
        cache_lock = Lock()

        def process_frame(evt):
            img_p = evt["image_path"]
            img_name = Path(img_p).name
            attrs = captioner.extract_structured_attributes(str(img_p))
            with cache_lock:
                cache[img_name] = attrs

            evt["summary"] = attrs.get("summary", "")
            evt["subjects"] = attrs.get("subjects", "")
            evt["equipment"] = attrs.get("equipment", "")
            evt["vehicles"] = attrs.get("vehicles", "")
            evt["signs"] = attrs.get("signs", "")
            evt["tags"] = attrs.get("tags", "")
            evt["searchable_text"] = attrs.get("searchable_text", "")
            evt["description"] = evt["searchable_text"]
            return img_name

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Extracting VLM attributes…", total=len(extracted_frames))

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(process_frame, f) for f in extracted_frames]
                done_cnt = 0
                for fut in as_completed(futures):
                    try:
                        fut.result()
                    except Exception as exc:
                        logger.warning("Error extracting frame attributes: %s", exc)
                    done_cnt += 1
                    progress.advance(task, 1)

                    if done_cnt % 5 == 0:
                        with cache_lock:
                            with open(cache_file, "w", encoding="utf-8") as fh:
                                json.dump(cache, fh, indent=2, ensure_ascii=False)
                            with open(events_file, "w", encoding="utf-8") as fh:
                                json.dump(extracted_frames, fh, indent=2, ensure_ascii=False)

        with open(cache_file, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2, ensure_ascii=False)
        with open(events_file, "w", encoding="utf-8") as fh:
            json.dump(extracted_frames, fh, indent=2, ensure_ascii=False)
        console.print(f"  [green]Saved enriched dataset with {len(extracted_frames)} frames.[/green]")
    else:
        # Create base event descriptions
        for f in extracted_frames:
            f["description"] = f"Surveillance keyframe at {f['timestamp']}."
            f["searchable_text"] = f["description"]
        with open(events_file, "w", encoding="utf-8") as fh:
            json.dump(extracted_frames, fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Step 3: Two-Tier Multimodal FAISS Indexing
    # ------------------------------------------------------------------
    console.print(f"\n[bold yellow]Stage 3: Building Two-Tier Multimodal FAISS Vector Index...[/bold yellow]")
    loader = CCTVDataLoader()
    documents = loader.load_as_documents(str(events_file))
    chunker = DocumentChunker()
    chunks = chunker.chunk_individual(documents)

    embedder = MultimodalEmbedder(model_name="MobileCLIP-S2")
    import numpy as np

    embeddings_list = []
    metadata_list = []

    for c in chunks:
        img_p = c["metadata"].get("image_path", "")
        local_img = Path(img_p)
        if not local_img.is_absolute():
            cand1 = _PROJECT_ROOT / img_p.lstrip("/")
            cand2 = _PROJECT_ROOT / "data" / img_p.lstrip("/")
            local_img = cand1 if cand1.exists() else cand2

        if local_img.exists():
            try:
                crop_results = embedder.embed_image_with_crops(local_img)
                for cr in crop_results:
                    embeddings_list.append(cr["embedding"])
                    meta = dict(c["metadata"])
                    meta["text"] = c["text"]
                    meta["chunk_id"] = f"{c['chunk_id']}_{cr['crop_region']}"
                    meta["crop_region"] = cr["crop_region"]
                    meta["crop_box"] = cr["crop_box"]
                    meta["vector_type"] = "visual"
                    meta["description"] = c["metadata"].get("description", "")
                    meta["image_path"] = img_p
                    metadata_list.append(meta)
            except Exception as exc:
                logger.warning("Error embedding visual crops: %s", exc)

        # Tier-2: Dense VLM Semantic Vector
        s_text = c["metadata"].get("searchable_text") or c.get("text") or c["metadata"].get("description", "")
        if s_text and len(s_text.strip()) > 5:
            try:
                v_text = embedder.embed_query(s_text)
                embeddings_list.append(v_text)
                meta_text = dict(c["metadata"])
                meta_text["text"] = s_text
                meta_text["chunk_id"] = f"{c['chunk_id']}_vlm_semantic"
                meta_text["crop_region"] = "vlm_semantic"
                meta_text["crop_box"] = (0.0, 0.0, 1.0, 1.0)
                meta_text["vector_type"] = "vlm_semantic"
                meta_text["image_path"] = img_p
                metadata_list.append(meta_text)
            except Exception as exc:
                logger.warning("Error embedding VLM text: %s", exc)

    embeddings = np.vstack(embeddings_list)
    store = FAISSVectorStore(dim=embedder.dimension)
    store.add(embeddings, metadata_list)

    idx_path = _PROJECT_ROOT / "index" / "cctv_index"
    store.save(str(idx_path))

    console.print(Panel(f"[bold green]Re-Indexing Complete! Created FAISS Index with {store.size} vectors.[/bold green]", expand=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Full clean re-indexing pipeline")
    parser.add_argument("--video", default="Video Footage/sample_cctv.mp4", help="Path to input video")
    parser.add_argument("--interval", type=float, default=4.0, help="Sampling interval in seconds")
    parser.add_argument("--dhash", type=int, default=10, help="dHash Hamming distance threshold")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent VLM extraction workers")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM attribute extraction")
    args = parser.parse_args()

    run_full_pipeline(
        video_path=args.video,
        sample_interval=args.interval,
        dhash_threshold=args.dhash,
        workers=args.workers,
        skip_vlm=args.skip_vlm,
    )
