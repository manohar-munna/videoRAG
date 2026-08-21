"""
scripts/process_video.py
------------------------
Video processing script for the VideoRAG pipeline with optional dHash/pHash Edge Filtering.

Steps
-----
1. Extract sampled frame images using ``VideoFrameExtractor`` and optional ``EdgeFrameFilter``.
2. Generate timestamped surveillance descriptions using ``VLMCaptioner``.
3. Save the resulting event dataset to JSON (e.g. ``data/real_cctv_events.json``).
4. Run ``scripts/index.py`` to update/build the FAISS vector index.

Usage
-----
    python scripts/process_video.py --video "data/videos/sample_cctv.mp4" --camera-id CAM_01 --interval 15 --enable-hash-filter --hash-method dhash --threshold 10
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import io

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console(force_terminal=True, highlight=True)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from videorag.ingestion.video_processor import VideoFrameExtractor
from videorag.ingestion.hash_filter import EdgeFrameFilter
from videorag.captioning.vlm_captioner import VLMCaptioner
from scripts.index import run_indexing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def process_video_pipeline(
    video_path: str,
    camera_id: str = "CAM_01",
    sample_interval: float = 15.0,
    output_json: str = "data/real_cctv_events.json",
    vlm_backend: str = "local",
    config_path: str = "config/config.yaml",
    enable_hash_filter: bool = False,
    hash_method: str = "dhash",
    hash_threshold: int = 10,
) -> None:

    console.print(Panel(
        f"[bold cyan]VideoRAG — Video Processing & Captioning Pipeline[/bold cyan]\n"
        f"[dim]Video       : {video_path}\n"
        f"Camera      : {camera_id}\n"
        f"Sample      : Every {sample_interval}s\n"
        f"Smart Filter: {'ENABLED (' + hash_method.upper() + ', thresh=' + str(hash_threshold) + ')' if enable_hash_filter else 'DISABLED'}\n"
        f"Output      : {output_json}\n"
        f"Backend     : {vlm_backend}[/dim]",
        expand=False,
    ))

    # 1. Extract frames with optional EdgeFrameFilter
    console.print("\n[bold cyan]Step 1/3: Extracting video frames…[/bold cyan]")
    extractor = VideoFrameExtractor(output_dir=str(_PROJECT_ROOT / "data" / "extracted_frames"))
    
    hash_filter = EdgeFrameFilter(method=hash_method, threshold=hash_threshold) if enable_hash_filter else None
    result = extractor.extract_frames(
        video_path=video_path,
        camera_id=camera_id,
        sample_interval=sample_interval,
        hash_filter=hash_filter,
    )

    extracted_frames = result["extracted_frames"]
    filter_stats = result["filter_stats"]

    console.print(f"  [OK] Sampled [yellow]{result['total_sampled']}[/yellow] frames -> Kept [green]{len(extracted_frames)}[/green] keyframes, Skipped [red]{result['skipped_count']}[/red] static frames.")
    if enable_hash_filter:
        console.print(f"  [bold green]⚡ Saved {filter_stats.get('llm_compute_saved_pct', 0.0)}% of LLM compute using {hash_method.upper()} Hamming Distance thresholding![/bold green]")

    if not extracted_frames:
        console.print("[yellow]No keyframes passed the dHash filter threshold. Nothing to caption.[/yellow]")
        return

    # 2. Keyframe Records Preparation (Lazy VLM or Full Captioning)
    if vlm_backend in ("lazy", "skip", "none"):
        console.print("\n[bold cyan]Step 2/3: Fast Lazy VLM Mode (Instant direct visual embedding)…[/bold cyan]")
        records = []
        for item in extracted_frames:
            rec = dict(item)
            rec["description"] = f"Surveillance keyframe captured by {rec.get('camera', camera_id)} at {rec.get('timestamp', '00:00:00')}."
            rec["image_path"] = str(rec.get("image_path", "")).replace("\\", "/")
            records.append(rec)
    else:
        console.print("\n[bold cyan]Step 2/3: Captioning keyframes with local VLM…[/bold cyan]")
        captioner = VLMCaptioner(backend=vlm_backend)
        records = captioner.caption_batch(extracted_frames, show_progress=True)

    out_file = Path(output_json)
    if not out_file.is_absolute():
        out_file = _PROJECT_ROOT / out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    console.print(f"  [OK] Saved [green]{len(records)}[/green] CCTV event records -> [yellow]{out_file}[/yellow]")

    # Display sample records
    tbl = Table(title="Sample Processed CCTV Events", box=box.ROUNDED, show_header=True)
    tbl.add_column("#", style="dim", width=3)
    tbl.add_column("Camera", style="cyan", no_wrap=True)
    tbl.add_column("Timestamp", style="yellow", no_wrap=True)
    tbl.add_column("Description", style="white")

    for idx, r in enumerate(records[:5], 1):
        desc = r["description"][:90] + ("..." if len(r["description"]) > 90 else "")
        tbl.add_row(str(idx), r["camera"], r["timestamp"], desc)

    console.print(tbl)

    # 3. Indexing
    console.print("\n[bold cyan]Step 3/3: Building FAISS Index…[/bold cyan]")
    cfg_file = Path(config_path)
    if not cfg_file.is_absolute():
        cfg_file = _PROJECT_ROOT / cfg_file

    run_indexing(config_path=str(cfg_file), data_path=str(out_file))

    console.print(Panel(
        f"[bold green]Video processing and indexing complete![/bold green]\n"
        f"[dim]Total events indexed: {len(records)}\n"
        f"LLM Compute Saved: {filter_stats.get('llm_compute_saved_pct', 0.0)}%\n"
        f"You can now run queries against this footage using scripts/query.py or Web UI[/dim]",
        expand=False,
        border_style="green",
    ))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process CCTV video footage, caption frames, and index events into FAISS with optional dHash filtering."
    )
    parser.add_argument(
        "--video",
        default=str(_PROJECT_ROOT / "data" / "videos" / "sample_cctv.mp4"),
        help="Path to video file (default: data/videos/sample_cctv.mp4)",
    )
    parser.add_argument(
        "--camera-id",
        default="CAM_01",
        help="Camera identifier (default: CAM_01)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=15.0,
        help="Frame sampling interval in seconds (default: 15.0)",
    )
    parser.add_argument(
        "--output",
        default="data/real_cctv_events.json",
        help="Output JSON path (default: data/real_cctv_events.json)",
    )
    parser.add_argument(
        "--backend",
        default="lazy",
        help="VLM backend: lazy | local | gemini | heuristic (default: lazy)",
    )
    parser.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "config" / "config.yaml"),
        help="Path to config YAML",
    )
    parser.add_argument(
        "--enable-hash-filter",
        action="store_true",
        default=True,
        help="Enable dHash/pHash frame filtering to skip duplicate/static frames (default: True)",
    )
    parser.add_argument(
        "--hash-method",
        default="dhash",
        choices=["dhash", "phash", "ahash"],
        help="Hash algorithm method (default: dhash)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=8,
        help="Hamming distance threshold (default: 8)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    process_video_pipeline(
        video_path=args.video,
        camera_id=args.camera_id,
        sample_interval=args.interval,
        output_json=args.output,
        vlm_backend=args.backend,
        config_path=args.config,
        enable_hash_filter=args.enable_hash_filter,
        hash_method=args.hash_method,
        hash_threshold=args.threshold,
    )
