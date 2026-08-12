"""
scripts/process_video.py
------------------------
Video processing script for the VideoRAG pipeline.

Steps
-----
1. Extract sampled frame images from a CCTV video file using ``VideoFrameExtractor``.
2. Generate timestamped surveillance descriptions using ``VLMCaptioner``.
3. Save the resulting event dataset to JSON (e.g. ``data/real_cctv_events.json``).
4. Run ``scripts/index.py`` to update/build the FAISS vector index.

Usage
-----
    python scripts/process_video.py --video "Video Footage/sample_cctv.mp4" --camera-id CAM_01 --interval 15
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
) -> None:

    console.print(Panel(
        f"[bold cyan]VideoRAG — Video Processing & Captioning Pipeline[/bold cyan]\n"
        f"[dim]Video : {video_path}\n"
        f"Camera: {camera_id}\n"
        f"Sample: Every {sample_interval}s\n"
        f"Output: {output_json}\n"
        f"Backend: {vlm_backend}[/dim]",
        expand=False,
    ))

    # 1. Extract frames
    console.print("\n[bold cyan]Step 1/3: Extracting video frames…[/bold cyan]")
    extractor = VideoFrameExtractor(output_dir=str(_PROJECT_ROOT / "data" / "extracted_frames"))
    extracted_frames = extractor.extract_frames(
        video_path=video_path,
        camera_id=camera_id,
        sample_interval=sample_interval,
    )
    console.print(f"  [OK] Extracted [green]{len(extracted_frames)}[/green] frames.")

    # 2. VLM Captioning
    console.print("\n[bold cyan]Step 2/3: Captioning frames with VLM…[/bold cyan]")
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
        f"You can now run queries against this footage using scripts/query.py[/dim]",
        expand=False,
        border_style="green",
    ))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process CCTV video footage, caption frames, and index events into FAISS."
    )
    parser.add_argument(
        "--video",
        default=str(_PROJECT_ROOT / "Video Footage" / "sample_cctv.mp4"),
        help="Path to video file (default: Video Footage/sample_cctv.mp4)",
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
        default="local",
        help="VLM backend: local | gemini | heuristic (default: local)",
    )
    parser.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "config" / "config.yaml"),
        help="Path to config YAML",
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
    )
