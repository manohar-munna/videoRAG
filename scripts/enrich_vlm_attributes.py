"""
scripts/enrich_vlm_attributes.py
---------------------------------
Batch extracts rich forensic visual attributes (subjects, equipment, signs,
vehicles, searchable text) using Local Qwen3-VL and persists them to the
CCTV dataset for two-tier dense + semantic hybrid indexing.
"""

import sys
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from videorag.captioning.vlm_captioner import VLMCaptioner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("enrich_vlm")
console = Console()


def run_enrichment(events_path: str = None, cache_path: str = None, max_frames: int = None, max_workers: int = 3):
    console.print(Panel("[bold cyan]VideoRAG — High-Speed Multi-Threaded VLM Forensic Enrichment[/bold cyan]", expand=False))

    events_file = Path(events_path) if events_path else _PROJECT_ROOT / "data" / "real_cctv_events.json"
    cache_file = Path(cache_path) if cache_path else _PROJECT_ROOT / "data" / "vlm_forensic_cache.json"

    if not events_file.exists():
        console.print(f"[bold red]Error: Events file not found: {events_file}[/bold red]")
        return

    with open(events_file, "r", encoding="utf-8") as fh:
        events = json.load(fh)

    cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as fh:
                cache = json.load(fh)
            console.print(f"  [green]Loaded {len(cache)} existing cached VLM attributes.[/green]")
        except Exception as exc:
            logger.warning("Could not load cache: %s", exc)

    console.print(f"  [cyan]Target dataset: {len(events)} keyframes | Parallel Workers: {max_workers}[/cyan]")
    captioner = VLMCaptioner(backend="local")
    cache_lock = Lock()

    target_events = events[:max_frames] if max_frames else events

    def process_single_event(event):
        img_p = event.get("image_path", "")
        local_img = Path(img_p)
        if not local_img.is_absolute():
            cand1 = _PROJECT_ROOT / img_p.lstrip("/")
            cand2 = _PROJECT_ROOT / "data" / img_p.lstrip("/")
            local_img = cand1 if cand1.exists() else cand2

        cache_key = str(local_img.name)
        with cache_lock:
            cached_val = cache.get(cache_key)

        if cached_val:
            attrs = cached_val
        else:
            if local_img.exists():
                attrs = captioner.extract_structured_attributes(str(local_img))
                with cache_lock:
                    cache[cache_key] = attrs
            else:
                attrs = {
                    "summary": event.get("description", "Frame image missing."),
                    "subjects": "",
                    "equipment": "",
                    "vehicles": "",
                    "signs": "",
                    "tags": "",
                    "searchable_text": event.get("description", ""),
                }

        # Apply to event dict
        event["summary"] = attrs.get("summary", "")
        event["subjects"] = attrs.get("subjects", "")
        event["equipment"] = attrs.get("equipment", "")
        event["vehicles"] = attrs.get("vehicles", "")
        event["signs"] = attrs.get("signs", "")
        event["tags"] = attrs.get("tags", "")
        event["searchable_text"] = attrs.get("searchable_text", event.get("description", ""))
        event["description"] = event["searchable_text"]
        return cache_key, attrs

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Extracting VLM attributes (Parallel)…", total=len(target_events))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_event, evt) for evt in target_events]
            completed_count = 0
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    logger.warning("Error in worker thread: %s", exc)

                completed_count += 1
                progress.advance(task, 1)

                if completed_count % 5 == 0:
                    with cache_lock:
                        with open(cache_file, "w", encoding="utf-8") as fh:
                            json.dump(cache, fh, indent=2, ensure_ascii=False)
                        with open(events_file, "w", encoding="utf-8") as fh:
                            json.dump(events, fh, indent=2, ensure_ascii=False)

    # Final persist
    with open(cache_file, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, ensure_ascii=False)
    with open(events_file, "w", encoding="utf-8") as fh:
        json.dump(events, fh, indent=2, ensure_ascii=False)

    console.print(Panel(f"[bold green]Enrichment complete! Saved {len(events)} events with VLM forensic attributes.[/bold green]", expand=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enrich CCTV events with structured VLM attributes")
    parser.add_argument("--events", default=None, help="Path to real_cctv_events.json")
    parser.add_argument("--cache", default=None, help="Path to vlm_forensic_cache.json")
    parser.add_argument("--max", type=int, default=None, help="Max frames to process")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent worker threads")
    args = parser.parse_args()
    run_enrichment(events_path=args.events, cache_path=args.cache, max_frames=args.max, max_workers=args.workers)
