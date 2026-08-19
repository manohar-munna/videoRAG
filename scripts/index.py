"""
scripts/index.py
----------------
Runnable indexing script for the CCTV VideoRAG pipeline.

Usage
-----
    python scripts/index.py
    python scripts/index.py --config config/config.yaml --data data/mock_cctv.json

Steps
-----
1. Load configuration from a YAML file.
2. Load CCTV documents from a JSON file using ``CCTVDataLoader``.
3. Chunk documents (individual mode) via ``DocumentChunker``.
4. Embed chunks using ``TextEmbedder``.
5. Build and save a ``FAISSVectorStore`` to disk.
"""

import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # loads .env from project root automatically

# ---------------------------------------------------------------------------
# Make sure the src/ directory is on the import path when run directly.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from videorag.ingestion.loader import CCTVDataLoader
from videorag.indexing.chunker import DocumentChunker
from videorag.indexing.embedder import TextEmbedder
from videorag.indexing.vector_store import FAISSVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)
console = Console(force_terminal=True, highlight=True)
import io, sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> dict:
    """Read YAML config from *config_path*."""
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_indexing(config_path: str, data_path: str) -> None:
    """Execute the full indexing pipeline.

    Args:
        config_path: Path to the YAML configuration file.
        data_path: Path to the CCTV JSON data file (overrides config if
            provided).
    """
    console.print(Panel("[bold cyan]VideoRAG — CCTV Indexing Pipeline[/bold cyan]", expand=False))

    # ------------------------------------------------------------------
    # 1. Configuration
    # ------------------------------------------------------------------
    config = _load_config(config_path)
    cfg_data = config.get("data", {})
    cfg_idx = config.get("indexing", {})

    effective_data_path = data_path or cfg_data.get("data_path", cfg_data.get("mock_path", "data/real_cctv_events.json"))
    model_name: str = cfg_idx.get("model_name", "MobileCLIP-S2")
    model_path = cfg_idx.get("model_path")
    index_save_path: str = cfg_idx.get("index_save_path", "index/cctv_index")

    console.print(f"[bold]Config:[/bold]       {config_path}")
    console.print(f"[bold]Data file:[/bold]    {effective_data_path}")
    console.print(f"[bold]Embed model:[/bold]  {model_name}")
    console.print(f"[bold]Index path:[/bold]   {index_save_path}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        # ------------------------------------------------------------------
        # 2. Load data
        # ------------------------------------------------------------------
        task_load = progress.add_task("[cyan]Loading CCTV data…", total=1)
        loader = CCTVDataLoader()
        documents = loader.load_as_documents(effective_data_path)
        progress.update(task_load, completed=1)
        console.print(f"  [OK] Loaded [green]{len(documents)}[/green] documents")

        # ------------------------------------------------------------------
        # 3. Chunk documents
        # ------------------------------------------------------------------
        task_chunk = progress.add_task("[cyan]Chunking documents…", total=1)
        chunker = DocumentChunker()
        chunks = chunker.chunk_individual(documents)
        progress.update(task_chunk, completed=1)
        console.print(f"  [OK] Created [green]{len(chunks)}[/green] chunks (individual mode)")

        # ------------------------------------------------------------------
        # 4. Embed chunks (Multimodal CLIP)
        # ------------------------------------------------------------------
        task_embed = progress.add_task("[cyan]Loading MobileCLIP embedder…", total=1)
        from videorag.indexing.embedder import MultimodalEmbedder
        embedder = MultimodalEmbedder(model_name=model_name, model_path=model_path)
        progress.update(task_embed, completed=1)

        task_enc = progress.add_task("[cyan]Encoding chunks (images & spatial crops)…", total=len(chunks))
        import numpy as np
        embeddings_list = []
        metadata_list = []

        for c in chunks:
            img_p = c["metadata"].get("image_path", "")
            img_embedded = False
            if img_p:
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
                            meta["description"] = c["metadata"].get("description", "")
                            meta["image_path"] = img_p
                            metadata_list.append(meta)
                        img_embedded = True
                    except Exception as exc:
                        logger.warning("Error embedding image crops for %s: %s", local_img, exc)

            if not img_embedded:
                # Text embedding fallback
                v = embedder.embed_query(c["text"])
                embeddings_list.append(v)
                meta = dict(c["metadata"])
                meta["text"] = c["text"]
                meta["chunk_id"] = c["chunk_id"]
                meta["crop_region"] = "text_only"
                meta["crop_box"] = (0.0, 0.0, 1.0, 1.0)
                meta["description"] = c["metadata"].get("description", "")
                meta["image_path"] = c["metadata"].get("image_path", "")
                metadata_list.append(meta)

            progress.advance(task_enc, 1)

        embeddings = np.vstack(embeddings_list)
        console.print(f"  [OK] Produced multimodal embeddings of shape [green]{embeddings.shape}[/green]")

        # ------------------------------------------------------------------
        # 5. Build FAISS index
        # ------------------------------------------------------------------
        task_index = progress.add_task("[cyan]Building FAISS index…", total=1)
        dim = embeddings.shape[1]
        store = FAISSVectorStore(dim=dim)
        store.add(embeddings, metadata_list)
        progress.update(task_index, completed=1)
        console.print(f"  [OK] Index built with [green]{store.size}[/green] vectors (global + regional spatial crops)")

        # ------------------------------------------------------------------
        # 6. Save index
        # ------------------------------------------------------------------
        task_save = progress.add_task("[cyan]Saving index to disk…", total=1)
        # Resolve relative to project root
        save_path = Path(index_save_path)
        if not save_path.is_absolute():
            save_path = _PROJECT_ROOT / save_path
        store.save(str(save_path))
        progress.update(task_save, completed=1)
        console.print(f"  [OK] Index saved -> [green]{save_path}[/green]")

    console.print(Panel("[bold green]Indexing complete![/bold green]", expand=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and save a FAISS index from CCTV JSON data."
    )
    parser.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "config" / "config.yaml"),
        help="Path to config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to CCTV JSON data file (overrides config if provided)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_indexing(config_path=args.config, data_path=args.data)
