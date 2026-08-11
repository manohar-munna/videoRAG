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

    effective_data_path = data_path or cfg_data.get("mock_path", "data/mock_cctv.json")
    model_name: str = cfg_idx.get("model_name", "all-MiniLM-L6-v2")
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
        # 4. Embed chunks
        # ------------------------------------------------------------------
        task_embed = progress.add_task("[cyan]Loading embedding model…", total=1)
        embedder = TextEmbedder(model_name=model_name)
        progress.update(task_embed, completed=1)

        task_enc = progress.add_task("[cyan]Encoding chunks…", total=1)
        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed(texts)
        progress.update(task_enc, completed=1)
        console.print(f"  [OK] Produced embeddings of shape [green]{embeddings.shape}[/green]")

        # ------------------------------------------------------------------
        # 5. Build FAISS index
        # ------------------------------------------------------------------
        task_index = progress.add_task("[cyan]Building FAISS index…", total=1)
        dim = embeddings.shape[1]
        store = FAISSVectorStore(dim=dim)
        metadata = [c["metadata"] for c in chunks]
        # Augment metadata with the full chunk text and chunk_id for retrieval
        for chunk, meta in zip(chunks, metadata):
            meta["text"] = chunk["text"]
            meta["chunk_id"] = chunk["chunk_id"]
            meta["description"] = chunk["metadata"].get("description", "")
        store.add(embeddings, metadata)
        progress.update(task_index, completed=1)
        console.print(f"  [OK] Index built with [green]{store.size}[/green] vectors")

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
