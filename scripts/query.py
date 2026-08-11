"""
scripts/query.py
----------------
Interactive query CLI for the CCTV VideoRAG pipeline.

Usage
-----
    # Interactive loop
    python scripts/query.py

    # Single query mode
    python scripts/query.py --query "Was there any suspicious activity near Gate 3?"

    # Custom config / index
    python scripts/query.py --config config/config.yaml

The script loads the saved FAISS index, accepts natural-language queries,
retrieves & reranks results, generates an LLM answer, and evaluates the
response — all displayed with rich formatting.
"""

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src/ is importable when run directly
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.rule import Rule
from rich.text import Text

from videorag.indexing.embedder import TextEmbedder
from videorag.indexing.vector_store import FAISSVectorStore
from videorag.retrieval.retriever import CCTVRetriever
from videorag.retrieval.reranker import CrossEncoderReranker, ScoreReranker
from videorag.llm.prompter import RAGPrompter, LLMClient
from videorag.evaluation.evaluator import RAGEvaluator

logging.basicConfig(
    level=logging.WARNING,          # keep console clean during interactive use
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)
console = Console(force_terminal=True, highlight=True)
import io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _build_pipeline(config: dict):
    """Instantiate all pipeline components and return them as a tuple."""
    cfg_idx = config.get("indexing", {})
    cfg_ret = config.get("retrieval", {})
    cfg_llm = config.get("llm", {})

    model_name: str = cfg_idx.get("model_name", "all-MiniLM-L6-v2")
    index_save_path: str = cfg_idx.get("index_save_path", "index/cctv_index")

    top_k: int = cfg_ret.get("top_k", 10)
    use_reranker: bool = cfg_ret.get("use_reranker", True)
    rerank_top_k: int = cfg_ret.get("rerank_top_k", 5)

    llm_backend: str = cfg_llm.get("backend", "mock")
    llm_model: str = cfg_llm.get("model", "gpt-4o")
    llm_base_url = cfg_llm.get("base_url") or None

    # Resolve index path
    idx_path = Path(index_save_path)
    if not idx_path.is_absolute():
        idx_path = _PROJECT_ROOT / idx_path

    console.print(f"[dim]Loading embedding model '[cyan]{model_name}[/cyan]'…[/dim]")
    embedder = TextEmbedder(model_name=model_name)

    console.print(f"[dim]Loading FAISS index from '[cyan]{idx_path}[/cyan]'…[/dim]")
    store = FAISSVectorStore(dim=embedder.dimension)
    store.load(str(idx_path))
    console.print(
        f"[dim]  Index loaded — [green]{store.size}[/green] vectors[/dim]\n"
    )

    retriever = CCTVRetriever(vector_store=store, embedder=embedder)

    if use_reranker:
        console.print("[dim]Loading cross-encoder reranker…[/dim]")
        try:
            reranker = CrossEncoderReranker()
        except Exception as exc:
            console.print(f"[yellow]Warning: cross-encoder failed ({exc}). Using score reranker.[/yellow]")
            reranker = ScoreReranker()
    else:
        reranker = ScoreReranker()

    prompter = RAGPrompter()
    llm_client = LLMClient(backend=llm_backend, model=llm_model, base_url=llm_base_url)
    evaluator = RAGEvaluator()

    return (
        retriever, reranker, prompter, llm_client, evaluator,
        top_k, rerank_top_k,
        config.get("evaluation", {}).get("enabled", True),
    )


def _answer_query(
    query: str,
    retriever: CCTVRetriever,
    reranker,
    prompter: RAGPrompter,
    llm_client: LLMClient,
    evaluator: RAGEvaluator,
    top_k: int,
    rerank_top_k: int,
    eval_enabled: bool,
) -> None:
    """Execute a single query and print all results to the console."""
    console.print(Rule(f"[bold cyan]Query[/bold cyan]"))
    console.print(Panel(f"[bold]{query}[/bold]", expand=False))

    # ------------------------------------------------------------------
    # 1. Retrieve
    # ------------------------------------------------------------------
    console.print("[dim]Retrieving…[/dim]")
    results = retriever.retrieve(query, top_k=top_k)

    # ------------------------------------------------------------------
    # 2. Rerank
    # ------------------------------------------------------------------
    console.print("[dim]Reranking…[/dim]")
    # Add 'text' key for cross-encoder (may live in metadata)
    for r in results:
        if "text" not in r:
            r["text"] = r.get("metadata", {}).get("text", "")
    reranked = reranker.rerank(query, results, top_k=rerank_top_k)

    # ------------------------------------------------------------------
    # 3. Display results table
    # ------------------------------------------------------------------
    tbl = Table(
        title=f"Top {len(reranked)} Retrieved Evidence Chunks",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    tbl.add_column("#", style="dim", width=3)
    tbl.add_column("Camera", style="cyan", no_wrap=True)
    tbl.add_column("Timestamp", style="yellow", no_wrap=True)
    tbl.add_column("Description", style="white")
    tbl.add_column("Score", style="green", width=8)

    for rank, r in enumerate(reranked, start=1):
        meta = r.get("metadata", {})
        camera = meta.get("camera", "—")
        timestamp = meta.get("start_timestamp", meta.get("timestamp", "—"))
        description = meta.get("description", r.get("text", "—"))
        if len(description) > 100:
            description = description[:97] + "..."
        score = r.get("rerank_score", r.get("score", 0.0))
        tbl.add_row(str(rank), camera, timestamp, description, f"{score:.4f}")

    console.print(tbl)

    # ------------------------------------------------------------------
    # 4. Build prompt & generate answer
    # ------------------------------------------------------------------
    console.print("[dim]Generating answer…[/dim]")
    prompt = prompter.build_prompt(query, reranked)
    answer = llm_client.generate(prompt)

    console.print(Panel(
        answer,
        title="[bold green]Assistant Answer[/bold green]",
        border_style="green",
        expand=False,
    ))

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    if eval_enabled:
        # Extract keywords from query for heuristic evaluation
        stop_words = {"the", "a", "an", "is", "was", "were", "are", "in",
                      "at", "on", "of", "to", "any", "did", "do", "what",
                      "when", "where", "who", "how", "there", "near", "with"}
        keywords = [
            w.strip("?.,!").lower()
            for w in query.split()
            if w.lower() not in stop_words and len(w) > 2
        ]
        eval_result = evaluator.full_evaluation(query, reranked, answer, keywords)
        evaluator.print_report(eval_result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive query CLI for the CCTV VideoRAG pipeline."
    )
    parser.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "config" / "config.yaml"),
        help="Path to config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Run a single query and exit (skips interactive loop)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = _load_config(args.config)

    console.print(Panel(
        "[bold cyan]VideoRAG — CCTV Query Interface[/bold cyan]",
        expand=False,
    ))

    # Build pipeline components
    (
        retriever, reranker, prompter, llm_client, evaluator,
        top_k, rerank_top_k, eval_enabled,
    ) = _build_pipeline(config)

    # ------------------------------------------------------------------
    # Single-query mode
    # ------------------------------------------------------------------
    if args.query:
        _answer_query(
            query=args.query,
            retriever=retriever,
            reranker=reranker,
            prompter=prompter,
            llm_client=llm_client,
            evaluator=evaluator,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            eval_enabled=eval_enabled,
        )
        sys.exit(0)

    # ------------------------------------------------------------------
    # Interactive loop
    # ------------------------------------------------------------------
    console.print(
        "[bold]Enter a query[/bold] to search the CCTV index. "
        "Type [bold red]exit[/bold red] or [bold red]quit[/bold red] to stop.\n"
    )

    while True:
        try:
            query = console.input("[bold cyan]Query > [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Exiting.[/dim]")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye.[/dim]")
            break

        _answer_query(
            query=query,
            retriever=retriever,
            reranker=reranker,
            prompter=prompter,
            llm_client=llm_client,
            evaluator=evaluator,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            eval_enabled=eval_enabled,
        )
        console.print()
