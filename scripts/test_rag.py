"""
scripts/test_rag.py
--------------------
Deep inspection / debug test script for the CCTV VideoRAG pipeline.

Runs a set of test queries and for EACH query prints:
  1.  The raw FAISS retrieval results with exact cosine-similarity scores
  2.  The cross-encoder reranking scores
  3.  The exact JSON records from mock_cctv.json that each chunk maps to
  4.  The full prompt sent to the LLM (verbatim)
  5.  The LLM response (Gemini / OpenAI / mock)
  6.  Evaluation metrics table

Usage
-----
    # Set your Gemini API key first:
    $env:GOOGLE_API_KEY = "your-api-key-here"

    python scripts/test_rag.py
    python scripts/test_rag.py --query "Did anyone enter the server room after hours?"
    python scripts/test_rag.py --config config/config.yaml --data data/mock_cctv.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # loads .env from project root automatically

# ---------------------------------------------------------------------------
# Add src/ to import path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich import box
import io

console = Console(force_terminal=True, highlight=True)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from videorag.ingestion.loader import CCTVDataLoader
from videorag.indexing.embedder import TextEmbedder
from videorag.indexing.vector_store import FAISSVectorStore
from videorag.retrieval.retriever import CCTVRetriever
from videorag.retrieval.reranker import CrossEncoderReranker, ScoreReranker
from videorag.llm.prompter import RAGPrompter, LLMClient
from videorag.evaluation.evaluator import RAGEvaluator

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Default test queries covering different incident types
# ---------------------------------------------------------------------------
DEFAULT_QUERIES = [
    "Was there any suspicious activity near the fence at night?",
    "Show me all vehicle sightings with partial plate reads",
    "When did the fight or altercation happen and which camera caught it?",
    "Was there any drone spotted near the facility?",
    "Which camera saw unauthorized access attempts or denied entry?",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_raw_json(data_path: str) -> list[dict]:
    """Load original mock_cctv.json records keyed by (camera, timestamp)."""
    with open(data_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_lookup(raw_records: list[dict]) -> dict:
    """Build a fast lookup dict: (camera, timestamp) -> record."""
    return {(r["camera"], r["timestamp"]): r for r in raw_records}


def _print_section(title: str) -> None:
    console.print()
    console.print(Rule(f"[bold yellow]{title}[/bold yellow]"))


def _score_bar(score: float, width: int = 20) -> str:
    """ASCII score bar: score expected in [-10, +10] range, normalized for display."""
    normalized = min(max((score + 10) / 20, 0.0), 1.0)
    filled = int(normalized * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ---------------------------------------------------------------------------
# Core test function
# ---------------------------------------------------------------------------

def run_test_query(
    query: str,
    retriever: CCTVRetriever,
    reranker,
    prompter: RAGPrompter,
    llm_client: LLMClient,
    evaluator: RAGEvaluator,
    top_k: int,
    rerank_top_k: int,
    raw_lookup: dict,
    show_prompt: bool = True,
) -> None:

    console.print()
    console.print(Panel(
        f"[bold white]{query}[/bold white]",
        title="[bold cyan]Query[/bold cyan]",
        border_style="cyan",
        expand=False,
    ))

    # -----------------------------------------------------------------------
    # STEP 1 — FAISS Vector Retrieval
    # -----------------------------------------------------------------------
    _print_section("STEP 1 — FAISS Vector Retrieval (raw cosine similarity scores)")

    results = retriever.retrieve(query, top_k=top_k)

    faiss_table = Table(
        title=f"Top {len(results)} FAISS Results (before reranking)",
        box=box.SIMPLE_HEAVY,
        header_style="bold blue",
        expand=True,
    )
    faiss_table.add_column("#", width=3, style="dim")
    faiss_table.add_column("Camera", style="cyan", no_wrap=True)
    faiss_table.add_column("Timestamp", style="yellow", no_wrap=True)
    faiss_table.add_column("FAISS Score", style="green", width=12)
    faiss_table.add_column("Score Bar", width=22)
    faiss_table.add_column("Description (truncated)", style="white")

    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        camera = meta.get("camera", "?")
        ts = meta.get("start_timestamp", meta.get("timestamp", "?"))
        desc = meta.get("description", "")[:80] + ("..." if len(meta.get("description", "")) > 80 else "")
        score = r.get("score", 0.0)
        faiss_table.add_row(
            str(i), camera, ts,
            f"{score:.6f}",
            _score_bar(score),
            desc,
        )

    console.print(faiss_table)

    # -----------------------------------------------------------------------
    # STEP 2 — Map chunks back to original JSON records
    # -----------------------------------------------------------------------
    _print_section("STEP 2 — Original JSON Records (from mock_cctv.json)")

    json_table = Table(
        title="Exact JSON records that the retrieved chunks map to",
        box=box.SIMPLE_HEAVY,
        header_style="bold magenta",
        expand=True,
    )
    json_table.add_column("#", width=3, style="dim")
    json_table.add_column("camera", style="cyan", no_wrap=True)
    json_table.add_column("timestamp", style="yellow", no_wrap=True)
    json_table.add_column("description (full)", style="white")
    json_table.add_column("Found in JSON?", style="green", width=14)

    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        camera = meta.get("camera", "?")
        ts = meta.get("start_timestamp", meta.get("timestamp", "?"))
        record = raw_lookup.get((camera, ts))
        if record:
            json_table.add_row(str(i), record["camera"], record["timestamp"], record["description"], "[green]YES[/green]")
        else:
            json_table.add_row(str(i), camera, ts, "[dim]Not directly matched[/dim]", "[red]NO[/red]")

    console.print(json_table)

    # -----------------------------------------------------------------------
    # STEP 3 — Cross-Encoder Reranking
    # -----------------------------------------------------------------------
    _print_section("STEP 3 — Cross-Encoder Reranking")

    for r in results:
        if "text" not in r:
            r["text"] = r.get("metadata", {}).get("text", "")

    reranked = reranker.rerank(query, results, top_k=rerank_top_k)

    rerank_table = Table(
        title=f"Top {len(reranked)} Results After Reranking",
        box=box.ROUNDED,
        header_style="bold green",
        expand=True,
    )
    rerank_table.add_column("Rank", width=5, style="dim")
    rerank_table.add_column("Camera", style="cyan", no_wrap=True)
    rerank_table.add_column("Timestamp", style="yellow", no_wrap=True)
    rerank_table.add_column("FAISS Score", style="blue", width=12)
    rerank_table.add_column("Rerank Score", style="green", width=13)
    rerank_table.add_column("Score Bar", width=22)
    rerank_table.add_column("Description", style="white")

    for rank, r in enumerate(reranked, 1):
        meta = r.get("metadata", {})
        camera = meta.get("camera", "?")
        ts = meta.get("start_timestamp", meta.get("timestamp", "?"))
        desc = meta.get("description", "")[:70] + ("..." if len(meta.get("description", "")) > 70 else "")
        faiss_score = r.get("score", 0.0)
        rerank_score = r.get("rerank_score", 0.0)
        rerank_table.add_row(
            str(rank), camera, ts,
            f"{faiss_score:.4f}",
            f"{rerank_score:.4f}",
            _score_bar(rerank_score),
            desc,
        )

    console.print(rerank_table)

    # -----------------------------------------------------------------------
    # STEP 4 — Full Prompt (verbatim)
    # -----------------------------------------------------------------------
    _print_section("STEP 4 — Full Prompt Sent to LLM")

    prompt = prompter.build_prompt(query, reranked)
    prompt_chars = len(prompt)
    prompt_tokens_est = prompt_chars // 4  # rough estimate

    if show_prompt:
        console.print(Panel(
            prompt,
            title=f"[bold]Prompt  ({prompt_chars} chars, ~{prompt_tokens_est} tokens estimated)[/bold]",
            border_style="dim white",
            expand=True,
        ))
    else:
        console.print(f"  [dim]Prompt: {prompt_chars} chars, ~{prompt_tokens_est} estimated tokens (use --show-prompt to display)[/dim]")

    # -----------------------------------------------------------------------
    # STEP 5 — LLM Response
    # -----------------------------------------------------------------------
    _print_section("STEP 5 — LLM Response")

    console.print(f"  [dim]Calling backend: [cyan]{llm_client.backend}[/cyan] | model: [cyan]{llm_client.model}[/cyan][/dim]")

    try:
        answer = llm_client.generate(prompt)
        console.print(Panel(
            answer,
            title="[bold green]LLM Answer[/bold green]",
            border_style="green",
            expand=False,
        ))
    except Exception as exc:
        console.print(Panel(
            f"[red]LLM call failed:[/red] {exc}\n\n"
            "[yellow]Tip: Make sure GOOGLE_API_KEY is set in your environment.[/yellow]\n"
            "    $env:GOOGLE_API_KEY = 'your-key-here'",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        answer = ""

    # -----------------------------------------------------------------------
    # STEP 6 — Evaluation
    # -----------------------------------------------------------------------
    _print_section("STEP 6 — Evaluation Metrics")

    stop_words = {"the", "a", "an", "is", "was", "were", "are", "in",
                  "at", "on", "of", "to", "any", "did", "do", "what",
                  "when", "where", "who", "how", "there", "near", "with",
                  "show", "me", "all", "and", "which", "caught", "it"}
    keywords = [
        w.strip("?.,!").lower()
        for w in query.split()
        if w.lower() not in stop_words and len(w) > 2
    ]
    console.print(f"  [dim]Keywords used for retrieval evaluation: {keywords}[/dim]")

    if answer:
        eval_result = evaluator.full_evaluation(query, reranked, answer, keywords)
        evaluator.print_report(eval_result)

    console.print()
    console.print(Panel(
        "[bold green]Query complete.[/bold green] See above for full pipeline trace.",
        expand=False,
        border_style="dim green",
    ))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep debug/test script for the CCTV VideoRAG pipeline."
    )
    parser.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "config" / "config.yaml"),
        help="Path to config YAML",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to mock_cctv.json (overrides config)",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Run a single query instead of the default test suite",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Hide the full prompt text (just show char/token count)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini/OpenAI API key (overrides env variable)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = _load_config(args.config)

    cfg_data = config.get("data", {})
    cfg_idx = config.get("indexing", {})
    cfg_ret = config.get("retrieval", {})
    cfg_llm = config.get("llm", {})

    data_path = args.data or cfg_data.get("mock_path", "data/mock_cctv.json")
    model_name = cfg_idx.get("model_name", "all-MiniLM-L6-v2")
    index_path = cfg_idx.get("index_save_path", "index/cctv_index")
    top_k = cfg_ret.get("top_k", 10)
    rerank_top_k = cfg_ret.get("rerank_top_k", 5)
    use_reranker = cfg_ret.get("use_reranker", True)

    llm_backend = cfg_llm.get("backend", "mock")
    llm_model = cfg_llm.get("model", "gemini-2.0-flash-lite")
    llm_api_key = args.api_key or cfg_llm.get("api_key") or None
    llm_base_url = cfg_llm.get("base_url") or None

    console.print(Panel(
        f"[bold cyan]VideoRAG — RAG Debug & Test Suite[/bold cyan]\n"
        f"[dim]Data : {data_path}\n"
        f"Index: {index_path}\n"
        f"Embed: {model_name}\n"
        f"LLM  : {llm_backend} / {llm_model}[/dim]",
        expand=False,
    ))

    # Load raw JSON for source mapping
    console.print("\n[dim]Loading raw CCTV JSON records...[/dim]")
    raw_records = _load_raw_json(data_path)
    raw_lookup = _build_lookup(raw_records)
    console.print(f"  Loaded [green]{len(raw_records)}[/green] raw records into lookup table.")

    # Load embedder + FAISS index
    console.print(f"\n[dim]Loading embedding model '{model_name}'...[/dim]")
    embedder = TextEmbedder(model_name=model_name)

    idx_path = Path(index_path)
    if not idx_path.is_absolute():
        idx_path = _PROJECT_ROOT / idx_path
    console.print(f"[dim]Loading FAISS index from '{idx_path}'...[/dim]")
    store = FAISSVectorStore(dim=embedder.dimension)
    store.load(str(idx_path))
    console.print(f"  Index: [green]{store.size}[/green] vectors, dim=[green]{embedder.dimension}[/green]")

    # Retriever
    retriever = CCTVRetriever(vector_store=store, embedder=embedder)

    # Reranker
    if use_reranker:
        console.print("\n[dim]Loading cross-encoder reranker...[/dim]")
        try:
            reranker = CrossEncoderReranker()
            console.print("  [green]CrossEncoderReranker loaded.[/green]")
        except Exception as exc:
            console.print(f"  [yellow]CrossEncoder failed ({exc}), using ScoreReranker.[/yellow]")
            reranker = ScoreReranker()
    else:
        reranker = ScoreReranker()

    # LLM
    console.print(f"\n[dim]Initialising LLM client: {llm_backend} / {llm_model}...[/dim]")
    llm_client = LLMClient(
        backend=llm_backend,
        model=llm_model,
        api_key=llm_api_key,
        base_url=llm_base_url,
    )

    prompter = RAGPrompter()
    evaluator = RAGEvaluator()

    # Queries
    queries = [args.query] if args.query else DEFAULT_QUERIES

    console.print(f"\n[bold]Running [cyan]{len(queries)}[/cyan] test queries...[/bold]")

    for i, q in enumerate(queries, 1):
        console.print()
        console.print(Rule(f"[bold white]Test {i} / {len(queries)}[/bold white]", style="white"))
        run_test_query(
            query=q,
            retriever=retriever,
            reranker=reranker,
            prompter=prompter,
            llm_client=llm_client,
            evaluator=evaluator,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            raw_lookup=raw_lookup,
            show_prompt=not args.no_prompt,
        )

    console.print()
    console.print(Panel(
        f"[bold green]All {len(queries)} test(s) complete.[/bold green]",
        expand=False,
        border_style="green",
    ))
