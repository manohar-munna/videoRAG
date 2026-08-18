"""
evaluator.py
------------
Lightweight retrieval and answer quality evaluators for the CCTV RAG
pipeline.  Metrics are computed without ground-truth labels (heuristic)
and are intended for rapid development feedback.
"""

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _text_of(result: dict) -> str:
    """Extract displayable text from a retrieval result."""
    return result.get("text") or result.get("metadata", {}).get("description", "")


def _contains_any_keyword(text: str, keywords: List[str]) -> bool:
    """Return True if *text* contains any keyword (case-insensitive)."""
    lower = text.lower()
    return any(kw.lower() in lower for kw in keywords)


class RAGEvaluator:
    """Computes retrieval and answer quality metrics for a RAG pipeline.

    All metrics are heuristic estimates that do not require ground-truth
    relevance labels — they are based on keyword matching and structural
    analysis of the answer text.
    """

    # ------------------------------------------------------------------
    # Retrieval metrics
    # ------------------------------------------------------------------

    def evaluate_retrieval(
        self,
        query: str,
        retrieved: List[dict],
        relevant_keywords: List[str],
        answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute retrieval quality metrics for multimodal and text CCTV RAG.

        Args:
            query: The original query string.
            retrieved: Ordered list of retrieval result dicts.
            relevant_keywords: Keywords that a relevant result should contain.
            answer: Optional VLM answer to check for negative visual detection.

        Returns:
            A dict with keys: precision_at_k, recall_estimate, mrr, ndcg_at_k, num_retrieved, num_relevant.
        """
        logger.info(
            "evaluate_retrieval: query='%s', k=%d, keywords=%s",
            query[:60],
            len(retrieved),
            relevant_keywords,
        )

        if not retrieved:
            return {
                "precision_at_k": 0.0,
                "recall_estimate": 0.0,
                "mrr": 0.0,
                "ndcg_at_k": 0.0,
                "num_retrieved": 0,
                "num_relevant": 0,
            }

        # Check if the VLM forensic answer explicitly negated presence of the queried target
        negation_phrases = [
            "no evidence", "not observed", "none are visibly", "no person",
            "cannot be substantiated", "not visible", "no vehicle", "not seen",
            "no sign of", "no individual", "none visibly"
        ]
        is_negative_detection = (
            any(p in answer.lower() for p in negation_phrases)
            if answer else False
        )

        relevance: List[int] = []
        for r in retrieved:
            if is_negative_detection:
                relevance.append(0)
            else:
                score = float(r.get("score") or r.get("metadata", {}).get("score", 0.0))
                text = _text_of(r)
                has_kw = _contains_any_keyword(text, relevant_keywords) if relevant_keywords else False
                is_rel = (score >= 0.15) or has_kw
                relevance.append(1 if is_rel else 0)

        num_relevant = sum(relevance)
        k = len(retrieved)
        precision_at_k = num_relevant / k if k > 0 else 0.0
        recall_estimate = min(1.0, precision_at_k * 1.2)

        # MRR
        mrr = 0.0
        for rank, rel in enumerate(relevance, start=1):
            if rel == 1:
                mrr = 1.0 / rank
                break

        # NDCG
        dcg = sum(
            rel / math.log2(rank + 1)
            for rank, rel in enumerate(relevance, start=1)
        )
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum(
            rel / math.log2(rank + 1)
            for rank, rel in enumerate(ideal_relevance, start=1)
        )
        ndcg_at_k = dcg / idcg if idcg > 0 else 0.0

        metrics = {
            "precision_at_k": round(precision_at_k, 4),
            "recall_estimate": round(recall_estimate, 4),
            "mrr": round(mrr, 4),
            "ndcg_at_k": round(ndcg_at_k, 4),
            "num_retrieved": k,
            "num_relevant": num_relevant,
        }
        logger.debug("Retrieval metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Answer quality metrics
    # ------------------------------------------------------------------

    def evaluate_answer(
        self,
        query: str,
        answer: str,
        retrieved: List[dict],
    ) -> Dict[str, Any]:
        """Compute answer quality metrics.

        Args:
            query: The original query.
            answer: The LLM-generated answer string.
            retrieved: The list of retrieved chunks used to build the
                prompt (used to compute context utilisation).

        Returns:
            A dict with keys:

            * ``answer_length`` – character count of *answer*.
            * ``context_utilization`` – fraction of retrieved chunks
              whose camera or description text appears in the answer.
            * ``has_timestamp`` – True if the answer contains any digit
              sequence that looks like a timestamp (``HH:MM`` or
              ``YYYY-MM-DD``).
            * ``has_camera`` – True if the answer mentions any camera ID
              found in *retrieved*.
        """
        logger.info("evaluate_answer: query='%s'", query[:60])

        answer_lower = answer.lower()

        # Context utilisation
        referenced = 0
        for r in retrieved:
            meta = r.get("metadata", {})
            camera = meta.get("camera", "")
            description = meta.get("description", _text_of(r))
            if camera and camera.lower() in answer_lower:
                referenced += 1
            elif description and description[:30].lower() in answer_lower:
                referenced += 1

        context_utilization = (
            round(referenced / len(retrieved), 4) if retrieved else 0.0
        )

        # Timestamp detection (HH:MM, YYYY-MM-DD, or YYYY-MM-DDTHH:MM)
        import re
        has_timestamp = bool(
            re.search(r"\d{2}:\d{2}", answer)
            or re.search(r"\d{4}-\d{2}-\d{2}", answer)
        )

        # Camera mention
        cameras_in_retrieved = {
            r.get("metadata", {}).get("camera", "").lower()
            for r in retrieved
        } - {""}
        has_camera = any(cam in answer_lower for cam in cameras_in_retrieved)

        metrics = {
            "answer_length": len(answer),
            "context_utilization": context_utilization,
            "has_timestamp": has_timestamp,
            "has_camera": has_camera,
        }
        logger.debug("Answer metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Combined evaluation
    # ------------------------------------------------------------------

    def full_evaluation(
        self,
        query: str,
        retrieved: List[dict],
        answer: str,
        relevant_keywords: List[str],
    ) -> Dict[str, Any]:
        """Run both retrieval and answer evaluations and combine results.

        Args:
            query: The natural-language query.
            retrieved: Retrieved chunk results.
            answer: The LLM-generated answer.
            relevant_keywords: Keywords used for retrieval evaluation.

        Returns:
            A combined dict with ``retrieval`` and ``answer`` sub-dicts,
            plus a top-level ``query`` key.
        """
        retrieval_metrics = self.evaluate_retrieval(query, retrieved, relevant_keywords, answer=answer)
        answer_metrics = self.evaluate_answer(query, answer, retrieved)
        return {
            "query": query,
            "retrieval": retrieval_metrics,
            "answer": answer_metrics,
        }

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------

    def print_report(self, eval_dict: Dict[str, Any]) -> None:
        """Pretty-print an evaluation report to stdout.

        Uses ``rich`` if available, otherwise falls back to ``tabulate``,
        and finally to plain text.

        Args:
            eval_dict: The dict returned by :meth:`full_evaluation`.
        """
        query = eval_dict.get("query", "")
        retrieval = eval_dict.get("retrieval", {})
        answer = eval_dict.get("answer", {})

        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box

            console = Console()
            console.rule(f"[bold cyan]Evaluation Report[/bold cyan]")
            console.print(f"[bold]Query:[/bold] {query}\n")

            # Retrieval table
            ret_table = Table(title="Retrieval Metrics", box=box.ROUNDED, show_header=True)
            ret_table.add_column("Metric", style="cyan", no_wrap=True)
            ret_table.add_column("Value", style="green")
            for key, val in retrieval.items():
                ret_table.add_row(key, str(val))
            console.print(ret_table)

            # Answer table
            ans_table = Table(title="Answer Metrics", box=box.ROUNDED, show_header=True)
            ans_table.add_column("Metric", style="cyan", no_wrap=True)
            ans_table.add_column("Value", style="green")
            for key, val in answer.items():
                ans_table.add_row(key, str(val))
            console.print(ans_table)
            return

        except ImportError:
            pass

        try:
            from tabulate import tabulate

            print(f"\n{'='*50}")
            print(f"Evaluation Report")
            print(f"Query: {query}")
            print(f"\n--- Retrieval Metrics ---")
            print(tabulate(retrieval.items(), headers=["Metric", "Value"], tablefmt="rounded_outline"))
            print(f"\n--- Answer Metrics ---")
            print(tabulate(answer.items(), headers=["Metric", "Value"], tablefmt="rounded_outline"))
            print(f"{'='*50}\n")
            return

        except ImportError:
            pass

        # Plain-text fallback
        print(f"\nEvaluation Report\nQuery: {query}")
        print("Retrieval:", retrieval)
        print("Answer:", answer)
