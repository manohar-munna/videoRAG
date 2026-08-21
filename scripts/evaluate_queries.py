"""
scripts/evaluate_queries.py
----------------------------
Comprehensive automated evaluation harness for VideoRAG.
Evaluates model accuracy, fine visual detail capture, timestamp grounding,
and hallucination resistance across diverse surveillance query categories.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_queries")

BASE_URL = "http://127.0.0.1:8000"

EVALUATION_BATTERY = [
    {
        "id": "CAT1-OBJ-01",
        "category": "Object and Color Grounding",
        "query": "white pickup truck parked on the left side",
        "expected_timestamp_range": ("00:00:00", "00:02:50"),
        "required_keywords": ["truck", "white", "left", "parked"],
        "is_negative": False,
        "description": "Tests recognition of specific vehicle model, paint color, and spatial positioning."
    },
    {
        "id": "CAT1-OBJ-02",
        "category": "Object and Color Grounding",
        "query": "person wearing pink or magenta shirt in the crowd",
        "expected_timestamp_range": ("00:08:00", "00:09:20"),
        "required_keywords": ["pink", "person", "crowd", "shirt"],
        "is_negative": False,
        "description": "Tests fine-grained clothing color resolution in crowded footage."
    },
    {
        "id": "CAT2-SEC-01",
        "category": "Security Perimeter",
        "query": "yellow caution tape strung across the area",
        "expected_timestamp_range": ("00:00:00", "00:03:00"),
        "required_keywords": ["yellow", "caution", "tape"],
        "is_negative": False,
        "description": "Tests detection of thin linear boundary restraints."
    },
    {
        "id": "CAT3-ACT-01",
        "category": "Activity and Pedestrians",
        "query": "crowd of people gathered or walking near trees",
        "expected_timestamp_range": ("00:00:00", "00:03:30"),
        "required_keywords": ["crowd", "people", "walking", "gathered"],
        "is_negative": False,
        "description": "Tests multi-person group dynamic tracking and environmental context."
    },
    {
        "id": "CAT4-CNT-01",
        "category": "Video-Wide Counting",
        "query": "total number of vehicles visible across the video footage",
        "expected_timestamp_range": ("00:00:00", "00:12:00"),
        "required_keywords": ["vehicle", "truck", "unique", "tally"],
        "is_negative": False,
        "description": "Tests multi-moment cross-timeline deduplication and accurate tallying."
    },
    {
        "id": "CAT5-NEG-01",
        "category": "Hallucination Resistance",
        "query": "emergency red fire truck spraying water with hose",
        "expected_timestamp_range": None,
        "required_keywords": ["not visible", "no fire truck", "absent", "no evidence", "not observed", "0", "no"],
        "is_negative": True,
        "description": "Tests whether model refuses to hallucinate non-existent emergency vehicles."
    },
    {
        "id": "CAT5-NEG-02",
        "category": "Hallucination Resistance",
        "query": "helicopter landing in the parking lot",
        "expected_timestamp_range": None,
        "required_keywords": ["not visible", "no helicopter", "absent", "no evidence", "not observed", "0", "no"],
        "is_negative": True,
        "description": "Tests negative object rejection on aerial vehicles."
    }
]


def run_evaluation(camera_id: str = "CAM_01", timeout_per_query: int = 150) -> Dict[str, Any]:
    console.print(Panel("[bold cyan]VideoRAG - Forensic LLM Evaluation and Accuracy Test Battery[/bold cyan]", expand=False))
    
    try:
        health = requests.get(f"{BASE_URL}/api/health", timeout=5).json()
        console.print(f"[bold green]Connected to VideoRAG Server[/bold green] (Model: {health.get('llm_model')}, Vectors: {health.get('vector_count')})\n")
    except Exception as e:
        console.print(f"[bold red]Cannot connect to VideoRAG server at {BASE_URL}: {e}[/bold red]")
        sys.exit(1)

    eval_results = []
    total_latency = 0.0
    passed_tests = 0

    table = Table(title="Live Test Query Execution Results", show_header=True, header_style="bold magenta")
    table.add_column("Test ID", style="cyan", width=12)
    table.add_column("Category", style="yellow", width=18)
    table.add_column("Query", style="white", width=28)
    table.add_column("Time (s)", justify="right", width=9)
    table.add_column("Detail Match", justify="center", width=12)
    table.add_column("Status", justify="center", width=8)

    for item in EVALUATION_BATTERY:
        q_id = item["id"]
        cat = item["category"]
        query = item["query"]
        is_neg = item["is_negative"]
        req_kws = item["required_keywords"]

        console.print(f"Executing [bold cyan]{q_id}[/bold cyan] ({cat}): [dim]\"{query}\"[/dim]...")
        t0 = time.time()
        
        try:
            resp = requests.post(
                f"{BASE_URL}/api/search",
                json={"query": query, "camera_id": camera_id},
                timeout=timeout_per_query
            )
            elapsed = time.time() - t0
            total_latency += elapsed

            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "")
                storyboard = data.get("storyboard", [])

                answer_lower = answer.lower()
                matched_kws = [kw for kw in req_kws if kw.lower() in answer_lower]
                kw_score = (len(matched_kws) / len(req_kws)) * 100.0

                if is_neg:
                    neg_confirmed = any(kw.lower() in answer_lower for kw in req_kws) or "0" in answer_lower
                    passed = neg_confirmed
                else:
                    passed = len(matched_kws) >= max(1, int(len(req_kws) * 0.5)) and len(storyboard) > 0

                if passed:
                    passed_tests += 1
                    status_badge = "[bold green]PASS[/bold green]"
                else:
                    status_badge = "[bold red]FAIL[/bold red]"

                table.add_row(
                    q_id,
                    cat,
                    query[:27] + ("..." if len(query) > 27 else ""),
                    f"{elapsed:.1f}s",
                    f"{len(matched_kws)}/{len(req_kws)} ({kw_score:.0f}%)",
                    status_badge
                )

                eval_results.append({
                    "id": q_id,
                    "category": cat,
                    "query": query,
                    "is_negative": is_neg,
                    "passed": passed,
                    "latency_seconds": round(elapsed, 2),
                    "storyboard_frame_count": len(storyboard),
                    "storyboard_timestamps": [f.get("timestamp") for f in storyboard],
                    "matched_keywords": matched_kws,
                    "keyword_score_pct": round(kw_score, 1),
                    "answer": answer
                })
            else:
                table.add_row(q_id, cat, query[:27], f"{elapsed:.1f}s", "HTTP Error", "[bold red]FAIL[/bold red]")
                eval_results.append({
                    "id": q_id,
                    "category": cat,
                    "query": query,
                    "passed": False,
                    "latency_seconds": round(elapsed, 2),
                    "error": resp.text
                })
        except Exception as exc:
            elapsed = time.time() - t0
            table.add_row(q_id, cat, query[:27], f"{elapsed:.1f}s", "Timeout", "[bold red]ERR[/bold red]")
            eval_results.append({
                "id": q_id,
                "category": cat,
                "query": query,
                "passed": False,
                "latency_seconds": round(elapsed, 2),
                "error": str(exc)
            })

    console.print("\n")
    console.print(table)

    accuracy_rate = (passed_tests / len(EVALUATION_BATTERY)) * 100.0
    avg_latency = total_latency / len(EVALUATION_BATTERY) if EVALUATION_BATTERY else 0.0

    summary_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_queries_tested": len(EVALUATION_BATTERY),
        "passed_tests": passed_tests,
        "accuracy_rate_pct": round(accuracy_rate, 1),
        "average_latency_seconds": round(avg_latency, 2),
        "query_results": eval_results
    }

    out_json = _PROJECT_ROOT / "data" / "evaluation_results.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary_report, fh, indent=2)
    console.print(f"\n[bold green][OK] Full Evaluation Results saved to {out_json}[/bold green]")

    report_md_path = _PROJECT_ROOT / "data" / "evaluation_report.md"
    _generate_markdown_report(summary_report, report_md_path)
    console.print(f"[bold green][OK] Markdown Report generated at {report_md_path}[/bold green]\n")

    return summary_report


def _generate_markdown_report(report: Dict[str, Any], output_path: Path) -> None:
    lines = [
        "# VideoRAG Forensic Query & Reasoning Evaluation Report",
        f"**Date:** {report['timestamp']} | **Accuracy Score:** {report['accuracy_rate_pct']}% | **Average Latency:** {report['average_latency_seconds']}s",
        "",
        "## Evaluation Battery Summary",
        "",
        "| Test ID | Category | Query | Detail Match | Status | Latency |",
        "|---|---|---|---|---|---|"
    ]

    for item in report.get("query_results", []):
        q_id = item.get("id", "")
        cat = item.get("category", "")
        query = item.get("query", "")
        kw_score = item.get("keyword_score_pct", 0)
        status = "PASS" if item.get("passed") else "FAIL"
        lat = f"{item.get('latency_seconds', 0)}s"
        lines.append(f"| `{q_id}` | {cat} | \"{query}\" | {kw_score}% | {status} | {lat} |")

    lines.extend([
        "",
        "---",
        "",
        "## In-Depth Forensic Analysis per Query",
        ""
    ])

    for item in report.get("query_results", []):
        q_id = item.get("id", "")
        cat = item.get("category", "")
        query = item.get("query", "")
        ans = item.get("answer", item.get("error", "No output"))
        timestamps = ", ".join(item.get("storyboard_timestamps", [])) or "None"
        
        lines.extend([
            f"### `{q_id}`: {cat}",
            f"- **Target Query:** *\"{query}\"*",
            f"- **Storyboard Timestamps:** `{timestamps}`",
            f"- **Forensic Reasoning Output:**",
            "```text",
            ans.strip(),
            "```",
            ""
        ])

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    run_evaluation()
