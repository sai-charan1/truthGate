#!/usr/bin/env python3
import sys
sys.path.insert(0, ".")

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich import box
import numpy as np

from src.qa_engine import query

EVAL_FILE = Path("evals/eval_questions.json")
RESULTS_FILE = Path("evals/eval_results.json")

console = Console()


def run_eval(questions: List[Dict], verbose: bool = False) -> List[Dict]:
    results = []
    for i, q in enumerate(questions, 1):
        console.print(f"[{i:02d}/{len(questions)}] [{q['category']}] {q['question'][:70]}...")
        try:
            result = query(q["question"])
            predicted = result.get("answer_type", "unanswerable")
            expected = q["expected_type"]
            correct = predicted == expected
            row = {
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "expected_type": expected,
                "predicted_type": predicted,
                "correct": correct,
                "confidence": result.get("confidence", 0.0),
                "cost_usd": result.get("cost_usd", 0.0),
                "latency_ms": result.get("latency_ms", 0),
                "answer": result.get("answer", "")[:200],
                "citations": result.get("citations", []),
                "top_retrieval_score": result.get("top_retrieval_score", 0.0),
            }
            status = "✅" if correct else "❌"
            console.print(f"   {status} Expected: {expected} | Got: {predicted} | Cost: ${row['cost_usd']:.5f} | Latency: {row['latency_ms']}ms")
            if verbose and not correct:
                console.print(f"   [red]Answer: {row['answer'][:120]}[/red]")
            results.append(row)
        except Exception as e:
            console.print(f"   [red]ERROR: {e}[/red]")
            results.append({
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "expected_type": q["expected_type"],
                "predicted_type": "error",
                "correct": False,
                "confidence": 0.0,
                "cost_usd": 0.0,
                "latency_ms": 0,
                "answer": str(e),
                "citations": [],
                "top_retrieval_score": 0.0,
            })
        time.sleep(0.5)
    return results


def compute_metrics(results: List[Dict]) -> Dict:
    categories = {
        "answerable": [r for r in results if r["category"] == "answerable"],
        "unanswerable": [r for r in results if r["category"] == "unanswerable"],
        "false_premise": [r for r in results if r["category"] == "false_premise"],
        "adversarial": [r for r in results if r["category"] == "adversarial"],
    }

    def accuracy(rows):
        if not rows:
            return 0.0
        return sum(r["correct"] for r in rows) / len(rows)

    answerable_rows = categories["answerable"]
    answered_correctly = [r for r in answerable_rows if r["predicted_type"] == "answered" and r["correct"]]
    wrongly_refused = [r for r in answerable_rows if r["predicted_type"] == "unanswerable"]

    unans_rows = categories["unanswerable"]
    correctly_refused = [r for r in unans_rows if r["correct"]]
    hallucinated = [r for r in unans_rows if r["predicted_type"] == "answered"]

    refusal_tp = len(correctly_refused)
    refusal_fp = len(wrongly_refused)
    refusal_fn = len(hallucinated)
    refusal_precision = refusal_tp / (refusal_tp + refusal_fp) if (refusal_tp + refusal_fp) > 0 else 0.0
    refusal_recall = refusal_tp / (refusal_tp + refusal_fn) if (refusal_tp + refusal_fn) > 0 else 0.0

    fp_rows = categories["false_premise"]
    fp_detection_rate = accuracy(fp_rows)

    costs = [r["cost_usd"] for r in results if r["cost_usd"] > 0]
    latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]

    overall_acc = sum(r["correct"] for r in results) / len(results) if results else 0.0

    return {
        "total": len(results),
        "overall_accuracy": overall_acc,
        "answer_accuracy": accuracy(answerable_rows),
        "refusal_precision": refusal_precision,
        "refusal_recall": refusal_recall,
        "false_premise_detection_rate": fp_detection_rate,
        "adversarial_accuracy": accuracy(categories["adversarial"]),
        "hallucination_rate": len(hallucinated) / len(unans_rows) if unans_rows else 0.0,
        "mean_cost_usd": np.mean(costs) if costs else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "total_cost_usd": sum(costs),
        "per_category": {cat: {"count": len(rows), "accuracy": accuracy(rows)} for cat, rows in categories.items()},
    }


def print_metrics_table(metrics: Dict):
    console.print("\n")
    t = Table(title="TruthGate Eval Results", box=box.ROUNDED, show_header=True, header_style="bold cyan")
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")

    t.add_row("Total Questions", str(metrics["total"]))
    t.add_row("Overall Accuracy", f"{metrics['overall_accuracy']:.1%}")
    t.add_row("─" * 30, "─" * 15)
    t.add_row("Answer Accuracy (answerable)", f"{metrics['answer_accuracy']:.1%}")
    t.add_row("Refusal Precision", f"{metrics['refusal_precision']:.1%}")
    t.add_row("Refusal Recall", f"{metrics['refusal_recall']:.1%}")
    t.add_row("False-Premise Detection Rate", f"{metrics['false_premise_detection_rate']:.1%}")
    t.add_row("Adversarial Accuracy", f"{metrics['adversarial_accuracy']:.1%}")
    t.add_row("Hallucination Rate", f"{metrics['hallucination_rate']:.1%}")
    t.add_row("─" * 30, "─" * 15)
    t.add_row("Mean Cost / Query", f"${metrics['mean_cost_usd']:.5f}")
    t.add_row("Total Cost", f"${metrics['total_cost_usd']:.4f}")
    t.add_row("p95 Latency", f"{metrics['p95_latency_ms']:.0f}ms")

    console.print(t)

    console.print("\n[bold]Per-Category Breakdown:[/bold]")
    cat_table = Table(box=box.SIMPLE)
    cat_table.add_column("Category")
    cat_table.add_column("Count", justify="right")
    cat_table.add_column("Accuracy", justify="right")
    for cat, data in metrics["per_category"].items():
        cat_table.add_row(cat, str(data["count"]), f"{data['accuracy']:.1%}")
    console.print(cat_table)


def main():
    parser = argparse.ArgumentParser(description="Run TruthGate evaluation harness")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--category", "-c", help="Filter to specific category")
    parser.add_argument("--limit", "-n", type=int, help="Limit number of questions")
    args = parser.parse_args()

    with open(EVAL_FILE) as f:
        questions = json.load(f)

    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
    if args.limit:
        questions = questions[:args.limit]

    console.print(f"\n[bold cyan]TruthGate Evaluation Harness[/bold cyan]")
    console.print(f"Running {len(questions)} questions...\n")

    t0 = time.time()
    results = run_eval(questions, verbose=args.verbose)
    elapsed = time.time() - t0

    metrics = compute_metrics(results)
    print_metrics_table(metrics)

    console.print(f"\n[dim]Total wall time: {elapsed:.1f}s[/dim]")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)
    console.print(f"[dim]Full results saved to {RESULTS_FILE}[/dim]\n")


if __name__ == "__main__":
    main()
