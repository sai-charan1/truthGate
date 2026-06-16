#!/usr/bin/env python3
import sys
sys.path.insert(0, ".")

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

app_cli = typer.Typer(help="TruthGate CLI — RAG QA over FastAPI docs")
console = Console()


@app_cli.command()
def ask(question: str = typer.Argument(..., help="Question to ask")):
    from src.qa_engine import query
    console.print(f"\n[bold cyan]TruthGate[/bold cyan] — asking: [italic]{question}[/italic]\n")
    result = query(question)

    answer_type = result.get("answer_type", "unknown")
    color_map = {"answered": "green", "unanswerable": "yellow", "false_premise": "red"}
    color = color_map.get(answer_type, "white")

    console.print(Panel(
        f"[{color}]Type: {answer_type.upper()}[/{color}]\n\n{result.get('answer', '')}\n\n"
        f"[dim]Citations: {', '.join(result.get('citations', [])) or 'None'}[/dim]",
        title="Response",
        border_style=color,
    ))

    t = Table(box=box.SIMPLE)
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    t.add_row("Confidence", f"{result.get('confidence', 0):.2f}")
    t.add_row("Cost", f"${result.get('cost_usd', 0):.5f}")
    t.add_row("Latency", f"{result.get('latency_ms', 0)}ms")
    t.add_row("Top Retrieval Score", f"{result.get('top_retrieval_score', 0):.3f}")
    console.print(t)


@app_cli.command()
def setup():
    from src.ingestion.scraper import run_scraper
    from src.ingestion.chunker import build_chunks
    from src.ingestion.indexer import build_index

    console.print("[bold]Step 1/3: Scraping FastAPI documentation...[/bold]")
    run_scraper()
    console.print("[bold]Step 2/3: Chunking documents...[/bold]")
    build_chunks()
    console.print("[bold]Step 3/3: Building vector index...[/bold]")
    build_index()
    console.print("[green]Setup complete! Run: python cli.py ask 'your question'[/green]")


if __name__ == "__main__":
    app_cli()
