"""
SEC filing RAG pipeline — CLI entry point.

Usage:
    python main.py ingest --ticker AAPL --years 2022 2023 2024
    python main.py ask --ticker AAPL --years 2022 2023 2024 \
        --question "What was gross margin in FY2023 and how did it change YoY?"
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

load_dotenv()

console = Console()


def _collection_name(ticker: str, years: list[int]) -> str:
    """Stable, deterministic collection name for a ticker + set of years."""
    year_str = "_".join(str(y) for y in sorted(years))
    return f"{ticker.lower()}_{year_str}"


# ---------------------------------------------------------------------------
# Ingest command
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> None:
    """Download, parse, chunk, and embed SEC filings."""
    from pipeline.ingest import ingest
    from pipeline.embed import embed_and_store

    ticker = args.ticker.upper()
    years: list[int] = args.years
    collection = _collection_name(ticker, years)

    console.rule(f"[bold cyan]Ingesting {ticker} — {years}")

    with console.status(f"Downloading 10-K filings for {ticker}…"):
        chunks = ingest(ticker, years)

    if not chunks:
        console.print("[bold red]No chunks produced — check ticker and years.")
        sys.exit(1)

    console.print(f"[green]Produced {len(chunks)} chunks across {len(years)} filing(s).")

    with console.status("Embedding and storing in ChromaDB…"):
        embed_and_store(chunks, collection)

    console.print(
        Panel(
            f"[bold green]Done![/]\n\n"
            f"Collection: [cyan]{collection}[/]\n"
            f"Chunks stored: {len(chunks)}",
            title="Ingestion complete",
        )
    )


# ---------------------------------------------------------------------------
# Ask command
# ---------------------------------------------------------------------------

def cmd_ask(args: argparse.Namespace) -> None:
    """Retrieve relevant chunks and generate a grounded answer."""
    from pipeline.retrieve import retrieve
    from pipeline.generate import generate

    ticker = args.ticker.upper()
    years: list[int] = args.years
    question: str = args.question
    collection = _collection_name(ticker, years)

    console.rule(f"[bold cyan]Query: {ticker} — {years}")
    console.print(f"[bold]Question:[/] {question}\n")

    with console.status("Retrieving relevant passages…"):
        chunks = retrieve(question, collection)

    if not chunks:
        console.print(
            "[bold red]No chunks found. Have you run ingestion for this ticker/years?"
        )
        sys.exit(1)

    # Show retrieved chunks
    table = Table(
        title="Retrieved passages",
        box=box.SIMPLE_HEAD,
        show_lines=True,
    )
    table.add_column("#", style="bold cyan", width=3)
    table.add_column("Year", width=6)
    table.add_column("Section", width=12)
    table.add_column("Rerank score", width=12)
    table.add_column("Excerpt", ratio=1)

    for i, chunk in enumerate(chunks, start=1):
        excerpt = chunk["text"][:200].replace("\n", " ") + (
            "…" if len(chunk["text"]) > 200 else ""
        )
        table.add_row(
            str(i),
            str(chunk.get("filing_year", "?")),
            chunk.get("section", "?"),
            f"{chunk.get('rerank_score', 0.0):.4f}",
            excerpt,
        )

    console.print(table)

    with console.status("Generating answer…"):
        result = generate(question, chunks)

    console.print(
        Panel(
            result["answer"],
            title="[bold green]Answer",
            subtitle=f"Sources cited: {result['sources']}",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sec-rag",
        description="RAG pipeline for SEC 10-K filings powered by Cohere.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- ingest ----
    ingest_p = subparsers.add_parser("ingest", help="Download and embed SEC filings.")
    ingest_p.add_argument("--ticker", required=True, help="Stock ticker, e.g. AAPL.")
    ingest_p.add_argument(
        "--years",
        required=True,
        nargs="+",
        type=int,
        help="Fiscal years to ingest, e.g. 2022 2023 2024.",
    )

    # ---- ask ----
    ask_p = subparsers.add_parser("ask", help="Ask a question about ingested filings.")
    ask_p.add_argument("--ticker", required=True, help="Stock ticker, e.g. AAPL.")
    ask_p.add_argument(
        "--years",
        required=True,
        nargs="+",
        type=int,
        help="Fiscal years to query across.",
    )
    ask_p.add_argument("--question", required=True, help="Natural-language question.")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s | %(name)s | %(message)s",
    )

    dispatch = {"ingest": cmd_ingest, "ask": cmd_ask}
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
