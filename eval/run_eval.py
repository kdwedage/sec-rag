"""
Evaluation harness for the SEC filing RAG pipeline.

Loads golden_dataset.json, runs the full retrieve+generate pipeline for each
question, scores retrieval (precision@5) and answer quality (LLM-as-judge),
and saves results to eval/results/{timestamp}.json.

Usage:
    python eval/run_eval.py --ticker AAPL --years 2022 2023 2024
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import cohere
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import box

# Allow running directly from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.retrieve import retrieve
from pipeline.generate import generate
from pipeline.config import JUDGE_MODEL

load_dotenv()

console = Console()
logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of financial question-answering systems.
You will be given a question, retrieved context passages, a generated answer,
and an expected (gold-standard) answer.

Score the generated answer on three dimensions, each 1–5:

1. factual_correctness: Does the generated answer agree with the expected answer
   on all key facts (numbers, percentages, years, direction of change)?
   5 = perfect match, 1 = completely wrong.

2. faithfulness: Is the generated answer fully supported by the retrieved context?
   Does it avoid hallucinating facts not present in the context?
   5 = entirely grounded, 1 = significant hallucination.

3. appropriate_uncertainty: When the context is insufficient, does the model
   correctly say it cannot determine the answer rather than guessing?
   5 = perfectly calibrated, 1 = confidently wrong when uncertain.

Respond ONLY with valid JSON in exactly this format, no extra text:
{
  "factual_correctness": <1-5>,
  "faithfulness": <1-5>,
  "appropriate_uncertainty": <1-5>,
  "reasoning": "<one sentence explaining the scores>"
}
"""


def _get_cohere_client() -> cohere.Client:
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise EnvironmentError("COHERE_API_KEY is not set.")
    return cohere.Client(api_key)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_retrieval_precision(
    chunks: list[dict], source_passage: str
) -> float:
    """
    Precision@5: return 1.0 if any of the top-5 retrieved chunks contains
    the gold source_passage as a substring, else 0.0.
    """
    if source_passage in ("PLACEHOLDER", "", None):
        return -1.0  # sentinel: question not yet filled in
    needle = source_passage.strip().lower()
    for chunk in chunks:
        if needle in chunk.get("text", "").lower():
            return 1.0
    return 0.0


def score_faithfulness_with_judge(
    co: cohere.Client,
    question: str,
    chunks: list[dict],
    generated_answer: str,
    expected_answer: str,
) -> dict:
    """
    Call Cohere Command R+ as an LLM judge and return structured scores.

    Returns a dict with factual_correctness, faithfulness, appropriate_uncertainty
    (each 1-5) and a reasoning string.  On parse failure returns -1 for all scores.
    """
    context_block = "\n\n".join(
        f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)
    )
    user_msg = (
        f"Question: {question}\n\n"
        f"Expected answer: {expected_answer}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        f"Generated answer: {generated_answer}"
    )

    try:
        response = co.chat(
            model=JUDGE_MODEL,
            preamble=_JUDGE_SYSTEM_PROMPT,
            message=user_msg,
        )
        raw = response.text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        scores = json.loads(raw)
        return {
            "factual_correctness": int(scores.get("factual_correctness", -1)),
            "faithfulness": int(scores.get("faithfulness", -1)),
            "appropriate_uncertainty": int(scores.get("appropriate_uncertainty", -1)),
            "reasoning": scores.get("reasoning", ""),
        }
    except Exception as exc:
        logger.warning("Judge scoring failed: %s", exc)
        return {
            "factual_correctness": -1,
            "faithfulness": -1,
            "appropriate_uncertainty": -1,
            "reasoning": f"Parse error: {exc}",
        }


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(ticker: str, years: list[int]) -> None:
    """
    Run the full eval harness: load questions, retrieve, generate, score, report.
    """
    # Build collection name consistent with main.py
    year_str = "_".join(str(y) for y in sorted(years))
    collection_name = f"{ticker.lower()}_{year_str}"

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    questions = dataset["questions"]
    co = _get_cohere_client()

    all_results: list[dict] = []
    per_question_rows: list[dict] = []

    for entry in questions:
        qid = entry["id"]
        question = entry["question"]
        expected = entry["expected_answer"]
        source_passage = entry["source_passage"]

        if question == "PLACEHOLDER":
            console.print(f"[yellow]Skipping {qid} — placeholder not yet filled in.")
            continue

        console.print(f"[bold cyan]Evaluating {qid}: {question[:60]}…")

        # Retrieve
        chunks = retrieve(question, collection_name)

        # Score retrieval
        precision = score_retrieval_precision(chunks, source_passage)

        # Generate
        gen_result = generate(question, chunks)
        answer = gen_result["answer"]

        # Check if model correctly deferred
        could_not_determine = "cannot determine" in answer.lower()

        # Judge
        judge_scores = score_faithfulness_with_judge(
            co, question, chunks, answer, expected
        )

        row = {
            "id": qid,
            "question": question,
            "expected_answer": expected,
            "generated_answer": answer,
            "sources_cited": gen_result["sources"],
            "precision_at_5": precision,
            "could_not_determine": could_not_determine,
            **{f"judge_{k}": v for k, v in judge_scores.items()},
        }
        all_results.append(row)
        per_question_rows.append(row)

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    table = Table(title="Eval results", box=box.SIMPLE_HEAD, show_lines=False)
    table.add_column("ID", style="bold cyan", width=4)
    table.add_column("Precision@5", width=12)
    table.add_column("Factual", width=8)
    table.add_column("Faithful", width=9)
    table.add_column("Uncertainty", width=11)
    table.add_column("Could not det.", width=14)

    valid_precision = [r["precision_at_5"] for r in per_question_rows if r["precision_at_5"] >= 0]
    valid_factual = [r["judge_factual_correctness"] for r in per_question_rows if r["judge_factual_correctness"] > 0]
    valid_faithful = [r["judge_faithfulness"] for r in per_question_rows if r["judge_faithfulness"] > 0]
    valid_uncertainty = [r["judge_appropriate_uncertainty"] for r in per_question_rows if r["judge_appropriate_uncertainty"] > 0]
    n_could_not = sum(1 for r in per_question_rows if r["could_not_determine"])

    for r in per_question_rows:
        prec = f"{r['precision_at_5']:.1f}" if r["precision_at_5"] >= 0 else "N/A"
        factual = str(r["judge_factual_correctness"]) if r["judge_factual_correctness"] > 0 else "err"
        faithful = str(r["judge_faithfulness"]) if r["judge_faithfulness"] > 0 else "err"
        uncert = str(r["judge_appropriate_uncertainty"]) if r["judge_appropriate_uncertainty"] > 0 else "err"
        cnd = "[green]yes" if r["could_not_determine"] else "no"
        table.add_row(r["id"], prec, factual, faithful, uncert, cnd)

    console.print(table)

    # Aggregate
    def _mean(vals: list) -> str:
        return f"{sum(vals)/len(vals):.3f}" if vals else "N/A"

    console.print(
        f"\n[bold]Aggregates[/] ({len(per_question_rows)} questions evaluated)\n"
        f"  Mean precision@5:              {_mean(valid_precision)}\n"
        f"  Mean judge factual_correctness: {_mean(valid_factual)} / 5\n"
        f"  Mean judge faithfulness:        {_mean(valid_faithful)} / 5\n"
        f"  Mean appropriate_uncertainty:   {_mean(valid_uncertainty)} / 5\n"
        f"  Questions with 'cannot determine': {n_could_not} / {len(per_question_rows)}"
    )

    # ----------------------------------------------------------------
    # Persist results
    # ----------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_path = RESULTS_DIR / f"{timestamp}.json"

    payload = {
        "run_timestamp": timestamp,
        "ticker": ticker,
        "years": years,
        "collection_name": collection_name,
        "aggregates": {
            "n_questions": len(per_question_rows),
            "mean_precision_at_5": sum(valid_precision) / len(valid_precision) if valid_precision else None,
            "mean_factual_correctness": sum(valid_factual) / len(valid_factual) if valid_factual else None,
            "mean_faithfulness": sum(valid_faithful) / len(valid_faithful) if valid_faithful else None,
            "mean_appropriate_uncertainty": sum(valid_uncertainty) / len(valid_uncertainty) if valid_uncertainty else None,
            "n_could_not_determine": n_could_not,
        },
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    console.print(f"\n[dim]Results saved to {output_path}[/]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval harness for the SEC RAG pipeline.")
    parser.add_argument("--ticker", required=True, help="Ticker whose collection to evaluate against.")
    parser.add_argument("--years", required=True, nargs="+", type=int, help="Years the collection covers.")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s | %(name)s | %(message)s",
    )

    run_eval(args.ticker.upper(), args.years)


if __name__ == "__main__":
    main()
