"""
Answer generation using Cohere Command R+ with retrieved context.
"""

import logging
import os

import cohere

from pipeline.config import GENERATION_MODEL

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a financial analyst assistant that answers questions about SEC filings.

Rules:
1. Answer ONLY using the numbered source passages provided below.
2. Cite the source numbers you relied on (e.g. [1], [2,4]).
3. If the answer cannot be determined from the provided sources, respond with
   exactly: "I cannot determine this from the provided filings."
4. Do not speculate or add information from outside the sources.
5. Be concise and precise — this is a financial context where accuracy matters.
"""


def _get_cohere_client() -> cohere.Client:
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise EnvironmentError("COHERE_API_KEY is not set.")
    return cohere.Client(api_key)


def _build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks as a numbered list for injection into the prompt."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        meta = f"{chunk.get('ticker', '?')} | {chunk.get('filing_year', '?')} | {chunk.get('section', '?')}"
        lines.append(f"[{i}] ({meta})\n{chunk['text']}")
    return "\n\n".join(lines)


def _parse_sources(answer: str, n_chunks: int) -> list[int]:
    """
    Extract cited source indices from the model's answer text.

    Looks for patterns like [1], [2,3], [1][3], etc.
    Returns a sorted, deduplicated list of 1-based source numbers within range.
    """
    import re
    raw = re.findall(r"\[(\d+(?:,\s*\d+)*)\]", answer)
    sources: set[int] = set()
    for match in raw:
        for part in match.split(","):
            try:
                n = int(part.strip())
                if 1 <= n <= n_chunks:
                    sources.add(n)
            except ValueError:
                pass
    return sorted(sources)


def generate(query: str, chunks: list[dict]) -> dict:
    """
    Generate an answer to *query* grounded in *chunks*.

    Args:
        query: The user's natural-language question.
        chunks: Retrieved chunks (as returned by retrieve()), each containing
                at minimum 'text', 'ticker', 'filing_year', 'section'.

    Returns:
        Dict with:
            answer (str): The model's response, with inline citations.
            sources (list[int]): 1-based indices of chunks the model cited.
    """
    if not chunks:
        return {
            "answer": "I cannot determine this from the provided filings.",
            "sources": [],
        }

    co = _get_cohere_client()
    context_block = _build_context_block(chunks)

    user_message = (
        f"Sources:\n\n{context_block}\n\n"
        f"Question: {query}"
    )

    response = co.chat(
        model=GENERATION_MODEL,
        preamble=_SYSTEM_PROMPT,
        message=user_message,
    )

    answer = response.text.strip()
    sources = _parse_sources(answer, len(chunks))

    logger.debug("Generated answer (%d chars), cited sources: %s", len(answer), sources)

    return {"answer": answer, "sources": sources}
