"""
Embedding and vector-store management using Cohere embed-v3 and ChromaDB.
"""

import logging
import os
from typing import Optional

import chromadb
import cohere

from pipeline.config import EMBED_MODEL, CHROMA_PERSIST_DIR

logger = logging.getLogger(__name__)


def _get_cohere_client() -> cohere.Client:
    """Instantiate a Cohere client from the environment."""
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise EnvironmentError("COHERE_API_KEY is not set.")
    return cohere.Client(api_key)


def _get_chroma_collection(collection_name: str) -> chromadb.Collection:
    """Return (or create) a persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def _collection_has_ticker_year(collection: chromadb.Collection, ticker: str, year: int) -> bool:
    """Return True if the collection already contains embeddings for this ticker+year."""
    results = collection.get(
        where={"$and": [{"ticker": {"$eq": ticker}}, {"filing_year": {"$eq": year}}]},
        limit=1,
    )
    return len(results["ids"]) > 0


def embed_and_store(chunks: list[dict], collection_name: str) -> None:
    """
    Embed *chunks* with Cohere embed-v3 and upsert them into a ChromaDB collection.

    Skips any (ticker, filing_year) combination that is already present in the
    collection so re-running ingestion is idempotent.

    Args:
        chunks: List of chunk dicts with keys: ticker, filing_year, section,
                chunk_index, text.
        collection_name: Name of the ChromaDB collection to write to.
    """
    if not chunks:
        logger.warning("embed_and_store called with empty chunk list — nothing to do.")
        return

    co = _get_cohere_client()
    collection = _get_chroma_collection(collection_name)

    # Group chunks by (ticker, year) to skip already-embedded combinations
    groups: dict[tuple[str, int], list[dict]] = {}
    for chunk in chunks:
        key = (chunk["ticker"], chunk["filing_year"])
        groups.setdefault(key, []).append(chunk)

    to_embed: list[dict] = []
    for (ticker, year), group_chunks in groups.items():
        if _collection_has_ticker_year(collection, ticker, year):
            logger.info(
                "Skipping embedding for %s %d — already in collection '%s'.",
                ticker,
                year,
                collection_name,
            )
        else:
            to_embed.extend(group_chunks)

    if not to_embed:
        logger.info("All chunks already embedded — nothing to do.")
        return

    texts = [c["text"] for c in to_embed]

    # Cohere embed API accepts up to 96 texts per call
    BATCH_SIZE = 96
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        logger.info(
            "Embedding batch %d-%d / %d…", i + 1, i + len(batch), len(texts)
        )
        response = co.embed(
            texts=batch,
            model=EMBED_MODEL,
            input_type="search_document",
        )
        all_embeddings.extend(response.embeddings)

    # Build ChromaDB upsert payload
    ids = [
        f"{c['ticker']}_{c['filing_year']}_{c['chunk_index']}" for c in to_embed
    ]
    metadatas = [
        {
            "ticker": c["ticker"],
            "filing_year": c["filing_year"],
            "section": c["section"],
            "chunk_index": c["chunk_index"],
        }
        for c in to_embed
    ]
    documents = texts

    collection.upsert(
        ids=ids,
        embeddings=all_embeddings,
        metadatas=metadatas,
        documents=documents,
    )
    logger.info(
        "Stored %d chunks in collection '%s'.", len(to_embed), collection_name
    )
