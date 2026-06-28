"""
Two-stage retrieval: dense vector search via ChromaDB, then Cohere reranking.
"""

import logging
import os

import chromadb
import cohere

from pipeline.config import (
    EMBED_MODEL,
    RERANK_MODEL,
    CHROMA_PERSIST_DIR,
    VECTOR_TOP_K,
    RERANK_TOP_K,
)

logger = logging.getLogger(__name__)


def _get_cohere_client() -> cohere.Client:
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise EnvironmentError("COHERE_API_KEY is not set.")
    return cohere.Client(api_key)


def _get_chroma_collection(collection_name: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_collection(name=collection_name)


def retrieve(query: str, collection_name: str) -> list[dict]:
    """
    Retrieve the most relevant chunks for *query* from *collection_name*.

    Steps:
        1. Embed the query with Cohere embed-v3 (input_type="search_query").
        2. Fetch the top-VECTOR_TOP_K candidates from ChromaDB by cosine similarity.
        3. Rerank candidates to top-RERANK_TOP_K using Cohere rerank API.

    Returns:
        List of up to RERANK_TOP_K dicts, each containing:
            text, ticker, filing_year, section, chunk_index, rerank_score
    """
    co = _get_cohere_client()
    collection = _get_chroma_collection(collection_name)

    # Step 1: embed the query
    embed_response = co.embed(
        texts=[query],
        model=EMBED_MODEL,
        input_type="search_query",
    )
    query_embedding = embed_response.embeddings[0]

    # Step 2: vector search
    n_results = min(VECTOR_TOP_K, collection.count())
    if n_results == 0:
        logger.warning("Collection '%s' is empty.", collection_name)
        return []

    search_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    documents: list[str] = search_results["documents"][0]
    metadatas: list[dict] = search_results["metadatas"][0]

    if not documents:
        return []

    # Step 3: rerank
    rerank_response = co.rerank(
        query=query,
        documents=documents,
        model=RERANK_MODEL,
        top_n=RERANK_TOP_K,
    )

    results: list[dict] = []
    for hit in rerank_response.results:
        idx = hit.index
        results.append(
            {
                "text": documents[idx],
                "rerank_score": hit.relevance_score,
                **metadatas[idx],
            }
        )

    return results
