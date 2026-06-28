"""Central configuration — change experiment parameters here, not in pipeline code."""

# Cohere model identifiers
EMBED_MODEL = "embed-english-v3.0"
RERANK_MODEL = "rerank-english-v3.0"
GENERATION_MODEL = "command-r-plus-08-2024"
JUDGE_MODEL = "command-r-plus-08-2024"

# Chunking
TARGET_CHUNK_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50
# Rough character-to-token ratio for English prose
CHARS_PER_TOKEN = 4

TARGET_CHUNK_CHARS = TARGET_CHUNK_TOKENS * CHARS_PER_TOKEN      # 1600
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN    # 200

# Retrieval
VECTOR_TOP_K = 20       # candidates fetched from ChromaDB
RERANK_TOP_K = 5        # survivors after reranking

# SEC EDGAR
SEC_RATE_LIMIT_BACKOFF_BASE = 2   # seconds; doubles each retry
SEC_RATE_LIMIT_MAX_RETRIES = 5

# ChromaDB
CHROMA_PERSIST_DIR = ".chroma"

# Sections worth indexing (in order of financial relevance)
TARGET_SECTIONS = [
    "Item 1",
    "Item 1A",
    "Item 7",
    "Item 7A",
    "Item 8",
]
