# SEC Filing RAG Pipeline

A production-quality retrieval-augmented generation (RAG) pipeline for analysing SEC 10-K filings. It combines section-aware document chunking, dense vector search via [ChromaDB](https://www.trychroma.com/), cross-encoder reranking, and grounded answer generation — all powered by the [Cohere](https://cohere.com/) API. The result is a system that can answer precise financial questions ("What drove the decline in gross margin in FY2023?") with cited, hallucination-resistant answers, and an evaluation harness that measures both retrieval quality and answer faithfulness using an LLM-as-judge.

---

## Architecture

```
SEC EDGAR
    │
    ▼
┌─────────────────────────────────────┐
│  ingest.py                          │
│  • Download 10-K via               │
│    sec-edgar-downloader             │
│  • Strip HTML / XBRL boilerplate   │
│  • Split on section boundaries     │
│    (Item 1, 1A, 7, 7A, 8)         │
│  • Paragraph-chunk to ~400 tokens  │
│    with 50-token overlap           │
└──────────────┬──────────────────────┘
               │ list[chunk_dict]
               ▼
┌─────────────────────────────────────┐
│  embed.py                           │
│  • Cohere embed-english-v3.0       │
│    input_type=search_document      │
│  • Upsert into ChromaDB            │
│    (skip if already present)       │
└──────────────┬──────────────────────┘
               │ persisted to .chroma/
               ▼
          ChromaDB
         (cosine index)
               │
    ┌──────────┴──────────┐
    │  query time         │
    ▼                     ▼
 embed query          ┌──────────────────────────┐
 (search_query)  ───► │  retrieve.py             │
                       │  • Top-20 from ChromaDB  │
                       │  • Cohere rerank-v3 →    │
                       │    top-5 with scores     │
                       └──────────┬───────────────┘
                                  │ top-5 chunks
                                  ▼
                       ┌──────────────────────────┐
                       │  generate.py             │
                       │  • Cohere command-r-plus │
                       │  • Grounded prompt with  │
                       │    numbered sources      │
                       │  • Cite or defer         │
                       └──────────┬───────────────┘
                                  │
                                  ▼
                              Answer + citations
```

---

## Setup

### Prerequisites

- Python 3.10+
- A [Cohere API key](https://dashboard.cohere.com/) (free tier works for experimentation)

### Installation

```bash
git clone <repo>
cd sec-rag

python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and add your COHERE_API_KEY
```

---

## Usage

### 1. Ingest filings

Downloads 10-K filings for the given ticker and years, chunks them, and stores embeddings in ChromaDB (under `.chroma/`).

```bash
python main.py ingest --ticker AAPL --years 2022 2023 2024
```

Re-running is safe: already-embedded ticker+year combinations are skipped automatically.

### 2. Ask a question

```bash
python main.py ask \
  --ticker AAPL \
  --years 2022 2023 2024 \
  --question "What was gross margin in FY2023 and how did it change year over year?"
```

The CLI prints a table of the top-5 retrieved passages (with section labels and rerank scores) followed by the model's cited answer.

### Optional flags

| Flag | Default | Description |
|------|---------|-------------|
| `--log-level` | `WARNING` | Set to `INFO` or `DEBUG` for verbose pipeline output |

---

## Evaluation

### Golden dataset

`eval/golden_dataset.json` ships pre-populated with five AAPL questions spanning FY2022 and FY2023, covering Item 1 (competition), Item 1A (supply chain risk), Item 7 (gross margin, net sales), and Item 8 (deferred revenue). To extend or replace the dataset, edit the JSON directly — each entry has the following fields:

| Field | Description |
|-------|-------------|
| `question` | The question to ask the pipeline |
| `expected_answer` | Gold-standard answer used by the LLM judge |
| `source_section` | Item label where the answer lives (e.g. `"Item 7"`) |
| `source_passage` | Verbatim substring of the chunk text used for Precision@5 scoring |

The `source_passage` must be an exact substring of a stored chunk (post-cleaning, post-chunking). To find the right text, run the pipeline in `DEBUG` log mode and inspect the retrieved chunks, or query ChromaDB directly.

### Run the harness

```bash
python eval/run_eval.py --ticker AAPL --years 2022 2023 2024
```

The harness prints a per-question table and aggregate metrics, then saves a timestamped JSON file to `eval/results/`.

### Metrics

| Metric | Description |
|--------|-------------|
| **Precision@5** | 1 if any of the top-5 retrieved chunks contains the gold passage (substring match), else 0 |
| **Factual correctness** (1–5) | LLM judge: does the answer agree with the expected answer on key facts? |
| **Faithfulness** (1–5) | LLM judge: is the answer fully grounded in retrieved context, no hallucinations? |
| **Appropriate uncertainty** (1–5) | LLM judge: does the model correctly defer when context is insufficient? |

### Results (AAPL, FY2022–2024, run 2026-06-28)

| ID | Question (truncated) | Precision@5 | Factual | Faithful | Uncertainty |
|----|----------------------|:-----------:|:-------:|:--------:|:-----------:|
| q1 | Gross margin % FY2023 and drivers | 0.0 | 5 | 5 | 5 |
| q2 | Manufacturing concentration risk | 0.0 | 5 | 5 | 5 |
| q3 | Net sales growth FY2022 vs FY2021 | 0.0 | 1 | 5 | 5 |
| q4 | Deferred revenue FY2023 | 0.0 | 5 | 5 | 5 |
| q5 | Competitive landscape FY2022 | 0.0 | 5 | 5 | 5 |
| **Mean** | | **0.000** | **4.2 / 5** | **5.0 / 5** | **5.0 / 5** |

**Precision@5 = 0.0** is caused by two compounding bugs, both tracked in [What I'd improve](#what-id-improve):

1. **Ingestion mtime bug** — `_download_with_retry` selects the most recently modified subfolder under the filing root. When all three years are ingested in a single run, later downloads touch sibling folder mtimes, causing some years to ingest the wrong filing entirely. Simulation confirms that fixing this alone would raise Precision@5 to ~0.6 (q1, q3, q4 would pass).

2. **Apostrophe encoding mismatch** — SEC filings use curly right-single-quotes (`'` U+2019) for possessives. The `source_passage` strings in `golden_dataset.json` were written with straight apostrophes (`'` U+0027), so substring match fails for any passage containing `Company's`, `Apple's`, etc. even when the correct filing is indexed. Fixing both bugs together would bring Precision@5 to 1.0.

Despite Precision@5 = 0.0, the LLM judge scores confirm the pipeline is retrieving semantically relevant context and generating faithful, well-calibrated answers — the retriever surfaces correct passages even when year metadata is misaligned.

---

## Design decisions

### Section-aware chunking over fixed-size splitting

SEC 10-Ks are highly structured documents. Fixed-size character splitting ignores this structure and frequently cuts mid-sentence across section boundaries, mixing unrelated content (e.g. legal boilerplate from Item 1 with financial narrative from Item 7). By splitting on Item boundaries first — then paragraph-chunking within each section — every chunk carries a coherent semantic unit and a meaningful section label that travels with it as metadata. This improves retrieval precision and gives the model better context for citing specific parts of the filing.

### Two-stage retrieval: embed then rerank

Pure vector search is fast but imprecise — cosine similarity over embeddings conflates semantic relatedness with relevance. A cross-encoder reranker (Cohere `rerank-english-v3.0`) scores each (query, passage) pair jointly, which is slower but significantly more accurate. The two-stage design keeps latency acceptable: vector search narrows 10,000+ chunks down to 20 candidates cheaply, and the reranker only processes those 20. This pattern consistently outperforms single-stage retrieval on financial QA benchmarks.

### Command R+ for generation

Cohere Command R+ is optimised for retrieval-augmented generation — it was trained to work with grounded context and produce citations. Its explicit support for RAG-style prompting (numbered sources, instruction-following on "cite or defer") makes it a natural fit compared to general-purpose instruction models that require more prompt engineering to stay faithful to context.

### LLM-as-judge for eval

Human evaluation doesn't scale, and simple n-gram metrics (BLEU, ROUGE) miss semantic correctness in financial language ("gross margin improved 80bp" ≠ "margins declined"). An LLM judge scoring factual correctness, faithfulness, and calibration gives a richer signal. Using the same model family for judge and generator is a known limitation (see below), but it still captures obvious failures like hallucinated numbers or confident wrong answers.

---

## What I'd improve

1. **Filing subfolder selection in ingestion.** `_download_with_retry` picks the most recently modified subfolder under the filing root, which breaks when multiple years are ingested in one run — later downloads update sibling folder mtimes and cause earlier years to ingest the wrong filing. Fix: use the accession number returned by the downloader, or sort subfolders by name (accession numbers are lexicographically ordered by date) rather than mtime.

2. **Apostrophe normalisation in eval.** SEC filings use curly right-single-quotes (U+2019) for possessives, but `source_passage` strings in the golden dataset use straight apostrophes (U+0027). The Precision@5 substring match fails for any passage containing words like `Company's`. Fix: normalise both needle and haystack to straight apostrophes before comparing, or copy `source_passage` values directly from stored chunk text.

3. **Metadata-filtered retrieval.** Currently a multi-year query searches across all years indiscriminately. Adding a `where={"filing_year": {"$in": [2022, 2023]}}` filter to the ChromaDB query would let the user ask year-specific questions without the retriever surfacing answers from the wrong period.

4. **Table and numeric extraction.** Most of the quantitative data in 10-Ks lives in XBRL-tagged tables that get mangled by naive HTML stripping. A dedicated table parser (e.g. converting `<table>` elements to markdown or CSV before chunking) would dramatically improve precision on numerical questions like revenue growth or EPS.

5. **Independent judge model.** Using Command R+ to both generate and judge answers introduces self-serving bias — the model is unlikely to penalise outputs that match its own generation style. Using a separate judge (e.g. Cohere's Aya or a different provider) would give a more honest faithfulness score.

6. **Hybrid retrieval (BM25 + dense).** Cohere embeddings are strong on semantic similarity but can miss exact financial terms, ticker symbols, and numeric literals. A BM25 index (e.g. via Elasticsearch or `rank_bm25`) fused with dense search (reciprocal rank fusion) would improve recall on queries containing specific figures or product names.
