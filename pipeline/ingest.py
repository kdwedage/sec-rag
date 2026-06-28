"""
SEC EDGAR ingestion: download 10-K filings, strip boilerplate, and chunk by section.
"""

import re
import time
import logging
from pathlib import Path
from typing import Optional

from sec_edgar_downloader import Downloader

from pipeline.config import (
    TARGET_CHUNK_CHARS,
    CHUNK_OVERLAP_CHARS,
    TARGET_SECTIONS,
    SEC_RATE_LIMIT_BACKOFF_BASE,
    SEC_RATE_LIMIT_MAX_RETRIES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section boundary patterns
# ---------------------------------------------------------------------------

# Matches "Item 1", "Item 1A", "Item 7", "Item 7A", "Item 8" etc.
# Tolerates extra whitespace, all-caps variants, and optional periods.
_SECTION_PATTERN = re.compile(
    r"(?:^|\n)\s*"
    r"(Item\s+\d+[A-Z]?\.?)"
    r"[\.\s—–\-:]+",
    re.IGNORECASE | re.MULTILINE,
)

# HTML/XBRL noise to strip
_HTML_TAG = re.compile(r"<[^>]+>", re.DOTALL)
_XML_DECL = re.compile(r"<\?xml[^>]+\?>", re.DOTALL)
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _strip_boilerplate(raw: str) -> str:
    """Remove HTML tags, XBRL markup, and excess whitespace from a raw filing."""
    text = _XML_DECL.sub("", raw)
    text = _HTML_TAG.sub(" ", text)
    # Decode common HTML entities
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
        .replace("&#160;", " ")
        .replace("&apos;", "'")
        .replace("&quot;", '"')
    )
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

def _split_into_sections(text: str) -> dict[str, str]:
    """
    Split cleaned filing text into a dict of {section_label: section_text}.

    Only sections listed in TARGET_SECTIONS are kept; all other content is
    grouped under the key 'Other'.
    """
    matches = list(_SECTION_PATTERN.finditer(text))
    if not matches:
        return {"Other": text}

    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        label_raw = match.group(1).strip().rstrip(".")
        # Normalise: "ITEM  1A" → "Item 1A"
        label = re.sub(r"\s+", " ", label_raw).title()
        # Replace "Item  1A" → "Item 1A" (extra space between word and number)
        label = re.sub(r"Item\s+", "Item ", label)

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        # Keep first occurrence of each section (filings repeat TOC entries)
        if label not in sections:
            sections[label] = body

    # Filter to relevant sections
    filtered: dict[str, str] = {}
    for label, body in sections.items():
        for target in TARGET_SECTIONS:
            if label.lower() == target.lower():
                filtered[label] = body
                break

    return filtered if filtered else {"Other": text}


# ---------------------------------------------------------------------------
# Paragraph-level chunking within a section
# ---------------------------------------------------------------------------

def _chunk_section(
    text: str,
    ticker: str,
    filing_year: int,
    section: str,
    start_index: int,
) -> list[dict]:
    """
    Split a single section into overlapping chunks targeting TARGET_CHUNK_CHARS.

    Splits first on double-newlines (paragraph boundaries), then concatenates
    paragraphs until the target size is reached, with CHUNK_OVERLAP_CHARS of
    trailing text carried forward.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    chunks: list[dict] = []
    current = ""
    chunk_idx = start_index

    for para in paragraphs:
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= TARGET_CHUNK_CHARS:
            current = candidate
        else:
            # Flush what we have
            if current:
                chunks.append(
                    {
                        "ticker": ticker,
                        "filing_year": filing_year,
                        "section": section,
                        "chunk_index": chunk_idx,
                        "text": current,
                    }
                )
                chunk_idx += 1
                # Carry overlap forward
                current = current[-CHUNK_OVERLAP_CHARS:] + "\n\n" + para
                current = current.strip()
            else:
                # Single paragraph exceeds target — split by sentence
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sent in sentences:
                    candidate = (current + " " + sent).strip() if current else sent
                    if len(candidate) <= TARGET_CHUNK_CHARS:
                        current = candidate
                    else:
                        if current:
                            chunks.append(
                                {
                                    "ticker": ticker,
                                    "filing_year": filing_year,
                                    "section": section,
                                    "chunk_index": chunk_idx,
                                    "text": current,
                                }
                            )
                            chunk_idx += 1
                            current = current[-CHUNK_OVERLAP_CHARS:] + " " + sent
                            current = current.strip()
                        else:
                            # Single sentence is enormous — hard split
                            for i in range(0, len(sent), TARGET_CHUNK_CHARS - CHUNK_OVERLAP_CHARS):
                                chunk_text = sent[i : i + TARGET_CHUNK_CHARS]
                                chunks.append(
                                    {
                                        "ticker": ticker,
                                        "filing_year": filing_year,
                                        "section": section,
                                        "chunk_index": chunk_idx,
                                        "text": chunk_text,
                                    }
                                )
                                chunk_idx += 1
                            current = ""

    if current:
        chunks.append(
            {
                "ticker": ticker,
                "filing_year": filing_year,
                "section": section,
                "chunk_index": chunk_idx,
                "text": current,
            }
        )

    return chunks


# ---------------------------------------------------------------------------
# Filing discovery
# ---------------------------------------------------------------------------

def _find_filing_text(filing_dir: Path) -> Optional[str]:
    """
    Walk the filing directory and return the text of the best candidate file.

    sec-edgar-downloader saves filings as .txt (full submission) or .htm/.html.
    We prefer the .txt submission file; fall back to the largest .htm file.
    """
    txt_files = list(filing_dir.rglob("*.txt"))
    htm_files = list(filing_dir.rglob("*.htm")) + list(filing_dir.rglob("*.html"))

    # The primary submission file is typically named like '0000012345-23-000001.txt'
    # or simply 'filing-details.txt'. Pick the largest .txt file.
    if txt_files:
        return max(txt_files, key=lambda p: p.stat().st_size).read_text(
            encoding="utf-8", errors="replace"
        )
    if htm_files:
        return max(htm_files, key=lambda p: p.stat().st_size).read_text(
            encoding="utf-8", errors="replace"
        )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest(ticker: str, years: list[int], data_dir: str = "sec_data") -> list[dict]:
    """
    Download 10-K filings for *ticker* for each year in *years*, parse and
    chunk them, and return a flat list of chunk dicts.

    Each chunk dict has keys:
        ticker, filing_year, section, chunk_index, text

    Retries with exponential back-off on SEC EDGAR rate-limit errors.
    """
    dl = Downloader(company_name="SecRagPipeline", email_address="user@example.com", save_path=data_dir)
    all_chunks: list[dict] = []

    for year in years:
        logger.info("Downloading 10-K for %s (%d)…", ticker, year)
        raw_text = _download_with_retry(dl, ticker, year)
        if raw_text is None:
            logger.warning("No filing found for %s %d — skipping.", ticker, year)
            continue

        clean = _strip_boilerplate(raw_text)
        sections = _split_into_sections(clean)
        logger.info("  Found sections: %s", list(sections.keys()))

        chunk_counter = 0
        for section_label, section_text in sections.items():
            section_chunks = _chunk_section(
                section_text, ticker, year, section_label, chunk_counter
            )
            chunk_counter += len(section_chunks)
            all_chunks.extend(section_chunks)

        logger.info("  Total chunks so far: %d", len(all_chunks))

    return all_chunks


def _download_with_retry(dl: Downloader, ticker: str, year: int) -> Optional[str]:
    """
    Attempt to download the 10-K for *ticker* filed in *year*, retrying with
    exponential back-off if the SEC EDGAR API rate-limits us.
    """
    delay = SEC_RATE_LIMIT_BACKOFF_BASE

    for attempt in range(1, SEC_RATE_LIMIT_MAX_RETRIES + 1):
        try:
            # after_date / before_date narrow to filings *for* a given fiscal year
            dl.get(
                "10-K",
                ticker,
                limit=1,
                after=f"{year - 1}-06-01",
                before=f"{year + 1}-01-01",
            )

            # Locate the downloaded files on disk
            filing_root = Path(dl.save_path) / "sec-edgar-filings" / ticker / "10-K"
            if not filing_root.exists():
                logger.warning("Filing root not found: %s", filing_root)
                return None

            # Find the most recently created subfolder (= newest download)
            subdirs = sorted(filing_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if not subdirs:
                return None

            raw = _find_filing_text(subdirs[0])
            return raw

        except Exception as exc:
            err_str = str(exc).lower()
            is_rate_limit = any(
                kw in err_str for kw in ("429", "rate limit", "too many requests")
            )
            if is_rate_limit and attempt < SEC_RATE_LIMIT_MAX_RETRIES:
                logger.warning(
                    "Rate-limited by SEC EDGAR (attempt %d/%d). Retrying in %ds…",
                    attempt,
                    SEC_RATE_LIMIT_MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)
                delay *= 2
            else:
                logger.error("Failed to download %s %d: %s", ticker, year, exc)
                return None

    return None
