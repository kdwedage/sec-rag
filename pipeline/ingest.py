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

# Matches actual section headers in the filing body (not TOC entries).
#
# After stripping HTML, the filing body has headers like:
#   \n Item 7.    Management's Discussion and Analysis ... \n
# while TOC entries look like:
#   \n Item 7. \n\n Management's Discussion...
#
# The discriminator: the title appears on the SAME line as the item label in
# the body. Requiring at least one non-space character after the label+spaces
# skips the TOC-only lines.
_SECTION_PATTERN = re.compile(
    r"\n[ \t]*(Item[ \t]+\d+[A-Z]?\.)[ \t]+([A-Z’‘][^\n]+)\n",
    re.IGNORECASE,
)

# HTML/XBRL noise to strip
_BLOCK_TAG = re.compile(
    r"<(?:p|div|br|h[1-6]|li|tr|td|th|section|article|header|footer|table)[^>]*>",
    re.IGNORECASE,
)
_HTML_TAG = re.compile(r"<[^>]+>", re.DOTALL)
_XML_DECL = re.compile(r"<\?xml[^>]+\?>", re.DOTALL)
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")

# SGML envelope patterns for the full-submission.txt format
_SGML_DOCUMENT = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL | re.IGNORECASE)
_SGML_TYPE = re.compile(r"<TYPE>\s*10-K\b", re.IGNORECASE)
_SGML_TEXT = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _extract_primary_document(raw: str) -> str:
    """
    Pull the primary 10-K HTML/text body from an SEC full-submission.txt file.

    Full submissions are SGML envelopes containing multiple <DOCUMENT> blocks.
    The first <TYPE>10-K block is the filing body; everything else is exhibits.
    Falls back to the full raw text if the SGML structure is absent.
    """
    for doc_match in _SGML_DOCUMENT.finditer(raw):
        doc_body = doc_match.group(1)
        if _SGML_TYPE.search(doc_body[:500]):
            text_match = _SGML_TEXT.search(doc_body)
            if text_match:
                return text_match.group(1)
    return raw  # plain .htm / .txt filing without SGML wrapper


def _strip_boilerplate(raw: str) -> str:
    """
    Extract the primary 10-K body, strip HTML/XBRL markup, and normalise whitespace.

    Block-level elements are replaced with newlines (not spaces) so that section
    headers that live inside <p>/<div> tags survive as newline-delimited lines.
    """
    text = _extract_primary_document(raw)
    text = _XML_DECL.sub("", text)
    # Block-level tags → newline to preserve text structure
    text = _BLOCK_TAG.sub("\n", text)
    # Remaining inline tags → space
    text = _HTML_TAG.sub(" ", text)
    # Decode common HTML entities (numeric and named)
    text = re.sub(r"&#\d+;", lambda m: _decode_numeric_entity(m.group()), text)
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
        .replace("&apos;", "'")
        .replace("&quot;", '"')
    )
    # Normalize non-breaking spaces to regular spaces so section headers
    # (which use \xa0 as padding between item number and title) can be matched.
    text = text.replace("\xa0", " ")
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def _decode_numeric_entity(entity: str) -> str:
    """Convert &#NNN; HTML numeric entities to their Unicode characters."""
    try:
        n = int(entity[2:-1])
        return chr(n)
    except (ValueError, OverflowError):
        return " "


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
        # group(1) = "Item 7." — strip trailing period and normalise spacing
        label_raw = match.group(1).strip().rstrip(".")
        label = re.sub(r"\s+", " ", label_raw).title()

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        # Keep the longest occurrence — body sections are much larger than TOC stubs
        existing = sections.get(label, "")
        if len(body) > len(existing):
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
    dl = Downloader(company_name="SecRagPipeline", email_address="user@example.com", download_folder=data_dir)
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
            filing_root = Path(dl.download_folder) / "sec-edgar-filings" / ticker / "10-K"
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
