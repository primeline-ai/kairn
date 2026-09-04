"""FTS5 query shaping shared across the intelligence and experience layers.

Natural-language queries cannot be handed to SQLite FTS5 `MATCH` verbatim:
bare hyphens, colons and reserved words (`AND`/`OR`/`NOT`/`NEAR`) are parsed
as query operators and raise `sqlite3.OperationalError` (e.g. the query
"self-healing" is read as a column filter and fails with "no such column:
healing"). `to_fts_query` lowercases, tokenizes to safe alphanumerics, drops
stop-words and reserved tokens, then quotes each surviving term and joins with
OR so any keyword can match. The result is always a valid FTS5 string literal
sequence, or None when nothing searchable remains.

This module has no internal kairn dependencies so both `core.intelligence`
(which imports `core.experience`) and `core.experience` can import it without
creating an import cycle.
"""

from __future__ import annotations

import re

_STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "shall",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "about",
    "and",
    "or",
    "but",
    "not",
    "no",
    "so",
    "yet",
    "i",
    "me",
    "we",
    "us",
    "you",
    "he",
    "she",
    "it",
    "they",
    "them",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "this",
    "that",
    "these",
    "those",
    "need",
    "want",
    "try",
}

_FTS_RESERVED = {"and", "or", "not", "near"}


def fts_keywords(text: str) -> list[str]:
    """Tokenize natural language into the searchable keyword list.

    The single shared tokenizer behind `to_fts_query`: lowercased word
    tokens minus stop-words, FTS-reserved words, and short tokens. Exposed
    so callers that need the raw keyword set (e.g. the diversification
    pass's on-topic gate) share the exact same filtering instead of
    re-deriving it from the quoted query string.
    """
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    return [
        w for w in words if w not in _STOP_WORDS and w not in _FTS_RESERVED and len(w) > 2
    ]


def to_fts_query(text: str) -> str | None:
    """Convert natural language to a safe FTS5 OR query.

    Returns a quoted OR-joined keyword string (always valid FTS5), or None
    when no searchable keyword survives stop-word/length filtering.
    """
    keywords = fts_keywords(text)
    if not keywords:
        return None
    # DEDUPED, first-seen order. `fts_keywords` returns every OCCURRENCE, so a
    # pasted document used to produce an OR-query that repeated the same
    # handful of words thousands of times. Duplicated OR terms cannot change
    # an FTS5 result set, but they cost quadratically: measured on a
    # 15k-node / 12k-experience store with a THREE-term vocabulary, 1 repeat
    # searched in 7 ms, 500 in 1.7 s, 2,000 in 26 s and 4,000 did not finish
    # inside 45 s. The first-move hook caps its store call at 1.5 s, so the
    # prompts that carry the most content are exactly the ones whose block
    # silently disappeared. A query now costs what its VOCABULARY costs.
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(keywords))


# Backward-compatible private alias. Historical callers (and the LongMemEval
# benchmark harness) import the underscore name from core.intelligence; that
# re-export now resolves here.
_to_fts_query = to_fts_query


# bm25 score at which relevance = 0.5. Larger => the same bm25 match maps to a
# lower relevance, so weak keyword overlaps fall under a strict min_relevance
# floor while strong multi-term matches clear it.
BM25_RELEVANCE_MIDPOINT = 5.0


def bm25_to_relevance(rank: float | None) -> float:
    """Map an FTS5 bm25 `rank` to a bounded (0, 1] relevance.

    SQLite FTS5 exposes bm25 as a negative score where a more-negative value
    means a stronger match. A saturating transform (score / (score + K))
    preserves the raw bm25 ordering while yielding an absolute-ish relevance a
    min_relevance gate can act on. `rank is None` (a browse query with no
    MATCH) has no match strength to report, so it stays 1.0.

    Lives here rather than in `intelligence` because BOTH the node path and the
    experience path need it, and `intelligence` imports `experience`.
    """
    if rank is None:
        return 1.0
    score = max(0.0, -float(rank))
    return round(score / (score + BM25_RELEVANCE_MIDPOINT), 4)