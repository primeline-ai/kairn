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


def to_fts_query(text: str) -> str | None:
    """Convert natural language to a safe FTS5 OR query.

    Returns a quoted OR-joined keyword string (always valid FTS5), or None
    when no searchable keyword survives stop-word/length filtering.
    """
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    keywords = [
        w for w in words if w not in _STOP_WORDS and w not in _FTS_RESERVED and len(w) > 2
    ]
    if not keywords:
        return None
    return " OR ".join(f'"{w}"' for w in keywords)


# Backward-compatible private alias. Historical callers (and the LongMemEval
# benchmark harness) import the underscore name from core.intelligence; that
# re-export now resolves here.
_to_fts_query = to_fts_query
