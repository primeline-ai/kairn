"""Non-ASCII input, which nothing in this suite covered before.

Two defects lived here undisturbed through a passing 661-test suite, because
every existing test uses ASCII text. Both were found by chance, and both are
about the same thing: CODE THAT SCORES AN FTS5 MATCH MUST MODEL FTS5's
TOKENIZER, NOT PYTHON's STRING OPERATORS.

  1. `fts_keywords` split words at every accented letter, so a German query was
     cut to a fragment and never met the correctly-indexed token.
  2. `_term_coverage` compared query terms against raw text with `in`, which
     disagrees with `unicode61` in BOTH directions - it misses a diacritic-folded
     match, and it counts a substring FTS5 never matched.
"""

from __future__ import annotations

import sqlite3

import pytest

from kairn.core.fts import fts_keywords, to_fts_query
from kairn.core.intelligence import _fold_diacritics, _term_coverage


@pytest.fixture
def fts():
    """A real FTS5 table with this project's schema, so the tests compare
    against the actual tokenizer rather than an assumption about it."""
    c = sqlite3.connect(":memory:")
    c.execute("CREATE VIRTUAL TABLE nodes_fts USING fts5(name, description)")
    for row in [
        ("Änderung am Schwellenwert", "die Änderung wurde geprüft"),
        ("Büro Nord", "Zürich delegation workflow"),
        ("Zurich ascii office", "plain delegation notes"),
        ("Harpsichord notes", "Harpsichord repair category notes"),
    ]:
        c.execute("INSERT INTO nodes_fts VALUES (?,?)", row)
    return c


def hits(c, query):
    return c.execute(
        "SELECT count(*) FROM nodes_fts WHERE nodes_fts MATCH ?", (query,)
    ).fetchone()[0]


class TestTokenizerKeepsNonAsciiWordsWhole:
    def test_german_words_are_not_split_at_the_umlaut(self):
        assert fts_keywords("Ümlaut") == ["ümlaut"]
        assert fts_keywords("Änderung") == ["änderung"]
        assert fts_keywords("Prüfung") == ["prüfung"]
        assert fts_keywords("größer") == ["größer"]

    def test_a_german_query_actually_retrieves_german_content(self, fts):
        """The end-to-end failure. Before the fix this returned 0: the index
        held `änderung` and the query asked for `nderung`."""
        q = to_fts_query("Änderung")
        assert q == '"änderung"'
        assert hits(fts, q) == 1

    def test_ascii_queries_are_byte_identical(self):
        for text, want in [
            ("wal checkpoint starvation", '"wal" OR "checkpoint" OR "starvation"'),
            ("delegation enforcer threshold", '"delegation" OR "enforcer" OR "threshold"'),
            ("a-b c_d", '"c_d"'),
        ]:
            assert to_fts_query(text) == want
        assert to_fts_query("AND OR NOT") is None


class TestCoverageAgreesWithTheTokenizer:
    def test_ascii_term_covers_accented_text(self, fts):
        """unicode61 folds diacritics, so `zurich` really does MATCH `Zürich`.
        Coverage has to agree, or a real match is scored to zero and dropped.

        NOTE the fields: neither carries an ASCII `zurich`. The first version of
        this test used a name of "Zurich office", which supplied the token by
        itself - so a mutant that folded only ONE side, or neither, still
        passed. The accented text must be the only possible source."""
        assert hits(fts, '"zurich"') >= 1, "fixture broken: FTS did not match"
        assert _term_coverage(["zurich"], "Büro Nord", "Zürich delegation workflow") == 1.0

    def test_accented_term_covers_ascii_text(self):
        """The other direction, which nothing tested: an accented QUERY term
        against plain text. Folding the haystack alone is not enough."""
        assert _term_coverage(["zürich"], "Zurich ascii office", "plain delegation notes") == 1.0

    def test_a_substring_is_not_coverage(self, fts):
        """`cat` matches nothing in FTS5, so it must not count as covered just
        because `category` contains those letters."""
        assert hits(fts, '"cat"') == 0, "fixture broken: FTS matched 'cat'"
        assert _term_coverage(
            ["harpsichord", "cat"], "Harpsichord notes", "Harpsichord repair category notes"
        ) == 0.5

    def test_fold_is_the_same_folding_the_index_uses(self, fts):
        assert _fold_diacritics("Zürich".lower()) == "zurich"
        assert hits(fts, '"zurich"') == hits(fts, '"Zürich"')

    def test_boundaries(self):
        assert _term_coverage([], "anything", None) == 1.0     # no terms, no penalty
        assert _term_coverage(["wal"], None, None) == 0.0      # no text to cover
        assert _term_coverage(["wal", "zebra"], "wal checkpoint", None) == 0.5
