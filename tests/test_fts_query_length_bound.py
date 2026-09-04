"""A query costs what its VOCABULARY costs, not what the prompt's length costs.

`to_fts_query` joined every keyword OCCURRENCE, so a pasted document produced
an FTS5 query repeating the same handful of words thousands of times. Measured
on a 15,055-node / 11,954-experience store with a query whose vocabulary is
THREE words at every size:

    repeats   prompt chars   fts query chars      search
          1             24                35        7 ms
        100          2,400             3,896       92 ms
        500         12,000            19,496       1.7 s
      1,000         24,000            38,996       6.5 s
      2,000         48,000            77,996        26 s
      4,000         96,000                --   killed at 45 s

Roughly quadratic in prompt length.

WHO IS ACTUALLY EXPOSED, corrected by the control rather than assumed. NOT the
Evolving hooks: both cap their query at eight keywords before it reaches this
function (`extract_subject(max_terms=8)`, and the injector's own `[:8]`), and
a 600,000-character prompt through the live hook was 374 ms on the OLD engine.
The exposed callers are the ones that pass UNBOUNDED text, and the one that
matters runs on every save: `IntelligenceLayer` builds the `candidates[]` scan
from a note's FULL content, so saving a note costs what the note is long.
Measured against the 15k-node store, old engine against new, same six hits:

    note content   fts query    candidate scan OLD    NEW
       2,976 ch     4,460 ch             415 ms       9 ms
      11,616 ch    17,420 ch           4,762 ms       6 ms
      28,896 ch    43,340 ch          27,261 ms       6 ms

A 29,000-character note took twenty-seven seconds to save its candidate scan,
and returns the identical six candidates in six milliseconds now.

Duplicated OR terms cannot change an FTS5 result set, which is what makes this
free rather than a trade.
"""
import pytest

from kairn.core.fts import fts_keywords, to_fts_query


class TestLengthIsBoundedByVocabulary:
    def test_repeats_do_not_grow_the_query(self):
        assert to_fts_query("werkzeug pairing ledger") == to_fts_query(
            "werkzeug pairing ledger " * 1000
        )

    def test_first_seen_order_survives(self):
        assert to_fts_query("ledger werkzeug ledger pairing") == (
            '"ledger" OR "werkzeug" OR "pairing"'
        )

    def test_a_huge_prompt_yields_a_small_query(self):
        q = to_fts_query("werkzeug pairing ledger " * 20000)
        assert q is not None and len(q) < 60, len(q)

    def test_fts_keywords_still_reports_every_occurrence(self):
        """The dedup belongs to the QUERY builder only. Callers that count
        occurrences must keep seeing them."""
        assert len(fts_keywords("werkzeug werkzeug werkzeug")) == 3

    def test_an_unsearchable_query_is_still_none(self):
        assert to_fts_query("the and of it") is None
        assert to_fts_query("") is None


class TestTheSameRowsComeBack:
    @pytest.mark.asyncio
    async def test_a_repeated_query_returns_what_the_short_one_does(self, store):
        """The control that makes the dedup safe rather than merely fast."""
        from kairn.core.experience import ExperienceEngine
        from kairn.events.bus import EventBus

        eng = ExperienceEngine(store, EventBus())
        await eng.save(content="werkzeug pairing ledger union", type="gotcha")
        await eng.save(content="werkzeug alone here", type="gotcha")
        short = await eng.search(text="werkzeug pairing ledger", limit=10)
        long_ = await eng.search(text="werkzeug pairing ledger " * 200, limit=10)
        assert [e.id for e in short] == [e.id for e in long_]
