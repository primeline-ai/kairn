"""Coverage-weighted node relevance (fix/recall-coverage-weighted-relevance).

`to_fts_query` joins terms with OR so ANY keyword can match - deliberate, and
the source of Kairn's recall. But bm25 then scores whatever matched and
`_bm25_to_relevance` only sees that aggregate, so a 1-of-6 match scored like a
6-of-6 one. Measured on the 12,866-node store 2026-08-22: the query "baroque
harpsichord tuning temperament werckmeister" returned hits at 0.60 by matching
only "tuning"; genuinely relevant queries scored 0.67-0.73. An 0.08 separation
band makes `min_relevance` decorative.

These tests pin the fix. Each one FAILS on the pre-fix code.
"""

from __future__ import annotations

from kairn.core.intelligence import (
    _bm25_to_relevance,
    _fts_terms,
    _term_coverage,
)


class TestFtsTerms:
    def test_recovers_terms_from_an_or_query(self):
        assert _fts_terms('"alpha" OR "beta" OR "gamma"') == ["alpha", "beta", "gamma"]

    def test_browse_query_has_no_terms(self):
        assert _fts_terms(None) == []
        assert _fts_terms("") == []


class TestTermCoverage:
    def test_full_coverage_is_one(self):
        assert _term_coverage(["alpha", "beta"], "alpha beta gamma") == 1.0

    def test_partial_coverage_is_the_fraction(self):
        # 1 of 5 terms present -> 0.2, the case that used to score like a full match
        assert _term_coverage(
            ["harpsichord", "werckmeister", "temperament", "baroque", "tuning"],
            "autoevolve self-tuning plan",
        ) == 0.2

    def test_no_overlap_is_zero(self):
        assert _term_coverage(["harpsichord"], "delegation config") == 0.0

    def test_is_case_insensitive_and_dedupes_terms(self):
        assert _term_coverage(["Alpha", "ALPHA", "beta"], "alpha beta") == 1.0

    def test_no_terms_means_no_signal_not_zero(self):
        # A browse query must not be scored down to nothing.
        assert _term_coverage([], "anything") == 1.0

    def test_empty_fields_cannot_cover_anything(self):
        assert _term_coverage(["alpha"], None, "") == 0.0


class TestSeparation:
    """The property that actually matters: a weak match must be separable from
    a strong one by a threshold. This is what failed before the fix."""

    def test_weak_match_falls_below_a_floor_a_strong_match_clears(self):
        floor = 0.35
        # Same bm25 strength for both - only coverage differs.
        base = _bm25_to_relevance(-7.5)  # ~0.6, the measured alien-query score
        weak = base * _term_coverage(["a", "b", "c", "d", "e"], "a only")
        strong = base * _term_coverage(["a", "b", "c", "d", "e"], "a b c d e")
        assert weak < floor < strong, f"weak={weak} strong={strong}"

    def test_coverage_never_raises_relevance(self):
        base = _bm25_to_relevance(-13.5)
        assert base * _term_coverage(["a"], "a") == base
        assert base * _term_coverage(["a", "b"], "a") < base


# --- Wiring test -------------------------------------------------------
# The unit tests above cover the helpers. They would ALL still pass if the
# coverage multiplication were deleted from `_keyword_node_recall`, so this
# exercises the real path end to end. Verified by mutation: removing that
# multiplication makes this test fail and leaves the ten above green.

import pytest

from kairn.core.experience import ExperienceEngine
from kairn.core.graph import GraphEngine
from kairn.core.ideas import IdeaEngine
from kairn.core.intelligence import IntelligenceLayer
from kairn.core.memory import ProjectMemory
from kairn.core.router import ContextRouter
from kairn.events import EventBus
from kairn.storage.sqlite_store import SQLiteStore


async def _intel(tmp_path):
    store = SQLiteStore(tmp_path / "cov.db")
    await store.initialize()
    bus = EventBus()
    graph = GraphEngine(store, bus)
    return IntelligenceLayer(
        store=store,
        event_bus=bus,
        graph=graph,
        router=ContextRouter(store, bus),
        memory=ProjectMemory(store, bus),
        experience=ExperienceEngine(store, bus),
        ideas=IdeaEngine(store, bus),
    ), graph


@pytest.mark.asyncio
async def test_keyword_node_recall_applies_coverage(tmp_path):
    """The fixture is the load-bearing half here.

    A one-node store is NOT discriminating: bm25 alone already gates the node
    out, so the test passes with or without the fix. Measured on this fixture
    instead - one term repeated so bm25 is strong, plus filler so idf is not
    degenerate - the node scores bm25rel=0.5374, which CLEARS a 0.35 floor.
    Only coverage (0.20) pulls it to 0.1075 and gates it. So deleting the
    multiplication in `_keyword_node_recall` makes this test fail, which is
    the whole point of it.
    """
    intel, graph = await _intel(tmp_path)
    await graph.add_node(
        name="Tuning note",
        type="pattern",
        description=("tuning " * 40) + "threshold plan",
    )
    for i in range(30):
        await graph.add_node(
            name=f"Filler {i}", type="pattern",
            description="unrelated corpus filler document",
        )
    fts = '"baroque" OR "harpsichord" OR "tuning" OR "temperament" OR "werckmeister"'

    gated = await intel._keyword_node_recall(
        fts_query=fts, limit=5, min_relevance=0.35
    )
    assert gated == [], f"weak 1-of-5 match survived a 0.35 floor: {gated}"

    ungated = await intel._keyword_node_recall(
        fts_query=fts, limit=5, min_relevance=0.0
    )
    assert len(ungated) == 1, "coverage must not change RECALL, only the score"
    assert ungated[0]["relevance"] < 0.35, ungated[0]["relevance"]
