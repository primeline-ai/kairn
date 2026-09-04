"""The experience sort key is a COMPENSATORY score, not a decay bucket.

WHY THIS FILE EXISTS. `search` reported a match-aware `recall_relevance` but
still ORDERED by `relevance_bucket(decay)`, so the number a caller reads and
the order it reads it in disagreed: a fresh experience that barely matched
outranked an older one that matched well, and the reported score said the
opposite. That is the lexicographic, non-compensatory sort Kairn `cec86cb9`
names as the root cause of "the injector keeps showing me recent notes I do
not need", and the store's own rule from that node is explicit - compensatory
ranking, never a re-tuned gate.

The blend is `match * (LO + (HI - LO) * decay)` with match itself
`bm25_to_relevance(rank) * term_coverage(terms, ...)`. Recency stays a
bounded NUDGE (a +-10% multiplier by default), never the primary key, so a
strong old match beats a weak fresh one and two equal matches still order by
age.

`min_relevance` deliberately keeps its decay meaning - gating it on the
composite was tried on an earlier branch and reverted, because bm25 magnitude
scales with corpus size and any positive floor then filters a small store
empty.
"""

from datetime import UTC, datetime, timedelta

import pytest

from kairn.core.experience import ExperienceEngine
from kairn.core.fts import blend_match_and_recency, term_coverage
from kairn.events.bus import EventBus


@pytest.fixture
async def engine(store):
    return ExperienceEngine(store, EventBus())


async def _age(engine, exp, days):
    past = datetime.now(UTC) - timedelta(days=days)
    await engine.store.db.execute(
        "UPDATE experiences SET created_at = ? WHERE id = ?",
        (past.isoformat(), exp.id),
    )
    await engine.store.db.commit()


class TestTermCoverage:
    def test_full_coverage_is_one_and_no_terms_is_one(self):
        assert term_coverage(["rust", "borrow"], "rust borrow checker") == 1.0
        assert term_coverage([], "anything") == 1.0

    def test_partial_coverage_scales_down(self):
        assert term_coverage(["rust", "borrow", "lifetime"], "rust only") == pytest.approx(1 / 3)

    def test_no_haystack_is_zero(self):
        assert term_coverage(["rust"], None) == 0.0


class TestBlend:
    def test_recency_is_a_bounded_nudge_not_the_key(self):
        """A 10x match difference must not be reversible by age."""
        strong_old = blend_match_and_recency(match=0.80, decay=0.0)
        weak_fresh = blend_match_and_recency(match=0.08, decay=1.0)
        assert strong_old > weak_fresh

    def test_equal_matches_order_by_recency(self):
        assert blend_match_and_recency(match=0.5, decay=1.0) > blend_match_and_recency(
            match=0.5, decay=0.0
        )

    def test_the_nudge_band_is_bounded(self):
        """No decay value may move a score by more than the declared band."""
        hi = blend_match_and_recency(match=1.0, decay=1.0)
        lo = blend_match_and_recency(match=1.0, decay=0.0)
        assert hi / lo <= 1.25


class TestSearchOrder:
    @pytest.mark.asyncio
    async def test_a_strong_old_match_outranks_a_weak_fresh_one(self, engine):
        """THE DEFECT THIS FILE EXISTS FOR, driven through the real search."""
        strong = await engine.save(
            content="Rust borrow checker lifetimes explained for closures",
            type="gotcha",
        )
        await engine.save(content="A note about lifetimes in a lease contract", type="gotcha")
        await _age(engine, strong, 45)

        results = await engine.search(text="rust borrow checker lifetimes", limit=5)
        assert len(results) >= 2, "sanity: both experiences must be retrieved"
        assert results[0].id == strong.id, [
            (r.content[:40], r.recall_relevance) for r in results
        ]

    @pytest.mark.asyncio
    async def test_the_reported_score_agrees_with_the_order(self, engine):
        """A score a caller reads must not contradict the order it arrives in -
        the same defect one level up, and the reason this is asserted."""
        await engine.save(content="Rust borrow checker lifetimes", type="gotcha")
        await engine.save(content="Rust ownership and moves", type="gotcha")
        await engine.save(content="Gardening notes about rust on a fence", type="gotcha")
        results = await engine.search(text="rust borrow checker", limit=5)
        scores = [r.recall_relevance for r in results if r.recall_relevance is not None]
        assert scores == sorted(scores, reverse=True), scores

    @pytest.mark.asyncio
    async def test_a_browse_query_still_orders_by_recency(self, engine):
        """The positive control for the bound: with no text there is no match
        strength, so the old behaviour must survive untouched."""
        old = await engine.save(content="an older note", type="gotcha")
        await engine.save(content="a newer note", type="gotcha")
        await _age(engine, old, 60)
        results = await engine.search(limit=5)
        assert results[0].id != old.id


class TestAllStopwordQuery:
    """A question made only of stop words has no searchable term, and the
    honest answer to it is NOTHING - not the most recent notes.

    Today such a query collapses to `fts_text = None`, which is the same
    internal state as "no text was given at all", so it silently becomes a
    BROWSE and returns the freshest experiences dressed as an answer. That is
    the plausible-wrong-answer shape this whole arc exists to remove: an
    UNKNOWN reported as a result (Kairn `8a3a11af`).

    Deliberately NOT fixed by adding a German stop-word list - Kairn
    `27165d4c` decided against one, and "was ist das" keeps its terms.
    """

    @pytest.mark.asyncio
    async def test_an_all_stopword_query_returns_nothing(self, engine):
        await engine.save(content="Rust borrow checker lifetimes", type="gotcha")
        assert await engine.search(text="the and of it") == []
        assert await engine.search(text="the") == []

    @pytest.mark.asyncio
    async def test_a_browse_query_is_not_affected(self, engine):
        """The positive control: no text at all still browses."""
        await engine.save(content="Rust borrow checker lifetimes", type="gotcha")
        assert await engine.search(limit=5), "a browse query must still return rows"

    @pytest.mark.asyncio
    async def test_a_german_prompt_keeps_its_terms(self, engine):
        """No German stop-word list (Kairn `27165d4c`), so this is a real
        query and must not be swallowed by the guard above."""
        await engine.save(content="Das ist ein Werkzeug fuer die Paarung", type="gotcha")
        assert await engine.search(text="was ist das") != []

    @pytest.mark.asyncio
    async def test_bitemporal_has_the_same_guard(self, engine):
        """Both gates or neither (Kairn `d5272a91`)."""
        await engine.save(content="Rust borrow checker lifetimes", type="gotcha")
        assert await engine.search_bitemporal(text="the and of it") == []


class TestTheSortIsNotQuantised:
    """The two mutants that SURVIVED the first pass, now covered.

    Reverting the sort to `relevance_bucket(...)` and dropping
    `term_coverage` from the match both left every earlier test green, which
    means those tests were asserting the ORDER I wanted rather than the
    MECHANISM that produces it (Kairn `ee2d1a9f`: a check that cannot fail is
    not a check). These two drive the mechanism directly.
    """

    @pytest.mark.asyncio
    async def test_a_sub_bucket_difference_still_orders(self, engine):
        """Quantising the sort key throws away every difference smaller than
        one bucket and falls back to insertion order. Two experiences whose
        blends differ by less than a bucket must still come back in blend
        order, which no bucketed sort can do."""
        from kairn.core.experience import RELEVANCE_BUCKET_SIZE

        weaker = await engine.save(content="alpha beta gamma delta filler one", type="gotcha")
        stronger = await engine.save(content="alpha beta gamma delta filler two", type="gotcha")
        await _age(engine, stronger, 1)          # a tiny, sub-bucket difference
        results = await engine.search(text="alpha beta gamma delta", limit=5)
        assert len(results) == 2
        blends = [r.recall_relevance for r in results]
        assert abs(blends[0] - blends[1]) < RELEVANCE_BUCKET_SIZE, (
            "fixture no longer produces a SUB-BUCKET difference: %s" % blends
        )
        assert blends[0] >= blends[1]
        assert results[0].id == weaker.id, [(r.content[:30], r.recall_relevance) for r in results]

    @pytest.mark.asyncio
    async def test_coverage_beats_a_repeated_single_term(self, engine):
        """The defect term_coverage exists for: bm25 cannot tell a 1-of-4
        match from a 4-of-4 one, so a document repeating ONE query word
        outranks a document that answers the whole question. With coverage in
        the match, the full answer wins."""
        narrow = await engine.save(
            content="werkzeug werkzeug werkzeug werkzeug werkzeug werkzeug",
            type="gotcha",
        )
        full = await engine.save(
            content="werkzeug paarung ledger union across every worktree",
            type="gotcha",
        )
        results = await engine.search(text="werkzeug paarung ledger union", limit=5)
        ids = [r.id for r in results]
        assert narrow.id in ids and full.id in ids, "sanity: both must be retrieved"
        assert ids[0] == full.id, [(r.content[:40], r.recall_relevance) for r in results]

