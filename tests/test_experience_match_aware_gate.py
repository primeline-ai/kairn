"""Match-aware experience relevance (the experience half of the bm25 fix).

`relevance()` is pure time-decay by design - decay and match strength are
deliberately orthogonal in the MODEL. But recall was REPORTING that decay
value as the relevance of a search hit, so a fresh experience scored ~0.98
against any query that retrieved it at all, and alien queries came back
carrying ten confident-looking results. Same fake-relevance defect the node
path had before bm25 was surfaced (Kairn 86c00221).

These tests pin three things:
  1. the REPORTED relevance is match-aware,
  2. `min_relevance` still means DECAY - composing it was tried and reverted
     because bm25 magnitude scales with corpus size,
  3. `min_match` abstains, and does so at BOTH gates.
"""

import pytest

from kairn.core.experience import ExperienceEngine
from kairn.core.fts import bm25_to_relevance
from kairn.events.bus import EventBus


@pytest.fixture
async def engine(store):
    return ExperienceEngine(store, EventBus())


class TestBm25ToRelevance:
    def test_browse_query_has_no_match_signal(self):
        assert bm25_to_relevance(None) == 1.0

    def test_stronger_match_scores_higher(self):
        assert bm25_to_relevance(-12.0) > bm25_to_relevance(-3.0)

    def test_bounded(self):
        # Closed at BOTH ends after the 4-decimal rounding: a vanishing match
        # rounds to 0.0 and an overwhelming one rounds to 1.0. Asserting an
        # OPEN interval was my error, not the function's.
        assert bm25_to_relevance(-0.0001) == 0.0
        assert bm25_to_relevance(-1e9) == 1.0


class TestReportedRelevanceIsMatchAware:
    @pytest.mark.asyncio
    async def test_recall_relevance_is_below_pure_decay(self, engine):
        """A fresh experience has decay ~1.0; its reported recall relevance
        must NOT be ~1.0 just because it is recent."""
        await engine.save(content="Rust borrow checker lifetimes", type="gotcha")
        results = await engine.search(text="rust borrow checker", limit=5)
        assert results, "sanity: the query must retrieve the experience"
        exp = results[0]
        assert exp.recall_relevance is not None
        assert exp.recall_relevance < 1.0
        # and it must be strictly below the pure-decay value it replaced
        from datetime import UTC, datetime
        assert exp.recall_relevance < exp.relevance(at=datetime.now(UTC))

    @pytest.mark.asyncio
    async def test_browse_query_keeps_pure_decay(self, engine):
        """No text means no match signal, so nothing should change."""
        await engine.save(content="Browse me", type="gotcha")
        results = await engine.search(text=None, limit=5)
        assert results
        assert results[0].recall_relevance is None


class TestMinRelevanceStillMeansDecay:
    @pytest.mark.asyncio
    async def test_a_fresh_strong_match_survives_a_positive_floor(self, engine):
        """Regression pin. Gating on decay*match instead crushed small stores:
        every score collapsed and any positive floor filtered everything."""
        await engine.save(content="Fresh solution about queues", type="solution")
        results = await engine.search(text="queues", min_relevance=0.3, limit=5)
        assert len(results) == 1


class TestMinMatchAbstains:
    @pytest.mark.asyncio
    async def test_high_min_match_abstains(self, engine):
        await engine.save(content="Kubernetes helm chart values", type="gotcha")
        assert await engine.search(text="kubernetes helm", limit=5)
        gated = await engine.search(text="kubernetes helm", min_match=0.99, limit=5)
        assert gated == [], f"a 0.99 match floor must abstain, got {gated}"

    @pytest.mark.asyncio
    async def test_zero_min_match_is_the_unchanged_default(self, engine):
        await engine.save(content="Kubernetes helm chart values", type="gotcha")
        a = await engine.search(text="kubernetes helm", limit=5)
        b = await engine.search(text="kubernetes helm", min_match=0.0, limit=5)
        assert [e.id for e in a] == [e.id for e in b]

    @pytest.mark.asyncio
    async def test_bitemporal_gate_honours_min_match_too(self, engine):
        """The SECOND gate. Fixing only search() would look exactly like a fix
        while bi-temporal recall kept leaking (Kairn d5272a91)."""
        await engine.save(content="Kubernetes helm chart values", type="gotcha")
        assert await engine.search_bitemporal(text="kubernetes helm", limit=5)
        gated = await engine.search_bitemporal(
            text="kubernetes helm", min_match=0.99, limit=5
        )
        assert gated == [], f"bitemporal must abstain too, got {gated}"
