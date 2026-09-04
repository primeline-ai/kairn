"""A question with nothing searchable in it must not be answered with browse.

`_to_fts_query` returns None for BOTH "no query was given" (a deliberate
browse) and "a query whose every word is a stop word" (a question with nothing
searchable in it). Any node path that reads that None as "browse" answers an
unanswerable question with its newest rows, and `bm25_to_relevance(None)`
labels them 1.0 - a REPORTING contract ("no match to report") read as a
perfect score.

Measured on `recall` at the SHIPPED DEFAULT, so no abstention floor could ever
have caught it: `recall("is it ok")` returned 0 experiences and 5 unrelated
nodes at relevance 1.0.

THIS FILE EXISTS BECAUSE OF HOW THAT WAS MISSED. The first fix closed the
EXPERIENCE half, and a docstring then ASSERTED the node half was exempt, on
the reasoning that "nodes have no browse-means-everything failure mode". No
test stood behind that sentence and it was false. An exemption asserted
without a test is a blind spot wearing a docstring, so every one of the three
surfaces is checked here by NAME - the two that are genuinely exempt included,
because an exemption that nobody re-checks is how the next reader inherits
this.

    surface     state    why
    recall      NEEDS    fixed: `asked_but_unsearchable` returns [] nodes
    crossref    EXEMPT   already `if fts_query: ... else: ranked = []`
    context     EXEMPT   router-first on RAW keywords, and its FTS fallback is
                         guarded by `if not nodes and fts_query`
"""
import pytest

from kairn.core.experience import ExperienceEngine
from kairn.core.fts import to_fts_query
from kairn.core.graph import GraphEngine
from kairn.core.ideas import IdeaEngine
from kairn.core.intelligence import IntelligenceLayer
from kairn.core.memory import ProjectMemory
from kairn.core.router import ContextRouter
from kairn.events.bus import EventBus


@pytest.fixture
async def intelligence(store):
    """A layer over a store carrying a few nodes, so "returns no nodes" is a
    real result rather than an empty store."""
    bus = EventBus()
    graph = GraphEngine(store, bus)
    for name in ("Kubernetes helm chart values drift",
                 "Rust borrow checker lifetimes",
                 "Postgres autovacuum on a hot partition"):
        await graph.add_node(name=name, type="learned_gotcha", description=name)
    return IntelligenceLayer(
        store=store, event_bus=bus, graph=graph,
        router=ContextRouter(store, bus), memory=ProjectMemory(store, bus),
        experience=ExperienceEngine(store, bus), ideas=IdeaEngine(store, bus),
    )

ALL_STOP_WORDS = "is it ok"
REAL_QUESTION = "kubernetes helm chart"


def test_the_premise_holds_all_stop_words_shape_to_nothing():
    """Without this the three tests below would pass vacuously on a query that
    never reached the None branch at all."""
    assert to_fts_query(ALL_STOP_WORDS) is None
    assert to_fts_query(REAL_QUESTION) is not None


class TestRecall:
    @pytest.mark.asyncio
    async def test_an_unsearchable_question_returns_no_nodes(self, intelligence):
        rows = await intelligence.recall(topic=ALL_STOP_WORDS)
        nodes = [x for x in rows if x.get("source") == "node"]
        assert nodes == [], nodes

    @pytest.mark.asyncio
    async def test_a_real_question_still_matches(self, intelligence):
        """The control that stops the fix being 'return nothing'."""
        rows = await intelligence.recall(topic=REAL_QUESTION)
        nodes = [x for x in rows if x.get("source") == "node"]
        assert nodes, "a real question must still match a node: %s" % rows


class TestTheTwoExemptSurfaces:
    """Their exemption is structural, so it is asserted against the SOURCE.

    A behavioural test here would need a populated store per surface and would
    still not say WHY they are safe; the guard is one line in each, and if it
    ever goes, this fails and names it."""

    def test_crossref_returns_an_empty_ranking_rather_than_browsing(self):
        import inspect

        from kairn.core.intelligence import IntelligenceLayer

        src = inspect.getsource(IntelligenceLayer.crossref)
        assert "if fts_query:" in src, (
            "crossref no longer guards its node query on a shaped fts_query, so "
            "an all-stop-word problem now browses - see this module's docstring"
        )

    def test_context_guards_its_fts_fallback_on_a_shaped_query(self):
        import inspect

        from kairn.core.intelligence import IntelligenceLayer

        src = inspect.getsource(IntelligenceLayer.context)
        assert "if not nodes and fts_query:" in src, (
            "context no longer guards its FTS fallback on a shaped fts_query, so "
            "an all-stop-word keyword string now browses - see the docstring"
        )
