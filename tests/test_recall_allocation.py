"""Does recall() actually return both sources when both match?"""
import pytest
import pytest_asyncio

from kairn.core.experience import ExperienceEngine
from kairn.core.graph import GraphEngine
from kairn.core.ideas import IdeaEngine
from kairn.core.intelligence import IntelligenceLayer
from kairn.core.memory import ProjectMemory
from kairn.core.router import ContextRouter
from kairn.events.bus import EventBus
from kairn.storage.sqlite_store import SQLiteStore


@pytest_asyncio.fixture
async def engine(tmp_path):
    store = SQLiteStore(tmp_path / "alloc.db")
    await store.initialize()
    bus = EventBus()
    yield IntelligenceLayer(
        store=store, event_bus=bus, graph=GraphEngine(store, bus),
        router=ContextRouter(store, bus), memory=ProjectMemory(store, bus),
        experience=ExperienceEngine(store, bus), ideas=IdeaEngine(store, bus))


@pytest.mark.asyncio
async def test_recall_returns_both_sources_at_a_small_limit(engine):
    """The defect, stated as a test.

    Ten nodes and ten experiences all match. At limit=6 the caller should see
    BOTH kinds. Today nodes are appended first and results[:limit] cuts before
    a single experience is reached - measured against the live store as
    72 nodes / 0 experiences at limit 6.
    """
    for i in range(10):
        await engine.learn(content=f"wal checkpoint starvation note {i}",
                           type="gotcha", confidence="high")     # -> node
        await engine.learn(content=f"wal checkpoint starvation trace {i}",
                           type="gotcha", confidence="low")      # -> experience
    results = await engine.recall(topic="wal checkpoint starvation", limit=6)
    kinds = {r["source"] for r in results}
    assert len(results) == 6, f"expected 6 results, got {len(results)}"
    assert "node" in kinds, f"no node rows: {[r['source'] for r in results]}"
    assert "experience" in kinds, (
        f"no experience rows at limit=6 - the node list consumed the whole "
        f"budget: {[r['source'] for r in results]}"
    )


# Each row answers a DIFFERENT number of the query's terms, so the ranking has
# a defined order instead of twenty near-ties.
#
# WHY THIS MATTERS HERE, measured rather than assumed. The fixture used to give
# every row the same three query terms, which was fine while experiences were
# ordered by pure time-decay: the rows were created milliseconds apart, so
# decay ordered them and it ordered them the same way on every call. The
# experience score is now bm25 * term_coverage with a bounded recency nudge, so
# twenty identically-matching rows differ only in the nudge - and the nudge is
# recomputed from `datetime.now()` on each call. Two calls milliseconds apart
# then legitimately disagree about which near-tie leads, and this test compares
# exactly that: one call at limit=50 against another at limit=6. The node half
# kept passing throughout, because nodes carry no age term.
#
# That is Kairn `ec5b2241`: a compensatory ranking has no stable total order
# over near-ties across two instants. The property this test exists for -
# allocation preserves each group's internal rank - is unaffected, so the
# fixture is what changes, not the assertion.
_TERMS = ["wal", "checkpoint", "starvation", "reader", "commit", "throttle"]


def _graded(i: int, kind: str) -> str:
    """Content answering `i % 4 + 2` of the query's six terms, so scores differ."""
    return " ".join(_TERMS[: (i % 4) + 2]) + f" {kind} {i}"


@pytest.mark.asyncio
async def test_recall_preserves_rank_inside_each_group(engine):
    """The commit claimed "rank inside each group is preserved exactly" and
    nothing tested it. A reviewer applied `reversed(nodes_out)` and the whole
    679-test suite still passed."""
    for i in range(10):
        await engine.learn(content=_graded(i, "note"),
                           type="gotcha", confidence="high")
        await engine.learn(content=_graded(i, "trace"),
                           type="gotcha", confidence="low")
    topic = " ".join(_TERMS)
    big = await engine.recall(topic=topic, limit=50)
    small = await engine.recall(topic=topic, limit=6)
    for src in ("node", "experience"):
        full = [r["id"] for r in big if r["source"] == src]
        got = [r["id"] for r in small if r["source"] == src]
        assert got, f"no {src} rows at limit=6"
        assert len(set(full)) == len(full), f"duplicate {src} ids in the full list"
        assert got == full[:len(got)], (
            f"{src} rank not preserved: allocation returned {got}, "
            f"the head of the full ranking is {full[:len(got)]}"
        )


@pytest.mark.asyncio
async def test_crossref_also_allocates_across_sources(engine):
    """Same defect, one function away. Measured side by side before the fix:
    recall -> 3 nodes / 3 experiences, crossref -> 6 nodes / 0."""
    for i in range(10):
        await engine.learn(content=f"wal checkpoint starvation note {i}",
                           type="gotcha", confidence="high")
        await engine.learn(content=f"wal checkpoint starvation trace {i}",
                           type="gotcha", confidence="low")
    results = await engine.crossref(problem="wal checkpoint starvation", limit=6)
    rows = results if isinstance(results, list) else results.get("results", [])
    kinds = {r["source"] for r in rows}
    assert len(rows) == 6, f"expected 6, got {len(rows)}"
    assert kinds == {"node", "experience"}, (
        f"crossref returned one source only: {[r['source'] for r in rows]}"
    )


@pytest.mark.asyncio
async def test_access_is_credited_only_to_rows_the_caller_received(engine):
    """`exp_auto_promote` fires on access_count, so crediting the full fetch
    pushed experiences the caller never saw toward promotion on every call."""
    # BOTH sources, and this is the part the first version got wrong. With
    # experiences only, `experience.search(limit=limit)` fetches exactly as many
    # rows as the allocation shows, so crediting "the full fetch" and crediting
    # "the kept rows" are the same list and the mutant survived. The gap only
    # opens when nodes take half the budget.
    for i in range(10):
        await engine.learn(content=f"wal checkpoint starvation note {i}",
                           type="gotcha", confidence="high")   # -> node
        await engine.learn(content=f"wal checkpoint starvation trace {i}",
                           type="gotcha", confidence="low")    # -> experience
    results = await engine.recall(topic="wal checkpoint starvation", limit=4)
    returned = {r["id"] for r in results if r["source"] == "experience"}
    assert returned, "no experiences returned - this test would be vacuous"
    assert any(r["source"] == "node" for r in results), (
        "no nodes returned - without them the fetch and the allocation are the "
        "same list and this test cannot detect over-crediting"
    )
    assert len(returned) < 4, (
        f"the allocation returned {len(returned)} of 4 experiences - it must be "
        f"fewer than the fetch for this test to discriminate"
    )

    all_exps = await engine.experience.search(text=None, limit=50)
    touched = {e.id for e in all_exps if e.access_count > 0}
    assert touched, "nothing was credited at all - the fix went too far"
    assert touched <= returned, (
        f"credited {len(touched)} experiences but returned {len(returned)}: "
        f"{sorted(touched - returned)} were never shown to the caller"
    )


@pytest.mark.asyncio
async def test_allocator_keeps_rows_from_an_unknown_source(engine):
    """The partition drops anything that is neither node nor experience; the
    old `results[:limit]` would have kept it. Tested directly because no
    surface emits a third source today - which is exactly why a refactor could
    add one and nobody would notice."""
    from kairn.core.intelligence import _allocate_across_sources

    rows = [{"source": "node", "id": "n1"}, {"source": "idea", "id": "i1"},
            {"source": "experience", "id": "e1"}]
    out = _allocate_across_sources(rows, 10)
    assert {r["id"] for r in out} == {"n1", "i1", "e1"}, (
        f"a row was dropped by the partition: {[r['id'] for r in out]}"
    )
