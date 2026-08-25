"""context()'s nodes now carry a real ranking, and the ranking has no window.

The defect these pin: `context()` discovered nodes through the router, which
scores every route the same (0.5 on a real 23k-route store), then kept the first
`limit` of them. On 40 real hook queries the median candidate pool was 510 live
nodes and 10 came back - an arbitrary ten. The fix ranks the candidate ids by
bm25 with the match RESTRICTED to those ids, so there is no top-K window whose
size the result could be a property of.

Every test here is written so it fails when the behaviour is removed. The
mutation log lives in the commit message.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from kairn.core.experience import ExperienceEngine
from kairn.core.graph import GraphEngine
from kairn.core.ideas import IdeaEngine
from kairn.core.intelligence import IntelligenceLayer, _fold_diacritics
from kairn.core.memory import ProjectMemory
from kairn.core.router import ContextRouter
from kairn.events.bus import EventBus
from kairn.relevance import (
    RELEVANCE_KIND_MATCH,
    RELEVANCE_KIND_UNSCORED,
    RELEVANCE_KINDS,
)
from kairn.storage import sqlite_store as sqlite_store_module
from kairn.storage.sqlite_store import SQLiteStore


@pytest_asyncio.fixture
async def store(tmp_path):
    s = SQLiteStore(tmp_path / "ctxrank.db")
    await s.initialize()
    yield s
    await s.close()


@pytest_asyncio.fixture
async def engine(store):
    bus = EventBus()
    yield IntelligenceLayer(
        store=store,
        event_bus=bus,
        graph=GraphEngine(store, bus),
        router=ContextRouter(store, bus),
        memory=ProjectMemory(store, bus),
        experience=ExperienceEngine(store, bus),
        ideas=IdeaEngine(store, bus),
    )


async def _node(store: SQLiteStore, node_id: str, name: str, description: str) -> None:
    await store.insert_node(
        {
            "id": node_id,
            "namespace": "knowledge",
            "type": "concept",
            "name": name,
            "description": description,
            "properties": None,
            "tags": None,
            "created_by": None,
            "visibility": "workspace",
            "source_type": None,
            "source_ref": None,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": None,
        }
    )


# --- the behaviour --------------------------------------------------------


@pytest.mark.asyncio
async def test_best_match_survives_a_candidate_pool_far_larger_than_limit(engine, store):
    """The whole point, in one test.

    30 nodes all route on the keyword "harpsichord". 29 of them mention it once
    and nothing else. ONE - deliberately placed LAST in the route's id array, so
    the old truncate-then-return path could never reach it - answers the entire
    query. At limit=3 it has to come back, and first.
    """
    ids = []
    for i in range(29):
        nid = f"filler{i:02d}"
        await _node(store, nid, f"Harpsichord note {i}", "unrelated bookkeeping text")
        ids.append(nid)
    await _node(
        store,
        "winner",
        "Harpsichord temperament tuning",
        "werckmeister temperament tuning for a harpsichord",
    )
    ids.append("winner")
    await store.upsert_route("harpsichord", ids, 0.5)

    result = await engine.context(
        keywords="harpsichord werckmeister temperament tuning", limit=3
    )
    returned = [n["id"] for n in result["nodes"]]

    assert len(returned) == 3, returned
    assert returned[0] == "winner", (
        f"the one node answering the whole query is not first: {returned}"
    )
    assert result["nodes"][0]["relevance"] > result["nodes"][1]["relevance"]


@pytest.mark.asyncio
async def test_nodes_come_back_in_descending_relevance(engine, store):
    """Three nodes answering 3, 2 and 1 of the query's rare terms must come back
    in that order, with strictly decreasing relevance.

    The filler nodes are not decoration. bm25 weights a term by how RARE it is,
    so on a corpus where every document contains every term the scores collapse
    to a rounding artefact and the assertion below would hold for any order at
    all. The fillers give the discriminating terms something to be rare against.
    """
    ids = []
    for i in range(20):
        nid = f"filler{i:02d}"
        await _node(store, nid, f"Harpsichord {i}", "routine maintenance log")
        ids.append(nid)
    await _node(store, "three", "Harpsichord A", "werckmeister kirnberger vallotti")
    await _node(store, "two", "Harpsichord B", "werckmeister kirnberger")
    await _node(store, "one", "Harpsichord C", "werckmeister")
    ids += ["three", "two", "one"]
    await store.upsert_route("harpsichord", ids, 0.5)

    result = await engine.context(
        keywords="harpsichord werckmeister kirnberger vallotti", limit=3
    )
    returned = [n["id"] for n in result["nodes"]]
    scores = [n["relevance"] for n in result["nodes"]]

    assert returned == ["three", "two", "one"], returned
    assert scores == sorted(scores, reverse=True), scores
    assert len(set(scores)) == 3, f"no spread - the ranking proves nothing: {scores}"


@pytest.mark.asyncio
async def test_ranking_never_reaches_outside_the_routed_candidates(engine, store):
    """The mutant this exists to kill: replacing the id restriction with a
    global top-K window (`query_ranked(limit=limit*4)`).

    Every other test in this file passes under that mutant, because their whole
    corpus IS the candidate pool - so a global search and a restricted one give
    the same answer. THIS fixture makes them differ: 60 nodes match the query
    text and are NOT routed. A global window fills with them; the restriction
    cannot see them at all.
    """
    routed = []
    for i in range(5):
        nid = f"routed{i}"
        await _node(store, nid, f"Harpsichord {i}", "werckmeister")
        routed.append(nid)
    await _node(store, "routed_best", "Harpsichord best", "werckmeister kirnberger vallotti")
    routed.append("routed_best")
    for i in range(60):
        await _node(
            store,
            f"outsider{i:02d}",
            f"Outsider {i}",
            "werckmeister kirnberger vallotti harpsichord",
        )
    await store.upsert_route("harpsichord", routed, 0.5)

    result = await engine.context(
        keywords="harpsichord werckmeister kirnberger vallotti", limit=5
    )
    returned = [n["id"] for n in result["nodes"]]
    assert returned, "nothing came back at all"
    outside = [i for i in returned if i not in set(routed)]
    assert not outside, f"nodes the router never selected leaked in: {outside}"
    assert returned[0] == "routed_best", returned


@pytest.mark.asyncio
async def test_routed_but_non_matching_node_is_kept_and_labelled_unscored(engine, store):
    """Recall must not shrink. A node the router reached but the text does not
    match is still returned - last, at relevance 0.0, labelled `unscored`.

    The route is written by hand precisely so the node's own text shares nothing
    with the query; routes built from node text could not produce this case.
    """
    await _node(store, "match", "Harpsichord tuning", "harpsichord tuning notes")
    await _node(store, "stranger", "Diesel injector torque", "wholly unrelated")
    await store.upsert_route("harpsichord", ["match", "stranger"], 0.5)

    result = await engine.context(keywords="harpsichord tuning", limit=5)
    by_id = {n["id"]: n for n in result["nodes"]}

    assert "stranger" in by_id, "a routed node was dropped - that is a recall loss"
    assert by_id["stranger"]["relevance_kind"] == RELEVANCE_KIND_UNSCORED
    assert by_id["stranger"]["relevance"] == 0.0
    assert by_id["match"]["relevance_kind"] == RELEVANCE_KIND_MATCH
    ordered = [n["id"] for n in result["nodes"]]
    assert ordered.index("match") < ordered.index("stranger")


@pytest.mark.asyncio
async def test_every_node_carries_a_valid_relevance_kind(engine, store):
    await _node(store, "a", "Harpsichord tuning", "harpsichord tuning")
    await store.upsert_route("harpsichord", ["a"], 0.5)
    result = await engine.context(keywords="harpsichord tuning", limit=5)
    assert result["nodes"]
    for n in result["nodes"]:
        assert n["relevance_kind"] in RELEVANCE_KINDS, n
        assert "confidence" in n, "the pre-existing confidence field must survive"


@pytest.mark.asyncio
async def test_limit_is_still_honoured_with_a_large_pool(engine, store):
    ids = []
    for i in range(40):
        nid = f"m{i:02d}"
        await _node(store, nid, f"Harpsichord {i}", "harpsichord tuning")
        ids.append(nid)
    await store.upsert_route("harpsichord", ids, 0.5)
    result = await engine.context(keywords="harpsichord tuning", limit=4)
    assert len(result["nodes"]) == 4


@pytest.mark.asyncio
async def test_fold_diacritics_ascii_shortcut_is_equivalent():
    """`_fold_diacritics` short-circuits on ASCII for speed. Prove the shortcut
    returns exactly what the unoptimised definition returns.

    Compared against a local reference implementation, not against itself - a
    self-comparison would pass however the shortcut behaved.
    """
    import unicodedata

    def reference(text: str) -> str:
        return "".join(
            c
            for c in unicodedata.normalize("NFD", text)
            if not unicodedata.combining(c)
        )

    cases = [
        "",
        "plain ascii tokens",
        "Zürich",
        "Änderung Prüfung Übersicht größer für",
        "über",  # already decomposed
        "ÅΩé",
        "straße",
        "mixed ascii and ü in one string",
        "_underscore-and-digits-123",
    ]
    for case in cases:
        assert _fold_diacritics(case) == reference(case), case


# --- the storage-layer contract the ranking rests on ----------------------


@pytest.mark.asyncio
async def test_id_restriction_does_not_change_a_rows_bm25_rank(store):
    """bm25 comes from corpus-global statistics, not the result set.

    If restricting the match changed a row's rank, ranking a candidate subset
    would not be comparable to ranking the whole corpus and the merge in
    `_query_nodes_fts_by_ids` would be unsound.
    """
    for i in range(12):
        await _node(store, f"r{i:02d}", f"Harpsichord {i}", "harpsichord tuning notes")
    unrestricted = await store.query_nodes(text='"harpsichord" OR "tuning"', limit=100)
    global_ranks = {r["id"]: r["rank"] for r in unrestricted}
    subset = [f"r{i:02d}" for i in (1, 4, 7)]
    restricted = await store.query_nodes(
        text='"harpsichord" OR "tuning"', node_ids=subset, limit=100
    )
    assert {r["id"] for r in restricted} == set(subset)
    for row in restricted:
        assert row["rank"] == global_ranks[row["id"]], row["id"]


@pytest.mark.asyncio
async def test_chunking_reproduces_the_single_shot_result(store, monkeypatch):
    """The id list is chunked to stay under SQLITE_LIMIT_VARIABLE_NUMBER (999 on
    older SQLite builds). Chunking must not change the answer."""
    ids = []
    for i in range(20):
        nid = f"c{i:02d}"
        # Vary document length: bm25 is length-normalised, so this gives the
        # rows genuinely different ranks. Equal ranks would make the "is it
        # sorted" assertion below true for any order at all.
        #
        # The padding DECREASES with i on purpose, so rank order is the exact
        # reverse of id order. Chunks are built in id order, so a merge that
        # forgets to re-sort produces a visibly wrong sequence. With the
        # padding increasing instead, id order and rank order coincide and the
        # missing sort is undetectable - that fixture was tried first and let a
        # mutant through.
        await _node(
            store, nid, f"Harpsichord {i}", "harpsichord tuning" + " padding" * (19 - i)
        )
        ids.append(nid)

    single = await store.query_nodes(
        text='"harpsichord" OR "tuning"', node_ids=ids, limit=100
    )
    monkeypatch.setattr(sqlite_store_module, "_MAX_ID_BINDINGS", 3)
    chunked = await store.query_nodes(
        text='"harpsichord" OR "tuning"', node_ids=ids, limit=100
    )
    assert [(r["id"], r["rank"]) for r in chunked] == [
        (r["id"], r["rank"]) for r in single
    ]
    assert len(single) == 20, "fixture too small to exercise more than one chunk"
    # Comparing the two against EACH OTHER cannot catch a missing post-merge
    # sort, because both paths would lose it together. Assert the absolute
    # property as well: chunks are appended in id order, so without the re-sort
    # this list is not in rank order.
    chunked_ranks = [r["rank"] for r in chunked]
    assert chunked_ranks == sorted(chunked_ranks), chunked_ranks
    assert len(set(chunked_ranks)) > 1, "all ranks equal - sortedness proves nothing"


@pytest.mark.asyncio
async def test_empty_id_set_and_none_are_different(store):
    """`[]` restricts to nothing. `None` means no restriction. Collapsing the
    two would silently turn a "no candidates" call into a corpus-wide search."""
    for i in range(3):
        await _node(store, f"e{i}", f"Harpsichord {i}", "harpsichord tuning")
    assert await store.query_nodes(text='"harpsichord"', node_ids=[], limit=10) == []
    assert len(await store.query_nodes(text='"harpsichord"', node_ids=None, limit=10)) == 3


@pytest.mark.asyncio
async def test_id_restriction_without_text_is_rejected(store):
    with pytest.raises(ValueError, match="node_ids requires a text query"):
        await store.query_nodes(node_ids=["x"], limit=10)


@pytest.mark.asyncio
async def test_route_candidates_returns_the_pool_route_truncates(store):
    """`route()` is now `route_candidates()` plus the liveness truncation, so
    the pool has to be strictly larger than what `route()` hands back."""
    bus = EventBus()
    router = ContextRouter(store, bus)
    ids = []
    for i in range(15):
        nid = f"p{i:02d}"
        await _node(store, nid, f"Harpsichord {i}", "harpsichord")
        ids.append(nid)
    await store.upsert_route("harpsichord", ids, 0.5)

    candidates = await router.route_candidates("harpsichord")
    routed = await router.route("harpsichord", limit=4)
    assert len(candidates) == 15
    assert len(routed) == 4
    assert [r["node"]["id"] for r in routed] == list(candidates)[:4]


@pytest.mark.asyncio
async def test_soft_deleted_candidates_do_not_reach_the_caller(engine, store):
    """Soft-deleted ids stay in route arrays on purpose (restore keeps them
    routable). They must not appear in the ranked output either - the FTS join
    filters them, and this pins that it still does."""
    await _node(store, "live", "Harpsichord tuning", "harpsichord tuning")
    await _node(store, "dead", "Harpsichord tuning gone", "harpsichord tuning")
    await store.soft_delete_node("dead")
    await store.upsert_route("harpsichord", ["dead", "live"], 0.5)

    result = await engine.context(keywords="harpsichord tuning", limit=5)
    assert [n["id"] for n in result["nodes"]] == ["live"]


# --- what external review found in the first version of this change -------


@pytest.mark.asyncio
async def test_a_duplicate_id_spanning_two_chunks_does_not_duplicate_a_row(
    store, monkeypatch
):
    """`IN (...)` is set membership; chunk-and-extend is not.

    With the same id in two different chunks the naive merge returned the row
    twice, which under `limit` displaces a different node entirely. Chunk size
    is forced to 2 so the duplicate genuinely lands in separate queries.
    """
    for nid in ("a", "b", "c"):
        await _node(store, nid, f"Harpsichord {nid}", "harpsichord tuning")
    monkeypatch.setattr(sqlite_store_module, "_MAX_ID_BINDINGS", 2)

    rows = await store.query_nodes(
        text='"harpsichord" OR "tuning"', node_ids=["a", "c", "a", "b"], limit=3
    )
    ids = [r["id"] for r in rows]
    assert sorted(ids) == ["a", "b", "c"], ids
    assert len(ids) == len(set(ids)), f"a row came back twice: {ids}"


@pytest.mark.asyncio
async def test_equal_ranks_order_the_same_however_the_list_is_chunked(
    store, monkeypatch
):
    """Sorting on `rank` alone is stable over the CONCATENATION order, which
    differs between one chunk and many. Identical documents give identical
    ranks, so this fixture has nothing BUT ties."""
    ids = []
    for i in range(9):
        nid = f"t{i}"
        await _node(store, nid, "Harpsichord", "harpsichord tuning")
        ids.append(nid)
    # The id list is REVERSED on purpose. Chunks are cut from it in order and
    # each chunk comes back in the table's own order, so with the list in
    # insertion order the concatenation happens to match the single-shot result
    # and the missing tie-break is invisible. That fixture was tried first and
    # let the mutant through.
    probe = list(reversed(ids))
    single = [r["id"] for r in await store.query_nodes(
        text='"harpsichord" OR "tuning"', node_ids=probe, limit=100)]
    ranks = [r["rank"] for r in await store.query_nodes(
        text='"harpsichord" OR "tuning"', node_ids=probe, limit=100)]
    assert len(set(ranks)) == 1, f"fixture has no ties, so it tests nothing: {ranks}"

    monkeypatch.setattr(sqlite_store_module, "_MAX_ID_BINDINGS", 2)
    chunked = [r["id"] for r in await store.query_nodes(
        text='"harpsichord" OR "tuning"', node_ids=probe, limit=100)]
    assert chunked == single, f"chunked {chunked} != single {single}"


@pytest.mark.asyncio
async def test_route_at_limit_zero_returns_nothing(store):
    """A deliberate fix, not an accident: the old loop appended before testing
    the count, so asking for zero nodes returned one."""
    bus = EventBus()
    router = ContextRouter(store, bus)
    await _node(store, "z", "Harpsichord", "harpsichord")
    await store.upsert_route("harpsichord", ["z"], 0.5)
    assert await router.route("harpsichord", limit=0) == []
    assert await router.route("harpsichord", limit=-1) == []
    assert len(await router.route("harpsichord", limit=1)) == 1


@pytest.mark.asyncio
async def test_fallback_fts_path_is_ordered_by_the_relevance_it_reports(engine, store):
    """When the router finds nothing, `context()` falls back to a corpus-wide
    FTS search. That path reports a coverage-scaled relevance, so it has to be
    SORTED by it - otherwise the order is raw bm25 and disagrees with the
    number beside each node.

    No route is created at all here, which is what forces the fallback.

    The fixture is built so coverage genuinely REVERSES the raw bm25 order -
    two of the three query terms are common (30 padding documents carry them)
    and one is rare, so `narrow` wins on raw bm25 by hammering the rare term
    while `broad` wins on coverage by answering all three. Measured on this
    exact fixture:

        narrow   raw 0.4577  coverage 0.333  scaled 0.1526
        broad    raw 0.3652  coverage 1.000  scaled 0.3652

    A fixture where the two orders agree makes the assertion below vacuous.
    """
    await _node(store, "narrow", "Narrow", "werckmeister werckmeister werckmeister")
    await _node(store, "broad", "Broad", "werckmeister alpha beta")
    for i in range(30):
        await _node(store, f"pad{i}", f"Pad {i}", "alpha beta routine log")

    result = await engine.context(keywords="werckmeister alpha beta", limit=5)
    assert result["nodes"], "the fallback returned nothing"
    scores = [n["relevance"] for n in result["nodes"]]
    assert scores == sorted(scores, reverse=True), (
        f"reported relevance disagrees with the returned order: "
        f"{[(n['id'], n['relevance']) for n in result['nodes']]}"
    )
    assert len(set(scores)) > 1, "no spread - the ordering assertion proves nothing"
    ids = [n["id"] for n in result["nodes"]]
    assert ids.index("broad") < ids.index("narrow"), (
        f"raw bm25 order survived the coverage scaling: {ids}"
    )
