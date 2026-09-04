"""The three review findings that blocked the match-aware experience ranking.

Kairn `00f674bd` rejected the first cut of this feature. These tests pin the
three defects that were reachable at DEFAULT config, i.e. without anyone
opting in:

F1  `crossref` re-sorted the combined node+experience list on `relevance`,
    but node relevance there was the hardcoded constant 1.0 - a placeholder,
    not a measurement. Sorting a measurement against a placeholder means the
    placeholder always wins, so a strongly-matching experience lost its slot
    to a barely-matching node and the `[:limit]` truncation dropped it.

F4  The abstention floor `experience_min_match` reached ONE of the THREE
    `self.experience.search(...)` call sites in this module. `crossref` and
    `context` ran unfloored. A fix applied to N-1 of N sites looks exactly
    like a fix (Kairn `d5272a91`), and the review's mutation control proved
    it: deleting the single wiring line left 670 tests green.

F2  The reported relevance for a node collapsed to exactly 0.0 on a
    one-or-two document store, including for a verbatim 4-of-4 term match,
    because `round(match, 4)` cannot represent the ~1.1e-06 that bm25 yields
    when the corpus is too small to carry IDF. Any consumer filtering
    `relevance > 0` - and `min_relevance` itself, which compares against this
    same rounded value - therefore dropped everything on a fresh workspace.

Measured on this branch before the fix (instrument: scratch script over
`store.query_experiences` / `graph.query_ranked` raw ranks):

    docs   experience match      node match*coverage   node round(...,4)
       1   7.99999e-07           1.09999e-06           0.0
       2   7.99999e-07           1.11361e-06           0.0
       3   0.290105              0.363497              0.3635
      10   0.596230              0.674747              0.6747

so the degenerate regime is 1-2 documents and it recovers completely at 3.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

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

# The query every test below asks. Four searchable terms, so term coverage can
# separate a 4-of-4 match from a 1-of-4 one.
QUERY = "postgres connection pool exhaustion"
STRONG = "postgres connection pool exhaustion under sustained load"
# Shares exactly ONE of the four terms, so it is retrieved by the OR query and
# scores low on coverage.
WEAK = "connection etiquette for polite correspondence"
# All four terms, but long enough that bm25's length normalisation puts it
# BELOW a one-word note holding only the rarest term. Used by the coverage test.
LONG_FULL = (
    "postgres connection pool exhaustion happens when the service opens more "
    "sessions than the server allows and the operators must then decide whether "
    "to raise the ceiling or to queue the callers waiting for a free handle"
)


_SRC = Path(__file__).resolve().parents[1] / "src" / "kairn"

# Modules allowed to run a TEXT-query experience search without the abstention
# floor. EMPTY, deliberately: the package-wide census below is a real gate, not
# a report. Measured at the time of writing, all five text-query call sites
# forward it (intelligence.py x3, server.py kn_memories, cli.py memories); the
# three remaining sites are browse reads and are excluded by `has_text`, not by
# this list. A browse row has `rank is None` and `bm25_to_relevance(None)` is
# 1.0 by contract, so a floor at or below 1.0 is a no-op there - forwarding it
# would be noise, not safety.
#
# If a legitimate exemption ever appears, add the module here WITH the reason.
# An entry is a promise that someone checked, not a way to quiet the test.
_KNOWN_UNFLOORED_MODULES: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _SearchSite:
    lineno: int
    has_text: bool
    forwards_floor: bool


def _experience_search_sites(path: Path) -> list[_SearchSite]:
    """Every `<...experience...>.search(...)` CALL in a module, from the AST.

    The AST, not a substring count: the first version of this census counted
    the text `self.experience.search(`, which a comment mentioning it would
    have satisfied, and which cannot see whether `min_match` is a real keyword
    or a word inside a docstring.
    """
    sites: list[_SearchSite] = []
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "search":
            continue
        if "experience" not in ast.unparse(func.value):
            continue
        keywords = {kw.arg for kw in node.keywords}
        sites.append(
            _SearchSite(
                lineno=node.lineno,
                has_text="text" in keywords,
                forwards_floor="min_match" in keywords,
            )
        )
    return sites


async def _stack(tmp_path, name="p5.db", **intel_kwargs):
    store = SQLiteStore(tmp_path / name)
    await store.initialize()
    bus = EventBus()
    experience = ExperienceEngine(store, bus)
    graph = GraphEngine(store, bus)
    intel = IntelligenceLayer(
        store=store,
        event_bus=bus,
        graph=graph,
        router=ContextRouter(store, bus),
        memory=ProjectMemory(store, bus),
        experience=experience,
        ideas=IdeaEngine(store, bus),
        **intel_kwargs,
    )
    return store, intel


async def _fill_noise(intel, n):
    """Enough unrelated documents that bm25 has IDF to work with.

    At fewer than 3 documents bm25 is degenerate (see the table in the module
    docstring); these tests are about ranking, not about that regime, so they
    keep the corpus out of it deliberately.
    """
    for i in range(n):
        await intel.experience.save(
            content=f"unrelated filler note {i} about sourdough bread and baking",
            type="gotcha",
        )
        await intel.graph.add_node(
            name=f"filler node {i}",
            type="note",
            description=f"unrelated filler node {i} about sourdough bread and baking",
        )


@pytest_asyncio.fixture
async def ranked(tmp_path):
    """A store where the best answer is an EXPERIENCE and the nodes are weak.

    Three weakly-matching nodes (1 of 4 query terms) and one verbatim
    4-of-4 experience, with `limit=3` so exactly three of the four results
    survive the truncation.
    """
    store, intel = await _stack(tmp_path)
    await _fill_noise(intel, 12)
    for i in range(3):
        await intel.graph.add_node(name=f"weak node {i}", type="note", description=WEAK)
    await intel.experience.save(content=STRONG, type="solution")
    yield intel
    await store.close()


# ──────────────────────────────────────────────────────────────────────
# F1 - crossref sorted a measurement against a placeholder
# ──────────────────────────────────────────────────────────────────────


class TestCrossrefSortsOneScale:
    @pytest.mark.asyncio
    async def test_strong_experience_survives_the_truncation(self, ranked):
        """The repro. Three weak nodes at a hardcoded 1.0 take all three
        slots and the verbatim experience is truncated away."""
        results = await ranked.crossref(problem=QUERY, limit=3)

        assert len(results) == 3
        kinds = [r["source"] for r in results]
        assert "experience" in kinds, (
            "a 4-of-4 term match lost its slot to nodes that share ONE term: "
            f"{[(r['source'], r['relevance']) for r in results]}"
        )

    @pytest.mark.asyncio
    async def test_node_relevance_is_measured_not_a_placeholder(self, ranked):
        """A node that shares one term of four must not report the same
        relevance as anything else. 1.0 for every node is a constant."""
        results = await ranked.crossref(problem=QUERY, limit=10)
        node_scores = {r["relevance"] for r in results if r["source"] == "node"}

        assert node_scores, "sanity: the query must retrieve nodes at all"
        assert node_scores != {1.0}, (
            f"every node reports the placeholder 1.0: {node_scores}"
        )

    @pytest.mark.asyncio
    async def test_the_best_match_leads_its_own_group(self, ranked):
        """NARROWED, and the narrowing is the finding.

        This asserted `results[0]` was the 4-of-4 experience, which held while
        crossref ended in a plain sort. It no longer does: the sort is followed
        by `_allocate_across_sources`, which interleaves one row from each
        source in turn, nodes first, so slot 0 is always a node whenever any
        node matched.

        Those two rules genuinely conflict and the conflict was resolved in
        favour of allocation, because the two claims are not worth the same.
        F1's harm was that the strong experience was DROPPED by the truncation,
        and allocation is what prevents that - it guarantees a share of the
        budget to each source regardless of scores. "The best row is first" is
        a nicer property but nobody loses an answer to it. So the guarantee
        kept is: the strong match is returned, and it leads its own group.

        The stronger claim survives one function away, in `recall`, and is
        pinned there. Here it is deliberately not claimed."""
        results = await ranked.crossref(problem=QUERY, limit=10)

        exps = [r for r in results if r["source"] == "experience"]
        assert exps, "the 4-of-4 experience was not returned at all"
        assert exps[0]["content"] == STRONG, (
            "the strong experience did not lead the experience group: "
            f"{[(r['relevance'], r['content'][:30]) for r in exps]}"
        )
        # And it is not merely present at the end of a node-filled list: the
        # allocation must have given it an early slot.
        assert results.index(exps[0]) <= 1, [r["source"] for r in results]

    @pytest.mark.asyncio
    async def test_a_strong_node_still_outranks_a_weak_experience(self, tmp_path):
        """Positive control, the other direction: comparable scales must not
        become 'experiences always win'. Here the NODE holds the answer."""
        store, intel = await _stack(tmp_path)
        try:
            await _fill_noise(intel, 12)
            await intel.graph.add_node(
                name="postgres pooling", type="solution", description=STRONG
            )
            await intel.experience.save(content=WEAK, type="gotcha")

            results = await intel.crossref(problem=QUERY, limit=10)

            assert results[0]["source"] == "node"
        finally:
            await store.close()


# ──────────────────────────────────────────────────────────────────────
# F4 - the floor reached 1 of 3 call sites
# ──────────────────────────────────────────────────────────────────────

# Every surface in this module that searches experiences. The list is the
# test: a fourth surface added later without a floor fails to be covered here
# only if someone also forgets to add it, which is why the structural test
# below cross-checks it against the source rather than trusting the list.
SURFACES = ["recall", "crossref", "context"]


async def _call(intel, surface, query, limit=10):
    if surface == "recall":
        return await intel.recall(topic=query, limit=limit)
    if surface == "crossref":
        return await intel.crossref(problem=query, limit=limit)
    result = await intel.context(keywords=query, limit=limit)
    return result["experiences"]


def _experiences(surface, returned):
    if surface == "context":
        return returned
    return [r for r in returned if r["source"] == "experience"]


class TestAbstentionFloorReachesEverySite:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("surface", SURFACES)
    async def test_floor_is_forwarded(self, tmp_path, surface):
        """Structural. Records the `min_match` kwarg each surface actually
        passes, so a deleted wiring line fails here even if the behaviour
        happens to look the same. The review's mutation control removed
        exactly this line and 670 tests stayed green."""
        store, intel = await _stack(tmp_path, name=f"{surface}.db", experience_min_match=0.65)
        try:
            await _fill_noise(intel, 12)
            await intel.experience.save(content=STRONG, type="solution")

            seen: list[float | None] = []
            real = intel.experience.search

            async def spy(**kwargs):
                seen.append(kwargs.get("min_match"))
                return await real(**kwargs)

            intel.experience.search = spy
            await _call(intel, surface, QUERY)

            assert seen == [0.65], (
                f"{surface} passed min_match={seen} - the configured floor is 0.65"
            )
        finally:
            await store.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("surface", SURFACES)
    async def test_high_floor_abstains(self, tmp_path, surface):
        """Behavioural. A floor no bm25 match can clear must empty the
        experience side of EVERY surface, not just recall."""
        store, intel = await _stack(tmp_path, name=f"{surface}.db", experience_min_match=0.99)
        try:
            await _fill_noise(intel, 12)
            await intel.experience.save(content=STRONG, type="solution")

            returned = await _call(intel, surface, QUERY)

            assert _experiences(surface, returned) == []
        finally:
            await store.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("surface", SURFACES)
    async def test_default_floor_returns_the_experience(self, tmp_path, surface):
        """Positive control for the test above. Without it, an assertion of
        emptiness passes for any reason at all - a broken query, an empty
        store, a surface that never returns experiences."""
        store, intel = await _stack(tmp_path, name=f"{surface}.db")
        try:
            await _fill_noise(intel, 12)
            await intel.experience.save(content=STRONG, type="solution")

            returned = await _call(intel, surface, QUERY)

            assert _experiences(surface, returned), (
                f"{surface} returned no experience even at the default floor 0.0"
            )
        finally:
            await store.close()

    def test_the_census_detector_actually_detects(self, tmp_path):
        """Positive control for the INSTRUMENT, not for the code.

        The census asserts an ABSENCE (no unfloored surfaces). An absence that
        nobody proved the detector can see is not a measurement, so this feeds
        the scanner a module it MUST flag and one it must not. Without this,
        a broken matcher reports a clean package forever.
        """
        probe = tmp_path / "probe.py"
        probe.write_text(
            "async def floored(self):\n"
            "    return await self.experience.search(text=q, min_match=0.5)\n"
            "async def unfloored(self):\n"
            "    return await self.experience.search(text=q, limit=10)\n"
            "async def browsing(self):\n"
            "    return await self.experience.search(limit=10)\n"
            "async def unrelated(self):\n"
            "    return await self.graph.search(text=q)\n"
        )

        sites = _experience_search_sites(probe)

        assert len(sites) == 3, f"the matcher saw {len(sites)} sites, expected 3"
        assert [(s.has_text, s.forwards_floor) for s in sites] == [
            (True, True),
            (True, False),
            (False, False),
        ]

    def test_the_surface_list_covers_every_call_site_in_this_module(self):
        """The N-of-N guard, over the AST rather than over substrings.

        The first version counted `self.experience.search(` as TEXT, which a
        comment containing the token would have satisfied - a census that
        cannot fail. This walks real `Call` nodes and reads real keywords.
        """
        sites = _experience_search_sites(_SRC / "core" / "intelligence.py")
        text_sites = [s for s in sites if s.has_text]

        assert len(text_sites) == len(SURFACES), (
            f"{len(text_sites)} text-query experience searches in intelligence.py "
            f"but {len(SURFACES)} covered surfaces at lines "
            f"{[s.lineno for s in text_sites]} - add the new surface to SURFACES"
        )
        unfloored = [s.lineno for s in text_sites if not s.forwards_floor]
        assert not unfloored, (
            f"lines {unfloored} run a text query without forwarding the "
            "abstention floor"
        )

    def test_the_census_covers_the_whole_package_not_just_this_module(self):
        """Scope the census to the CLASS, not to the file being edited.

        A sweep scoped to the region you edit cannot find the class you are
        removing (Kairn `760b0025`, `f781b11b`), and the review's own count
        said 3-of-3 green while other modules ran the same search unfloored.
        `_KNOWN_UNFLOORED_MODULES` is the honest has/needs list: those files
        are outside this change's scope, and a NEW module joining them fails
        here instead of shipping quietly.
        """
        by_module: dict[str, list[int]] = {}
        text_sites = 0
        for path in sorted(_SRC.rglob("*.py")):
            for site in _experience_search_sites(path):
                if not site.has_text:
                    continue
                text_sites += 1
                if not site.forwards_floor:
                    by_module.setdefault(
                        str(path.relative_to(_SRC)), []
                    ).append(site.lineno)

        # NON-VACUITY FIRST. `set() - known` is empty, so a scan that matched
        # nothing at all would pass the real assertion silently. This is the
        # shape of a check that cannot fail (Kairn `ee2d1a9f`).
        assert text_sites >= len(SURFACES), (
            f"the AST scan found only {text_sites} text-query experience "
            f"searches across {_SRC}; it should find at least the "
            f"{len(SURFACES)} in intelligence.py, so the matcher is broken"
        )
        unexpected = set(by_module) - _KNOWN_UNFLOORED_MODULES
        assert not unexpected, (
            f"new unfloored experience-search surfaces: "
            f"{ {m: by_module[m] for m in unexpected} }"
        )


# ──────────────────────────────────────────────────────────────────────
# F2 - the reported relevance collapsed to exactly 0.0
# ──────────────────────────────────────────────────────────────────────


class TestTinyStoreReporting:
    @pytest.mark.asyncio
    async def test_node_relevance_is_not_zero_for_a_verbatim_match(self, tmp_path):
        """Two documents, a 4-of-4 term match, and the node reports 0.0."""
        store, intel = await _stack(tmp_path)
        try:
            await intel.graph.add_node(name="pg pooling", type="solution", description=STRONG)
            await intel.graph.add_node(
                name="bread", type="note", description="sourdough bread and baking"
            )

            results = await intel.recall(topic=QUERY, limit=10)
            nodes = [r for r in results if r["source"] == "node"]

            assert nodes, "sanity: the query must retrieve the node"
            assert [n for n in nodes if n["relevance"] > 0], (
                "every node reported relevance 0.0 on a 2-document store, so a "
                f"consumer filtering `relevance > 0` drops all of them: {nodes}"
            )
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_experience_relevance_is_not_zero_for_a_verbatim_match(self, tmp_path):
        """Regression pin for the half that is already right. The experience
        path reports at 6 decimals, which is what keeps 1.1e-06 off zero;
        rounding it to 4 like the node path would break this."""
        store, intel = await _stack(tmp_path)
        try:
            await intel.experience.save(content=STRONG, type="solution")
            await intel.experience.save(content="sourdough bread and baking", type="gotcha")

            results = await intel.recall(topic=QUERY, limit=10)
            exps = [r for r in results if r["source"] == "experience"]

            assert exps
            assert exps[0]["relevance"] > 0
        finally:
            await store.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("surface", ["recall", "crossref"])
    async def test_coverage_is_load_bearing_in_the_node_score(self, tmp_path, surface):
        """Added because a mutant survived: deleting `term_coverage` from
        `_node_relevance` broke nothing the other tests could see.

        bm25 rewards a rare term in a short document, so a one-word note
        containing only the rarest query term outscores the document that
        answers all four. Measured on this store (instrument: `graph.query_ranked`
        raw ranks):

            node    bm25_match   coverage   product
            rare      0.387127       0.25   0.096782
            full      0.174977       1.00   0.174977

        so bm25 ALONE ranks the one-word note first and coverage is the only
        thing that corrects it. Both node surfaces share `_node_relevance`,
        hence both are checked.
        """
        store, intel = await _stack(tmp_path, name=f"cov-{surface}.db")
        try:
            # postgres / connection / pool common, exhaustion rare
            for i in range(20):
                await intel.graph.add_node(
                    name=f"common {i}",
                    type="note",
                    description=f"postgres connection pool notes number {i} for the team",
                )
            await intel.graph.add_node(name="full", type="solution", description=LONG_FULL)
            await intel.graph.add_node(name="rare", type="note", description="exhaustion")

            returned = await _call(intel, surface, QUERY, limit=25)
            order = [r["name"] for r in returned if r["source"] == "node"]

            assert "full" in order and "rare" in order, order
            assert order.index("full") < order.index("rare"), (
                "a one-word note holding only the rarest term outranked the "
                f"document answering all four query terms: {order[:4]}"
            )
        finally:
            await store.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("surface", ["recall", "crossref"])
    @pytest.mark.parametrize("limit", [3, 5, 10])
    async def test_rerank_sees_more_than_limit_candidates(
        self, tmp_path, surface, limit
    ):
        """The store truncates in RAW bm25 order before the rerank runs.

        Added after the first version of this fix shipped a rerank that could
        only shuffle rows the raw order had already chosen. Measured then:

            limit=3   4-of-4 document ABSENT, three one-word notes fill it
            limit=5   ABSENT
            limit=10  leads, 0.101312 against 0.061231

        The earlier coverage test used limit=25 on a 22-node store, which is
        exactly why it could not fail. The parametrised small limits are the
        point of this test - do not raise them.
        """
        store, intel = await _stack(tmp_path, name=f"pool-{surface}-{limit}.db")
        try:
            for i in range(20):
                await intel.graph.add_node(
                    name=f"common {i}",
                    type="note",
                    description=f"postgres connection pool notes number {i} for the team",
                )
            await intel.graph.add_node(name="full", type="solution", description=LONG_FULL)
            for i in range(5):
                await intel.graph.add_node(
                    name=f"rare {i}", type="note", description="exhaustion"
                )

            returned = await _call(intel, surface, QUERY, limit=limit)
            names = [r["name"] for r in returned if r["source"] == "node"]

            assert "full" in names, (
                "the document answering all four query terms was truncated away "
                f"by raw bm25 before the rerank could see it: {names}"
            )
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_crossref_logs_only_the_nodes_it_surfaced(self, tmp_path):
        """The over-fetch must not leak into the activity log.

        `crossref` now pulls a rerank pool from the store, so logging the
        fetched set would credit ~30 nodes to a query that returned 3 and
        would corrupt whatever downstream analytics reads
        `activity_type='node_crossref'`. `recall` already logged only what it
        surfaced; this pins the same rule for crossref.
        """
        store, intel = await _stack(tmp_path)
        try:
            for i in range(20):
                await intel.graph.add_node(
                    name=f"common {i}",
                    type="note",
                    description=f"postgres connection pool notes number {i} for the team",
                )

            results = await intel.crossref(problem=QUERY, limit=3)
            surfaced = {r["id"] for r in results if r["source"] == "node"}

            log = await store.get_activity_log(entity_type="node", limit=200)
            logged = {e["entity_id"] for e in log if e["activity_type"] == "node_crossref"}

            assert logged, "sanity: crossref must log the nodes it surfaced"
            assert logged == surfaced, (
                f"logged {len(logged)} nodes but surfaced {len(surfaced)}"
            )
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_more_decimals_do_not_invert_the_order(self, tmp_path):
        """The reported number must not contradict the order it arrives in.
        A store big enough for bm25 to be informative, so this checks the
        resolution change and not the degenerate regime."""
        store, intel = await _stack(tmp_path)
        try:
            await _fill_noise(intel, 12)
            await intel.graph.add_node(name="strong", type="solution", description=STRONG)
            await intel.graph.add_node(name="weak", type="note", description=WEAK)

            results = await intel.recall(topic=QUERY, limit=20)
            scores = [r["relevance"] for r in results if r["source"] == "node"]

            assert scores == sorted(scores, reverse=True), scores
        finally:
            await store.close()
