"""The abstention floor: where it comes from, where it stops, and the state
between "matched" and "did not match".

Three findings from the review that REJECTED the first version of this feature
(Kairn `00f674bd`) are pinned here. Each test names the hop or the mechanism it
covers, so a future reader can tell what goes red and why.

F5 - THE ALL-STOP-WORD FLOOD. `recall("is it ok")` returned 5 unrelated
experiences at relevance 1.0 with the floor set to 0.65. The chain: the query
has no searchable keyword, so the intelligence layer shapes it to `None` and
calls `experience.search(text=None, ...)`; that is the same internal state as
"no text was given", so the engine browses; a browse row has `rank is None`;
`bm25_to_relevance(None)` returns 1.0 by a REPORTING contract ("there is no
match to report"); and a threshold then read that 1.0 as a perfect match and
passed every row. An UNKNOWN encoded as a value, which every later consumer
then has to special-case.

F8 - THE WIRING GAP THAT INDICTS THE ROUND. The review deleted the single line
`min_match=self.experience_min_match,` from `IntelligenceLayer.recall`, verified
the mutation applied (grep 1 -> 0), and 670 tests still passed. The mutation
testing covered the two ENGINE gates and never the feature's WIRING. The
`test_hop_*` tests below cover every hop the floor takes from the config file to
the gate, so removing any one of them turns a test red.

F2 - UNROUNDED FOR THE ORDER, ROUNDED FOR THE WIRE. `bm25_to_relevance` rounds
to 4 decimals for human/wire consumption. On a small store that round collapses
real matches to 0.0, at which point a sort by that value is insertion order
wearing a ranking's name, so ordering uses the unrounded `bm25_match`.

Every structural test in this file asserts NON-VACUITY first (it found at least
N sites) before asserting the property. An empty collection makes a universal
assertion trivially true, and that is the shape of a check that cannot fail.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from kairn.config import Config
from kairn.core.experience import ExperienceEngine
from kairn.core.fts import (
    blend_match_and_recency,
    bm25_match,
    bm25_to_relevance,
    fts_keywords,
    to_fts_query,
)
from kairn.core.graph import GraphEngine
from kairn.core.ideas import IdeaEngine
from kairn.core.intelligence import IntelligenceLayer
from kairn.core.memory import ProjectMemory
from kairn.core.router import ContextRouter
from kairn.events.bus import EventBus

SRC = Path(__file__).resolve().parents[1] / "src" / "kairn"

# Deliberately unrelated to every query used below.
UNRELATED = [
    ("Kubernetes helm chart values drift between clusters", "gotcha"),
    ("Rust borrow checker lifetimes on nested closures", "gotcha"),
    ("Postgres autovacuum starves on a hot partition", "solution"),
    ("Terraform state lock survives a killed apply", "workaround"),
    ("Redis keyspace notifications need a config flag", "pattern"),
]


@pytest.fixture
async def engine(store):
    return ExperienceEngine(store, EventBus())


@pytest.fixture
async def filled(engine):
    for content, exp_type in UNRELATED:
        await engine.save(content=content, type=exp_type)
    return engine


def _layer(store, experience, **kwargs) -> IntelligenceLayer:
    bus = EventBus()
    return IntelligenceLayer(
        store=store,
        event_bus=bus,
        graph=GraphEngine(store, bus),
        router=ContextRouter(store, bus),
        memory=ProjectMemory(store, bus),
        experience=experience,
        ideas=IdeaEngine(store, bus),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# F5 - the all-stop-word flood
# ---------------------------------------------------------------------------


class TestF5AllStopWordFlood:
    """WHERE THIS DEFECT LIVES, measured rather than argued.

    The engine is already right when it is handed the RAW question. The flood
    needs a caller that shapes the query FIRST and passes the shaped result,
    because `_to_fts_query(topic) if topic else None` maps BOTH "no topic" and
    "a topic with no searchable word" onto the same `None`. That expression is
    where the two meanings meet, and no value the engine returns can recover a
    distinction its caller already discarded.

    Measured on a scratch copy, `recall()` experiences returned, 5 unrelated
    rows in the store:

        intelligence.recall passes    floor 0.0            floor 0.65
        ---------------------------   ------------------   ------------------
        text=fts_query (today)        flood 5, browse 5    flood 5, browse 5
        text=topic     (the fix)      flood 0, browse 5    flood 0, browse 5

    Note the first column. At the DEFAULT floor the flood is wide open, so no
    change to the gate could ever have closed it - an abstention floor nobody
    has switched on cannot filter anything. Gating the browse instead was
    tried here first and refuted by three external lenses independently: it
    stops the flood only for operators who set a floor, and it does so by
    disabling topicless browse for exactly those operators.
    """

    @pytest.mark.asyncio
    async def test_the_reviews_exact_reproduction(self, store, filled):
        """`recall("is it ok")` at floor 0.65 returned 5 unrelated experiences
        at relevance 1.0. This is that call, unchanged.

        Written RED against `text=fts_query` and turned green by the caller
        fix (`text=topic`), which landed after this test did. It is written to
        the contract, not to the state of the day."""
        intel = _layer(store, filled, experience_min_match=0.65)
        results = await intel.recall(topic="is it ok", limit=10)
        experiences = [r for r in results if r["source"] == "experience"]
        assert experiences == [], [
            (r["relevance"], r["content"][:40]) for r in experiences
        ]

    @pytest.mark.asyncio
    async def test_the_flood_is_open_at_the_default_floor_too(self, store, filled):
        """The half a gate-side fix cannot reach. Nobody has to opt in to
        anything for an all-stop-word question to be answered with five
        confident-looking unrelated rows."""
        intel = _layer(store, filled)          # floor at its 0.0 default
        results = await intel.recall(topic="is it ok", limit=10)
        assert [r for r in results if r["source"] == "experience"] == []

    @pytest.mark.asyncio
    async def test_the_engine_is_right_when_handed_the_raw_question(self, filled):
        """The engine's own half, and it already passes: given the words
        rather than a pre-shaped query it finds no keyword and abstains before
        any gate runs - at every floor, including none at all."""
        assert await filled.search(text="is it ok", limit=10) == []
        assert await filled.search(text="is it ok", min_match=0.65, limit=10) == []

    @pytest.mark.asyncio
    async def test_a_browse_is_not_subject_to_the_floor(self, filled):
        """THE REGRESSION CONTROL, and the reason the gate does not abstain
        here. `min_match` filters on match strength; a browse row has none,
        which makes the floor INAPPLICABLE to it rather than failed. An
        operator with a floor set who asks for recent experiences must still
        get them."""
        rows = await filled.search(text=None, min_match=0.65, limit=10)
        assert len(rows) == len(UNRELATED)

    @pytest.mark.asyncio
    async def test_the_bitemporal_gate_has_the_same_rule(self, filled):
        """Both gates or neither (Kairn `d5272a91`): a rule applied to one of
        two gates looks exactly like a rule."""
        rows = await filled.search_bitemporal(text=None, min_match=0.65, limit=10)
        assert len(rows) == len(UNRELATED)

    @pytest.mark.asyncio
    async def test_positive_control_a_browse_without_a_floor_is_unchanged(self, filled):
        """With the floor at its default the browse returns every row.

        Asserted as a SET, not a sequence, and that is a measurement rather
        than a convenience: two identical browse calls agreed on the order in
        only 4 of 40 trials, on the PRISTINE file as well as this one (40/40
        agreed on the set). The browse sort key is a float decay recomputed
        against a fresh `now`, and rows written microseconds apart differ only
        in the last digits of it. Pre-existing, out of scope here, and
        recorded so the next reader does not read this relaxation as slack."""
        rows = await filled.search(text=None, min_match=0.0, limit=10)
        assert len(rows) == len(UNRELATED)
        again = await filled.search(text=None, limit=10)
        assert {e.id for e in again} == {e.id for e in rows}
        assert len(again) == len(rows)

    @pytest.mark.asyncio
    async def test_positive_control_a_real_match_still_clears_a_modest_floor(self, filled):
        """The other side of the control: rows with real match strength still
        pass, so the gate is not simply letting everything through."""
        rows = await filled.search(text="kubernetes helm chart", min_match=0.05, limit=10)
        assert len(rows) == 1
        assert rows[0].content.startswith("Kubernetes helm")

    def test_the_1_0_stays_a_reporting_contract(self):
        """`bm25_to_relevance(None) == 1.0` is not wrong - it is the wire
        answer to "how well did this match" when nothing was matched. What was
        wrong was feeding it to a THRESHOLD."""
        assert bm25_to_relevance(None) == 1.0
        assert bm25_match(None) == 1.0


# ---------------------------------------------------------------------------
# F8 - every hop from the config file to the gate
# ---------------------------------------------------------------------------


def _calls_named(tree: ast.AST, name: str) -> list[ast.Call]:
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Name) and func.id == name) or (
            isinstance(func, ast.Attribute) and func.attr == name
        ):
            out.append(node)
    return out


def _kwarg_names(call: ast.Call) -> set[str]:
    return {kw.arg for kw in call.keywords if kw.arg}


def _reads_config(scope: ast.AST) -> bool:
    """Does this scope read a `Config` at all?

    Deliberately scope-level and not kwarg-level. The kwarg-level version of
    this check (`experience_min_match=config.experience_min_match`) broke the
    moment the value was routed through a range-validating local, which is a
    refactor that IMPROVES the code - a detector that a good refactor turns
    off is a detector that reports a clean run for the wrong reason."""
    return any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "config"
        for node in ast.walk(scope)
    )


def _enclosing_scopes(tree: ast.AST) -> list[ast.AST]:
    return [
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.Module)
    ]


class TestF8FloorWiring:
    def test_hop_1_the_config_carries_the_floor(self, tmp_path):
        """The field exists, defaults to OFF, and survives a save/load round
        trip - a config key that silently reverts is not wired."""
        cfg = Config(workspace_path=tmp_path)
        assert cfg.experience_min_match == 0.0
        cfg.experience_min_match = 0.65
        cfg.save()
        assert Config.load(tmp_path).experience_min_match == 0.65

    def test_hop_2_every_construction_site_forwards_the_floor(self):
        """EVERY site, with no exemption clause.

        Two earlier versions tried to exempt sites that "do not read a
        config", and a review measured both of them wrong IN BOTH DIRECTIONS.
        The kwarg-level detector (`x=config.attr`) switched itself off the
        moment the value was routed through a range-validating local - a
        refactor that improves the code turning the detector off. The
        scope-level replacement collapsed to file-level, because `ast.walk` on
        the Module node reaches every call, so `owners` always contains the
        Module and the predicate degenerated to "does this FILE mention
        config anywhere"; it also could not see `self.config.x`, the shape any
        long-lived server object converges on.

        A classifier that is wrong in both directions is worse than no
        classifier. Whoever constructs this layer makes a deliberate decision
        about the floor, and passing the default explicitly is the cheap way
        to record it."""
        sites: list[tuple[str, set[str]]] = []
        for path in SRC.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for call in _calls_named(tree, "IntelligenceLayer"):
                sites.append((f"{path.name}:{call.lineno}", _kwarg_names(call)))

        assert len(sites) >= 3, f"expected serve, demo and the server; found {sites}"
        missing = [where for where, kws in sites if "experience_min_match" not in kws]
        assert not missing, f"construction sites that drop the floor: {missing}"

    def test_hop_3_the_layer_stores_the_floor(self, store):
        engine = ExperienceEngine(store, EventBus())
        assert _layer(store, engine).experience_min_match == 0.0
        assert _layer(store, engine, experience_min_match=0.65).experience_min_match == 0.65

    @pytest.mark.asyncio
    async def test_hop_4_recall_forwards_the_floor_to_the_engine(self, store, filled):
        """THE LINE THE REVIEW DELETED. `min_match=self.experience_min_match,`
        in `IntelligenceLayer.recall`. Removing it left 670 tests green; this
        one goes red."""
        topic = "kubernetes helm chart"
        wide = await _layer(store, filled).recall(topic=topic, limit=10)
        assert [r for r in wide if r["source"] == "experience"], (
            "sanity: the topic must retrieve an experience with the floor off"
        )
        gated = await _layer(store, filled, experience_min_match=0.99).recall(
            topic=topic, limit=10
        )
        assert [r for r in gated if r["source"] == "experience"] == []

    def test_hop_4b_every_experience_search_site_forwards_the_floor_and_raw_text(self):
        """N of N across `src/`, and it checks WHAT each site passes as text.

        Two things go wrong here and only one of them is the floor.

        The FLOOR half: a site that omits `min_match` does not honour the
        operator's setting. Scoping this scan to `intelligence.py` and to the
        literal shape `self.experience.search(...)` missed `cli.py` (`kairn
        memories`) and `server.py` (`kn_memories`, written as
        `s["experience"].search(...)`, a Subscript receiver) - 3 of 5. Scope
        to the CLASS, then the file.

        The TEXT half: passing a PRE-SHAPED query re-opens the all-stop-word
        flood, because `to_fts_query` maps both "no topic" and "a topic with
        no searchable word" onto None and the engine reads None as "browse".
        Requiring only `min_match` would go green again the moment someone
        tidied the raw text back into the shaped query, which is the shape
        that let the original wiring line be deleted with 670 tests passing.

        has / needs / exempt, at the time of writing:
          HAS   intelligence.py recall / crossref / context - text=topic,
                text=problem, text=keywords, min_match forwarded
          HAS   cli.py `memories`, server.py `kn_memories` - text=text
          EXEMPT server.py x3 (consolidate / stats / promote paths) - they
                pass NO text at all, so there is no match question to floor.
                Asserted below as its own count, not left implicit.
          NOT EXEMPT, corrected - the `fts_query` consumers on the NODE
                path (graph.query, graph.query_ranked,
                query_nodes_with_embeddings, _keyword_node_recall). This
                docstring used to exempt them, claiming "nodes have no
                browse-means-everything failure mode". That was asserted
                with no test behind it and it is FALSE: measured,
                `recall("is it ok")` returns 5 unrelated NODES at relevance
                1.0. See TestF5TheNodeHalfOfTheSameFlood, which reproduces it
                and carries the fix location.
                What IS true is narrower, and it is why this scan cannot just
                demand raw text everywhere: `query_ranked` forwards `text`
                straight to `nodes_fts MATCH ?` and shapes nothing, so the
                node path really does need the pre-shaped query. Its defect is
                the same CONFLATION one level up, not the same one-word fix,
                so it is checked by behaviour over there rather than by this
                AST scan. Listed here so the next reader meets the correction
                instead of the claim."""
        text_sites: list[tuple[str, str]] = []
        browse_sites: list[str] = []
        for path in sorted(SRC.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("search", "search_bitemporal")):
                    continue
                receiver = ast.unparse(node.func.value)
                if "experience" not in receiver:
                    continue
                where = f"{path.name}:{node.lineno}"
                kwargs = {k.arg: ast.unparse(k.value) for k in node.keywords if k.arg}
                if "text" not in kwargs:
                    browse_sites.append(where)
                    continue
                text_sites.append((where, kwargs.get("text", "")))
                assert "min_match" in kwargs, f"{where} drops the floor"

        assert len(text_sites) >= 5, f"expected 5 text-passing sites, found {text_sites}"
        assert len(browse_sites) >= 3, (
            f"expected the no-text sites to still exist; found {browse_sites}"
        )
        preshaped = [w for w, t in text_sites if "fts" in t.lower()]
        assert not preshaped, (
            f"sites passing a PRE-SHAPED query, which re-opens the flood: {preshaped}"
        )

    async def test_hop_4c_all_three_surfaces_honour_the_floor(self, store, filled):
        """The same N-of-N one level down, in BEHAVIOUR rather than syntax.

        The AST check above proves the keyword is present at each site. It
        cannot prove the value that arrives is the configured one - a site
        forwarding a hard-coded 0.0 would satisfy it. These three calls drive
        the floor through each surface for real. Reproduced in the review at
        floor 0.65: recall 0 experiences, crossref and context 2 each."""
        topic = "kubernetes helm chart"
        wide = _layer(store, filled)
        gated = _layer(store, filled, experience_min_match=0.99)

        assert [r for r in await wide.recall(topic=topic) if r["source"] == "experience"]
        assert [r for r in await wide.crossref(problem=topic) if r["source"] == "experience"]
        assert (await wide.context(keywords=topic))["experiences"]

        leaks = {
            "recall": [r for r in await gated.recall(topic=topic) if r["source"] == "experience"],
            "crossref": [
                r for r in await gated.crossref(problem=topic) if r["source"] == "experience"
            ],
            "context": (await gated.context(keywords=topic))["experiences"],
        }
        assert not any(leaks.values()), {k: len(v) for k, v in leaks.items()}

    @pytest.mark.asyncio
    async def test_hop_5_search_applies_the_floor(self, filled):
        assert await filled.search(text="kubernetes helm", limit=5)
        assert await filled.search(text="kubernetes helm", min_match=0.99, limit=5) == []

    @pytest.mark.asyncio
    async def test_hop_6_search_bitemporal_applies_the_floor(self, filled):
        assert await filled.search_bitemporal(text="kubernetes helm", limit=5)
        assert (
            await filled.search_bitemporal(text="kubernetes helm", min_match=0.99, limit=5)
            == []
        )


# ---------------------------------------------------------------------------
# F2 - unrounded for the order, rounded for the wire
# ---------------------------------------------------------------------------


class TestF2UnroundedOrdering:
    def test_the_rounded_transform_is_the_wire_contract(self):
        """The two functions are the same transform, and the only difference
        is the 4-decimal round that `bm25_to_relevance` owes the wire."""
        for rank in (-0.5, -3.0, -12.0, -1e-5, None):
            assert bm25_to_relevance(rank) == round(bm25_match(rank), 4)

    def test_the_round_really_does_collapse_a_weak_match(self):
        """The premise the ordering fix rests on, asserted rather than
        assumed: at this magnitude the wire value is 0.0 and the ordering
        value is not."""
        rank = -1e-5
        assert bm25_to_relevance(rank) == 0.0
        assert bm25_match(rank) > 0.0

    def test_the_match_quantity_is_unrounded(self):
        """Route it through the rounded wire transform and this is 0.0."""
        data = {"rank": -1e-5, "content": "alpha beta", "context": None}
        assert ExperienceEngine._match_strength(data, ["alpha", "beta"]) > 0.0

    def test_two_rows_the_round_would_tie_still_order(self):
        """Both ranks round to 0.0 on the wire, so a rounded key makes these
        indistinguishable and the order becomes whatever order they arrived
        in. The unrounded quantity still separates them."""
        strong = {"rank": -4e-5, "content": "alpha beta", "context": None}
        weak = {"rank": -1e-5, "content": "alpha beta", "context": None}
        assert bm25_to_relevance(strong["rank"]) == bm25_to_relevance(weak["rank"]) == 0.0
        terms = ["alpha", "beta"]
        assert ExperienceEngine._match_strength(
            strong, terms
        ) > ExperienceEngine._match_strength(weak, terms)

    def test_the_gate_and_the_sort_read_the_same_quantity(self):
        """One number, three consumers: the sort blends it, the caller is
        shown the blend, the floor compares against it.

        The gate used to read RAW bm25 while the sort and the report read
        bm25 * coverage, so a partial match cleared a floor it was then
        reported below."""
        data = {"rank": -5.0, "content": "alpha", "context": None}
        terms = ["alpha", "beta", "gamma", "delta"]
        strength = ExperienceEngine._match_strength(data, terms)
        assert strength == pytest.approx(bm25_match(-5.0) * 0.25)
        assert blend_match_and_recency(match=strength, decay=1.0) == pytest.approx(
            strength * 1.1
        )

    def test_the_gate_can_pass_a_row_the_report_shows_below_the_floor(self):
        """THE UNSAFE DIRECTION, which the test above never touched.

        The test above asserts decay=1.0, where the blend is 1.1x the strength
        and therefore always ABOVE the floor a row just cleared. An external
        review pointed out that the other end of the band was untested: at
        decay near 0 the blend is 0.9x, so a row can clear `min_match` on raw
        strength and be REPORTED just under that same number.

        This is documented behaviour, not a defect - flooring the blend would
        make abstention depend on age, which is the decay-bucket gate this
        work removed (Kairn `cec86cb9`). It is pinned here so nobody
        re-derives it as a bug, and so the 10% bound cannot widen unnoticed.
        """
        strength = 0.20
        min_match = 0.19
        assert strength >= min_match, "precondition: the gate lets this row through"
        reported_old = blend_match_and_recency(match=strength, decay=0.0)
        assert reported_old < min_match, reported_old
        assert reported_old == pytest.approx(strength * 0.9)

        # The bound: the report can never be more than 10% under the strength,
        # so a floor is never off by more than that.
        assert reported_old >= strength * 0.9 - 1e-12

        # Positive control on the other end, so this test cannot be satisfied
        # by a blend that simply always returns something small.
        reported_fresh = blend_match_and_recency(match=strength, decay=1.0)
        assert reported_fresh > min_match
        assert reported_fresh == pytest.approx(strength * 1.1)

    def test_no_path_in_the_engine_consumes_the_rounded_value(self):
        """The inventory, as a check. `experience.py` ORDERS and GATES; both
        need match strength, neither needs the wire's rounding. The rounded
        transform belongs to the reporting layer (`intelligence`, `server`,
        `cli`), and the engine must not reach for it - that is how a rounded
        value got into a threshold in the first place."""
        path = SRC / "core" / "experience.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        # Names the module actually CONSUMES. A docstring may still discuss
        # the rounded transform, and should - the comment is why the engine
        # does not use it - so this reads the AST, not the text.
        #
        # Collect the ORIGINAL import name as well as the alias, and every
        # attribute access: `bm25_to_relevance as _wire` and
        # `fts.bm25_to_relevance(x)` are both ways to reach it that an
        # asname-only, Name-only check could not see. A review reproduced the
        # first one and it survived 786/786.
        used: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                used.add(node.attr)
            elif isinstance(node, ast.alias):
                used.add(node.name)
                if node.asname:
                    used.add(node.asname)
        assert "bm25_match" in used, "sanity: the engine must use the unrounded transform"
        assert "bm25_to_relevance" not in used, (
            "experience.py still consumes the rounded wire transform"
        )


# ---------------------------------------------------------------------------
# The floor and a browse: a DESIGN CHOICE, written down so nobody closes it
# ---------------------------------------------------------------------------


class TestTopiclessBrowseAndTheFloor:
    """WHAT SHIPS: a topicless browse is NOT filtered by a positive floor.

    Measured, `search(text=None, min_match=f)` on a 3-row store:
    f=0.0 -> 3 rows, f=0.65 -> 3 rows, f=0.99 -> 3 rows.

    THE DISAGREEMENT, because this is a choice and not an obvious truth.
    Two external lens rounds were run on it and they split the opposite way
    from each other, which is exactly why it is pinned here rather than left
    implicit in a `strength is not None`.

      Round 1, on "a positive floor REJECTS every browse row" - 3 of 3
      refuted it. fugu: "browse starvation caused by unresolved query-intent
      conflation ... min_match is inapplicable, not failed". deepseek: "a
      browse row is not an abstention from matching, it was never a match
      query at all". grok: "Not SOUND".

      Round 2, on the same question asked the other way round - 2 of 3 held
      that rejecting is correct ("the caller requested a positive textual
      match threshold without supplying text to match"), 1 refuted.

    Across both rounds 4 of 6 lens-runs say the floor must not apply to a
    browse, and that is what ships. The losing reading is real and is stated
    here so a future reader meets the argument instead of re-deriving it: an
    operator who sets a match floor arguably means "never show me anything
    unmatched", and under this behaviour a topicless browse still shows them
    everything.

    VETOABLE. The alternative needs a sentinel from the caller separating
    "browse by choice" from "a question that shaped to nothing" - the engine
    cannot tell them apart, because `_to_fts_query(topic) if topic else None`
    destroys the difference before the call. That is more machinery than this
    case has earned so far, and the all-stop-word flood it would have caught
    is already closed one layer up by passing the raw topic.
    """

    @pytest.mark.asyncio
    async def test_a_topicless_browse_survives_a_positive_floor_on_purpose(self, filled):
        """DELIBERATE. Not an oversight, not a gap - see the class docstring
        for the two readings and which one ships."""
        for floor in (0.05, 0.65, 0.99):
            rows = await filled.search(text=None, min_match=floor, limit=10)
            assert len(rows) == len(UNRELATED), f"floor {floor} filtered a browse"

    @pytest.mark.asyncio
    async def test_positive_control_at_the_default_the_gate_never_fires(self, filled):
        """The line that bounds the blast radius. At the shipped default of
        0.0 the `if min_match` guard is false, so no operator who has not
        opted in can be affected by any of this."""
        assert await filled.search(text=None, min_match=0.0, limit=10)
        assert await filled.search(text="kubernetes helm", min_match=0.0, limit=10)

    @pytest.mark.asyncio
    async def test_a_real_query_is_still_floored(self, filled):
        """The other side of the control: the floor is not simply inert."""
        assert await filled.search(text="kubernetes helm", min_match=0.99, limit=10) == []


class TestTheFloorDiscriminates:
    """A floor set between two real matches, which no other test here does.

    Every other floor in this file is at an extreme - 0.0 and 0.05 pass
    everything, 0.99 rejects everything measurable - so a mutant that gated
    on the wrong QUANTITY satisfied all of them. This corpus is large enough
    for bm25 to separate rows, and the floor sits in the gap.
    """

    @pytest.mark.asyncio
    async def test_a_partial_match_is_dropped_by_a_floor_raw_bm25_would_clear(
        self, engine
    ):
        query = "zebra quantum helical thermodynamics"
        for i in range(60):
            await engine.save(
                content=f"filler note number {i} about widget{i} and gadget{i}",
                type="gotcha",
            )
        await engine.save(content="a zebra crossing on the high street", type="gotcha")
        full = await engine.save(
            content="zebra quantum helical thermodynamics in one note", type="solution"
        )

        terms = fts_keywords(query)
        rows = await engine.store.query_experiences(
            text=to_fts_query(query), limit=100, offset=0
        )
        by_content = {r["content"][:12]: r for r in rows}
        weak = by_content["a zebra cros"]
        raw = bm25_match(weak.get("rank"))
        strength = ExperienceEngine._match_strength(weak, terms)

        # The fixture must actually straddle the floor, or this proves nothing.
        floor = 0.3
        assert strength < floor < raw, (
            f"fixture no longer discriminates: strength={strength} floor={floor} raw={raw}"
        )

        kept = await engine.search(text=query, min_match=floor, limit=10)
        assert [e.id for e in kept] == [full.id], [
            (e.content[:40], e.recall_relevance) for e in kept
        ]


# ---------------------------------------------------------------------------
# F5, the half this file could not see
# ---------------------------------------------------------------------------


@pytest.fixture
async def filled_nodes(store):
    """Nodes as well as experiences. NOTHING in this file created a node
    before this fixture existed, which is the whole reason the class below
    had to be written."""
    bus = EventBus()
    graph = GraphEngine(store, bus)
    for content, _ in UNRELATED:
        await graph.add_node(name=content[:28], type="concept", description=content)
    return graph


class TestF5TheNodeHalfOfTheSameFlood:
    """The SAME all-stop-word flood, on the other half of the SAME call.

    Every test in `TestF5AllStopWordFlood` filters to
    `r["source"] == "experience"`, and until this fixture landed no test in
    this file created a single node. So the class named after the flood went
    green while the flood reproduced, in the same `recall()`, on the half
    nobody looked at. Measured on the tree that made this file green:

        recall("is it ok")      ->  0 experiences,  5 NODES at relevance 1.0
        recall("kubernetes helm") ->  0 experiences,  1 node  at 0.372691

    This file did not merely miss it. It ASSERTED the exemption:
    `test_hop_4b`'s docstring claimed the node path was exempt because
    "nodes have no browse-means-everything failure mode". Written with no
    test behind it, and false. An exemption asserted without a test is a
    blind spot wearing a docstring.

    WHERE IT IS FIXED, since it is not the same one-word change as the
    experience half. `GraphEngine.query_ranked` forwards `text` straight to
    `nodes_fts MATCH ?` and shapes nothing, so the node path genuinely NEEDS
    the pre-shaped query - passing the raw topic there would break it. The
    defect is the conflation one level up: `recall` computes
    `fts_query = _to_fts_query(topic) if topic else None`, and
    `_keyword_node_recall` reads that single None as "browse". The state it
    needs is `topic and not fts_query` - a question was asked and NOTHING
    searchable survived - which must return [], not everything. The reporting
    half compounds it: `_node_relevance` with no terms returns
    `bm25_to_relevance(None) == 1.0`, the same reporting-contract-as-threshold
    defect that `ExperienceEngine._match_strength` exists to keep out of the
    gate, living one file over.
    """

    @pytest.mark.asyncio
    async def test_an_all_stop_word_question_floods_the_node_half(
        self, store, filled, filled_nodes
    ):
        """RED until `recall` separates "no topic" from "a topic that shaped
        to nothing" on the NODE path. Written to the contract."""
        layer = _layer(store, filled, experience_min_match=0.65)
        results = await layer.recall(topic="is it ok", limit=10)
        nodes = [r for r in results if r["source"] != "experience"]
        assert nodes == [], (
            "an unanswerable question was answered with the most recent nodes: "
            f"{[(n['name'], n['relevance']) for n in nodes]}"
        )

    @pytest.mark.asyncio
    async def test_the_flood_is_open_at_the_default_floor_here_too(
        self, store, filled, filled_nodes
    ):
        """No abstention setting can close this one either: it reproduces at
        the shipped default, where every floor is off."""
        layer = _layer(store, filled)
        results = await layer.recall(topic="is it ok", limit=10)
        assert [r for r in results if r["source"] != "experience"] == []

    @pytest.mark.asyncio
    async def test_positive_control_a_topicless_browse_still_returns_nodes(
        self, store, filled, filled_nodes
    ):
        """The line that keeps the fix from being "return nothing". A browse
        with NO topic is a different state and must still work."""
        layer = _layer(store, filled)
        results = await layer.recall(topic=None, limit=10)
        assert [r for r in results if r["source"] != "experience"]

    @pytest.mark.asyncio
    async def test_positive_control_a_real_question_still_matches_nodes(
        self, store, filled, filled_nodes
    ):
        """And the other side: a question with a searchable word still finds
        its node, so a green run here is not the empty-collection trick."""
        layer = _layer(store, filled)
        results = await layer.recall(topic="kubernetes helm", limit=10)
        assert [r for r in results if r["source"] != "experience"]
