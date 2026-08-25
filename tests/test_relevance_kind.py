"""Every `relevance` a caller sees must say which of four things it is.

THE GATE, verbatim from the plan this implements: "no caller receives a number
that claims to be match quality when it is recency."

WHAT THESE TESTS COVER, stated exactly rather than generously. An earlier
version of this docstring claimed a tenth unlabelled site "fails the suite by
default". That is false and was measured to be false: a reviewer added a
`relevance` key to `GraphEngine.get_related` and the suite stayed green. These
walk the payloads of three `IntelligenceLayer` methods plus three named
server/CLI surfaces. That is a checklist expressed as traversal, and a genuinely
automatic guard needs a shared serialiser or a route registry, which this change
does not build.

WHAT THEY DO ENFORCE, and this is the part that matters: not merely that a kind
is PRESENT, but that it is the RIGHT one for the row's source. Presence-only was
the first version's real weakness - relabelling every experience "match" would
have passed it, which is precisely the defect the gate names. `wrong_kinds()`
derives the expected kind from `source` and fails on a mismatch, so a
substitution mutant dies, not only a deletion mutant.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import pytest_asyncio
from fastmcp import Client

from kairn.core.experience import ExperienceEngine
from kairn.core.graph import GraphEngine
from kairn.core.ideas import IdeaEngine
from kairn.core.intelligence import IntelligenceLayer
from kairn.core.memory import ProjectMemory
from kairn.core.router import ContextRouter
from kairn.events.bus import EventBus
from kairn.models.experience import Experience
from kairn.relevance import (
    RELEVANCE_KIND_MATCH,
    RELEVANCE_KIND_RECENCY,
    RELEVANCE_KIND_SIMILARITY,
    RELEVANCE_KIND_UNSCORED,
    RELEVANCE_KINDS,
)
from kairn.server import create_server
from kairn.storage.sqlite_store import SQLiteStore


@pytest_asyncio.fixture
async def intel(tmp_path):
    db_path = tmp_path / "relevance_kind.db"
    store = SQLiteStore(db_path)
    await store.initialize()
    bus = EventBus()
    graph = GraphEngine(store, bus)
    experience = ExperienceEngine(store, bus)
    layer = IntelligenceLayer(
        store=store,
        event_bus=bus,
        graph=graph,
        router=ContextRouter(store, bus),
        memory=ProjectMemory(store, bus),
        experience=experience,
        ideas=IdeaEngine(store, bus),
    )
    # One node and one experience that both match the query text, so every
    # surface below returns BOTH kinds of row and the test is not silently
    # walking an empty list.
    await layer.learn(
        content="sqlite wal checkpoint starvation under concurrent writers",
        type="gotcha",
        confidence="high",
    )
    await experience.save(
        content="sqlite wal checkpoint starvation was traced to a long reader",
        type="solution",
    )
    yield layer


def offenders(payload) -> list[dict]:
    """Every dict anywhere in `payload` that has `relevance` and no valid kind.

    Walks the whole structure rather than a known key, because the surfaces put
    their rows under four different keys (`results`, `nodes`, `experiences`,
    and a bare list) and hard-coding those is how the fifth one is missed."""
    found: list[dict] = []

    def walk(o):
        if isinstance(o, dict):
            if "relevance" in o and o.get("relevance_kind") not in RELEVANCE_KINDS:
                found.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(payload)
    return found


EXPECTED_BY_SOURCE = {
    "experience": {RELEVANCE_KIND_RECENCY},
    # A node row can legitimately be any of three: bm25 on the keyword path,
    # cosine when semantic recall is on, and the literal 1.0 a text-less browse
    # or crossref produces. It can never be recency.
    "node": {RELEVANCE_KIND_MATCH, RELEVANCE_KIND_SIMILARITY, RELEVANCE_KIND_UNSCORED},
}


def wrong_kinds(payload, *, default_source=None) -> list[tuple]:
    """Rows whose `relevance_kind` contradicts where the row came from.

    This is the check that kills a SUBSTITUTION mutant. `offenders()` only sees
    a missing or unknown kind, so flipping every experience to "match" - the
    exact thing the gate forbids - passes it."""
    bad: list[tuple] = []

    def walk(o, src=default_source):
        if isinstance(o, dict):
            src = o.get("source", src)
            if "relevance" in o:
                allowed = EXPECTED_BY_SOURCE.get(src)
                if allowed is not None and o.get("relevance_kind") not in allowed:
                    bad.append((src, o.get("relevance_kind"), o.get("id")))
            for k, v in o.items():
                walk(v, "experience" if k == "experiences" else ("node" if k == "nodes" else src))
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v, src)

    walk(payload)
    return bad


def count_relevance(payload) -> int:
    n = 0

    def walk(o):
        nonlocal n
        if isinstance(o, dict):
            if "relevance" in o:
                n += 1
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(payload)
    return n


@pytest.mark.asyncio
async def test_every_relevance_carries_its_kind(intel):
    """The invariant, over the four surfaces a caller actually reaches."""
    q = "sqlite wal checkpoint starvation"
    payloads = {
        "recall": await intel.recall(topic=q, limit=10),
        "crossref": await intel.crossref(problem=q, limit=10),
        "context": await intel.context(keywords=q, limit=10),
    }
    for name, payload in payloads.items():
        seen = count_relevance(payload)
        # An empty surface would pass the invariant vacuously. Assert it saw
        # something first, so a fixture that stops matching turns into a red
        # test rather than a quiet green one.
        assert seen > 0, (
            f"{name} returned no rows carrying relevance - the fixture no "
            f"longer exercises this surface, so the check below is vacuous"
        )
        bad = offenders(payload)
        assert not bad, (
            f"{name}: {len(bad)} row(s) report `relevance` with no valid "
            f"`relevance_kind`: {bad[:2]}"
        )
        mism = wrong_kinds(payload)
        assert not mism, (
            f"{name}: {len(mism)} row(s) carry a kind that contradicts their "
            f"source (source, kind, id): {mism[:3]}"
        )


@pytest.mark.asyncio
async def test_experience_rows_are_labelled_recency_not_match(intel):
    """The specific claim the plan cares about: an experience's number is
    recency, and it says so."""
    payload = await intel.context(keywords="sqlite wal checkpoint starvation", limit=10)
    exps = payload.get("experiences") or []
    assert exps, "fixture produced no experiences"
    for e in exps:
        assert e["relevance_kind"] == RELEVANCE_KIND_RECENCY, (
            f"experience {e.get('id')} reports kind {e.get('relevance_kind')!r}; "
            f"its number comes from Experience.relevance(at), which is time-decay"
        )


@pytest.mark.asyncio
async def test_node_rows_are_labelled_match_or_unscored(intel):
    payload = await intel.recall(topic="sqlite wal checkpoint starvation", limit=10)
    # recall() returns a bare list, crossref() too - the shapes differ per
    # surface, which is exactly why the invariant test above walks the whole
    # structure instead of indexing a known key.
    rows = payload if isinstance(payload, list) else payload.get("results", [])
    nodes = [r for r in rows if r.get("source") == "node"]
    assert nodes, "fixture produced no nodes"
    for n in nodes:
        # MATCH only. The first version accepted UNSCORED too, which meant
        # flipping recall's bm25 nodes to "unscored" - a caller-visible
        # regression of exactly the kind this file exists to catch - left the
        # test green. This fixture always queries with text, so every node here
        # has a real rank.
        assert n["relevance_kind"] == RELEVANCE_KIND_MATCH, (
            f"recall node {n.get('id')} reports {n.get('relevance_kind')!r}; "
            f"it came from _bm25_to_relevance(rank) on a text query"
        )


@pytest.mark.asyncio
async def test_crossref_nodes_declare_their_constant_is_not_a_score(intel):
    """crossref hands every node the literal 1.0 and then sorts nodes and
    experiences together on that field. The sort is out of scope here; saying
    the constant is not a ranking is not."""
    payload = await intel.crossref(problem="sqlite wal checkpoint starvation", limit=10)
    rows = payload if isinstance(payload, list) else payload.get("results", [])
    nodes = [r for r in rows if r.get("source") == "node"]
    assert nodes, "fixture produced no crossref nodes"
    for n in nodes:
        assert n["relevance"] == 1.0
        assert n["relevance_kind"] == RELEVANCE_KIND_UNSCORED


def test_experience_to_response_carries_the_kind():
    e = Experience(content="anything at all", type="solution", decay_rate=0.05)
    d = e.to_response()
    assert "relevance" in d
    assert d["relevance_kind"] == RELEVANCE_KIND_RECENCY


def test_the_invariant_can_actually_fail():
    """MUTATION CONTROL. Construct what a forgotten site emits and assert the
    checker rejects it - and that it accepts the labelled version. Both
    directions, because a checker stuck at 'reject everything' is as useless as
    one stuck at 'accept everything'."""
    forgotten = {"results": [{"id": "x", "relevance": 0.98}]}
    assert offenders(forgotten), "the checker accepted an unlabelled relevance"

    misspelled = {"results": [{"id": "x", "relevance": 0.98, "relevance_kind": "recent"}]}
    assert offenders(misspelled), "the checker accepted a kind outside the constant set"

    labelled = {
        "results": [{"id": "x", "relevance": 0.98, "relevance_kind": RELEVANCE_KIND_RECENCY}]
    }
    assert not offenders(labelled), "the checker rejected a correctly labelled row"

    # And the vacuity guard itself: an empty payload has nothing to offend, so
    # count_relevance must report 0 and the surface test must not silently pass
    # on one.
    assert count_relevance({"results": []}) == 0


# ---------------------------------------------------------------------------
# THE THREE SURFACES A REAL CALLER ACTUALLY REACHES.
#
# The tests above walk IntelligenceLayer output and are blind to server.py and
# cli.py. That is not a hypothesis - a mutation run over all nine production
# sites SURVIVED on exactly three: server.py's kn_memories tool, its
# kn://memories resource, and the CLI's memories command. Nine sites labelled,
# six covered, and the uncovered three are the MCP tool the hook calls and the
# command a human runs. "A fix applied to N-1 of N sites looks exactly like a
# fix" - so these exist to make the remaining three fail when broken.
# ---------------------------------------------------------------------------

@pytest.fixture
async def mcp_client(tmp_path):
    server = create_server(str(tmp_path / "relkind.db"))
    async with Client(server) as c:
        await c.call_tool(
            "kn_save",
            {"content": "wal checkpoint starvation traced to a long reader",
             "type": "solution"},
        )
        yield c


async def test_kn_memories_tool_labels_its_relevance(mcp_client: Client):
    res = await mcp_client.call_tool("kn_memories", {"text": "checkpoint starvation"})
    data = json.loads(res.content[0].text)
    exps = data["experiences"]
    assert exps, "kn_memories returned nothing - this test would be vacuous"
    for e in exps:
        assert e.get("relevance_kind") == RELEVANCE_KIND_RECENCY, (
            f"kn_memories reports relevance {e.get('relevance')} with kind "
            f"{e.get('relevance_kind')!r}"
        )


async def test_kn_memories_resource_labels_its_relevance(mcp_client: Client):
    res = await mcp_client.read_resource("kn://memories")
    data = json.loads(res[0].text)
    exps = data["experiences"]
    assert exps, "kn://memories returned nothing - this test would be vacuous"
    for e in exps:
        assert e.get("relevance_kind") == RELEVANCE_KIND_RECENCY


async def test_min_relevance_parameter_says_it_filters_recency(mcp_client: Client):
    """THE MIRROR DEFECT. A field that reports recency under a match-shaped name
    is half the problem; a PARAMETER that accepts a threshold under the same
    name is the other half. A caller passing min_relevance=0.8 for "only good
    matches" gets "only recent rows, however badly they match"."""
    tools = {t.name: t for t in await mcp_client.list_tools()}
    for name in ("kn_memories", "kn_recall"):
        schema = tools[name].inputSchema
        desc = schema["properties"]["min_relevance"].get("description", "")
        assert "RECENCY" in desc or "recency" in desc, (
            f"{name}.min_relevance is described as {desc!r} - a caller cannot "
            f"tell it thresholds on time-decay rather than match quality"
        )


def test_cli_memories_labels_its_relevance(tmp_path):
    ws = tmp_path / "ws"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}

    def run(*args):
        return subprocess.run(
            [sys.executable, "-m", "kairn.cli", *args],
            capture_output=True, text=True, env=env,
        )

    assert run("init", str(ws)).returncode == 0
    # `learn --confidence medium` stores an EXPERIENCE; high stores a node.
    r0 = run("learn", str(ws), "--content",
             "wal checkpoint starvation traced to a long reader",
             "--type", "solution", "--confidence", "medium")
    assert r0.returncode == 0, r0.stderr
    r = run("memories", str(ws), "--text", "checkpoint starvation")
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    exps = data.get("experiences") or []
    assert exps, f"cli memories returned nothing - test would be vacuous: {r.stdout[:200]}"
    for e in exps:
        assert e.get("relevance_kind") == RELEVANCE_KIND_RECENCY


def test_wrong_kinds_can_actually_fail():
    """MUTATION CONTROL for the SOURCE-aware checker, both directions.

    The deletion sweep over the production sites proves a missing label dies.
    It says nothing about a WRONG label, which is the failure the gate actually
    names ("claims to be match quality when it is recency"). This proves the
    checker that catches that can itself fail."""
    lying = {"results": [{"source": "experience", "id": "x", "relevance": 0.98,
                          "relevance_kind": RELEVANCE_KIND_MATCH}]}
    assert wrong_kinds(lying), "an experience labelled 'match' was accepted"

    node_lying = {"results": [{"source": "node", "id": "x", "relevance": 0.6,
                               "relevance_kind": RELEVANCE_KIND_RECENCY}]}
    assert wrong_kinds(node_lying), "a node labelled 'recency' was accepted"

    honest = {"results": [
        {"source": "experience", "id": "a", "relevance": 0.98,
         "relevance_kind": RELEVANCE_KIND_RECENCY},
        {"source": "node", "id": "b", "relevance": 0.6,
         "relevance_kind": RELEVANCE_KIND_MATCH},
        {"source": "node", "id": "c", "relevance": 0.9,
         "relevance_kind": RELEVANCE_KIND_SIMILARITY},
        {"source": "node", "id": "d", "relevance": 1.0,
         "relevance_kind": RELEVANCE_KIND_UNSCORED},
    ]}
    assert not wrong_kinds(honest), "correctly labelled rows were rejected"

    # And it must reach rows nested under a keyed section, where `source` is
    # absent and only the container name says what they are.
    nested = {"experiences": [{"id": "x", "relevance": 0.9,
                               "relevance_kind": RELEVANCE_KIND_MATCH}]}
    assert wrong_kinds(nested), "the checker did not descend into `experiences`"


async def test_semantic_path_is_labelled_similarity_not_match(tmp_path):
    """The site with no coverage at all until now.

    `_node_result` is a funnel for three numbers and an earlier version
    hardcoded MATCH. The semantic path feeds it embedding cosine - a different
    scale, corpus-independent - and no fixture in this file turned semantic
    recall on, so that mislabel was invisible.

    The embedder goes on the STORE (that is where embed-at-write happens) as
    well as the layer; passing it only to GraphEngine writes no vectors and the
    semantic path then finds nothing to rerank."""

    def embed(texts: list[str]) -> list[list[float]]:
        # Every text gets the same vector, so cosine is 1.0 and every node
        # clears the floor. The VALUE is irrelevant here; the LABEL is the test.
        return [[1.0, 0.0, 0.0] for _ in texts]

    store = SQLiteStore(tmp_path / "sem.db", embedder=embed, embedder_model="fake-3")
    await store.initialize()
    bus = EventBus()
    layer = IntelligenceLayer(
        store=store,
        event_bus=bus,
        graph=GraphEngine(store, bus),
        router=ContextRouter(store, bus),
        memory=ProjectMemory(store, bus),
        experience=ExperienceEngine(store, bus),
        ideas=IdeaEngine(store, bus),
        embedder=embed,
        embedder_model="fake-3",
        semantic_recall=True,
        semantic_floor=0.25,
    )
    await layer.learn(
        content="wal checkpoint starvation under concurrent writers",
        type="gotcha",
        confidence="high",
    )
    payload = await layer.recall(topic="checkpoint starvation", limit=10)
    rows = payload if isinstance(payload, list) else payload.get("results", [])
    nodes = [r for r in rows if r.get("source") == "node"]
    assert nodes, "semantic recall returned no nodes - this test would be vacuous"
    for n in nodes:
        assert n["relevance_kind"] == RELEVANCE_KIND_SIMILARITY, (
            f"semantic node {n.get('id')} reports {n.get('relevance_kind')!r}; "
            f"its number is an embedding cosine, not a bm25 match score"
        )
    assert not wrong_kinds(payload)


# ---------------------------------------------------------------------------
# context() NODES. Until the router candidates were ranked, these carried a
# `confidence` and no `relevance` at all, so the invariant above skipped them
# entirely - it only fires on rows that HAVE a relevance. That is the shape of
# gap worth naming: an invariant over "every row with X" is silent about rows
# that lack X, and the missing rows were the whole defect.
# ---------------------------------------------------------------------------


def test_bm25_relevance_is_monotone_in_rank():
    """The ordering above only means something if the transform preserves bm25
    order. FTS5 bm25 is NEGATIVE and more-negative is a stronger match, so the
    mapping must be decreasing in `rank`. Checked across the magnitudes a real
    store actually produces, not toy values."""
    from kairn.core.intelligence import _bm25_to_relevance

    ranks = [-12.0, -7.5, -5.0, -3.0, -1.0, -1e-3, -1e-6]
    rels = [_bm25_to_relevance(r) for r in ranks]
    pairs = list(zip(ranks, rels, strict=True))
    assert rels == sorted(rels, reverse=True), f"not monotone: {pairs}"
    assert rels[0] > rels[-1], "the transform is flat across a 12-point bm25 range"
    assert _bm25_to_relevance(-5.0) == 0.5, "midpoint moved without this test noticing"
    assert _bm25_to_relevance(None) == 1.0, "the no-match browse case changed"


