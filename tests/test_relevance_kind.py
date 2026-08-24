"""Every `relevance` a caller sees must say which of three things it is.

THE GATE, verbatim from the plan this implements: "no caller receives a number
that claims to be match quality when it is recency."

The tests below are written as an INVARIANT over real surface output rather than
as a checklist of the nine known sites, because a checklist passes forever while
a tenth site is added unlabelled. `test_every_relevance_carries_its_kind` walks
the actual dicts each surface returns and fails on any that carries `relevance`
without `relevance_kind` - so a new serialisation site fails the suite by
default, which is the direction the failure should point.

MUTATION CONTROL for this file lives in `test_the_invariant_can_actually_fail`:
it constructs the exact shape a forgotten site produces and asserts the checker
rejects it. Without that, an invariant that walks zero dicts, or a checker with
an inverted condition, passes silently - and a check that cannot fail is not a
check.
"""

from __future__ import annotations

import os
import pytest
import pytest_asyncio

from kairn.core.experience import ExperienceEngine
from kairn.core.graph import GraphEngine
from kairn.core.ideas import IdeaEngine
from kairn.core.intelligence import IntelligenceLayer
from kairn.core.memory import ProjectMemory
from kairn.core.relevance import (
    RELEVANCE_KIND_MATCH,
    RELEVANCE_KIND_RECENCY,
    RELEVANCE_KIND_UNSCORED,
    RELEVANCE_KINDS,
)
from kairn.core.router import ContextRouter
from kairn.events.bus import EventBus
from kairn.models.experience import Experience
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
            if "relevance" in o:
                if o.get("relevance_kind") not in RELEVANCE_KINDS:
                    found.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(payload)
    return found


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
        assert seen > 0, f"{name} returned no rows carrying relevance - the "
        f"fixture no longer exercises this surface, so the invariant below is "
        f"vacuous"
        bad = offenders(payload)
        assert not bad, (
            f"{name}: {len(bad)} row(s) report `relevance` with no valid "
            f"`relevance_kind`: {bad[:2]}"
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
        assert n["relevance_kind"] in {RELEVANCE_KIND_MATCH, RELEVANCE_KIND_UNSCORED}


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

import json
import subprocess
import sys

from fastmcp import Client

from kairn.server import create_server


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
