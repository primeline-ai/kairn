"""Tests for the Phase 2 candidates[] field in kn_learn response.

Plan: `.claude/plans/2026-05-26-kairn-judgment-envelope-and-doctor.md`

Spec recap:
- `learn()` runs an FTS5 scan over existing nodes after persisting the
  new save, using the content as seed query. Returns up to 5 candidates
  as `{id, name, type, snippet, sim_rank}` so the caller can decide
  whether to invoke `kn_judge` (Phase 3) to record a relationship verb.
- `with_candidates=False` (CLI `--no-candidates`) skips the scan and
  omits the `candidates` key entirely (None semantics, not empty list).
- The just-created high-confidence node is excluded from candidates.
- Empty `candidates: []` when scan ran but produced no FTS5 matches
  (distinguishable from omitted key for callers that care).
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from kairn.core.experience import ExperienceEngine
from kairn.core.graph import GraphEngine
from kairn.core.ideas import IdeaEngine
from kairn.core.intelligence import IntelligenceLayer, _CANDIDATES_LIMIT
from kairn.core.memory import ProjectMemory
from kairn.core.router import ContextRouter
from kairn.events.bus import EventBus
from kairn.storage.sqlite_store import SQLiteStore


@pytest_asyncio.fixture
async def engine(tmp_path):
    """Full intelligence stack wired to a temp database."""
    db_path = tmp_path / "intel_test.db"
    store = SQLiteStore(db_path)
    await store.initialize()
    bus = EventBus()

    graph = GraphEngine(store, bus)
    router = ContextRouter(store, bus)
    memory = ProjectMemory(store, bus)
    experience = ExperienceEngine(store, bus)
    ideas = IdeaEngine(store, bus)

    intel = IntelligenceLayer(
        store=store,
        event_bus=bus,
        graph=graph,
        router=router,
        memory=memory,
        experience=experience,
        ideas=ideas,
    )
    yield intel
    await store.close()


@pytest.mark.asyncio
async def test_candidates_returned_when_matches_exist(
    engine: IntelligenceLayer,
) -> None:
    """A save that overlaps with an existing node returns it in candidates."""
    # Seed: pre-existing node about Redis caching.
    await engine.learn(
        content="Redis caching strategy for distributed systems",
        type="pattern",
        confidence="high",
    )

    # New save with overlapping keywords. Should surface the seed.
    result = await engine.learn(
        content="Redis cluster failover playbook for distributed cache",
        type="solution",
        confidence="high",
    )

    assert "candidates" in result
    assert isinstance(result["candidates"], list)
    assert len(result["candidates"]) >= 1
    first = result["candidates"][0]
    assert set(first.keys()) == {"id", "name", "type", "snippet", "sim_rank"}
    assert "Redis" in first["name"] or "Redis" in first["snippet"]


@pytest.mark.asyncio
async def test_candidates_empty_when_no_matches(
    engine: IntelligenceLayer,
) -> None:
    """FTS5 ran against a non-empty DB but found no overlap → empty list, not missing key.

    Seeds a node that shares no keywords with the new save so the test
    actually exercises the "scan ran, found nothing" code path (and not
    the trivial "DB is empty so of course no candidates" path).
    """
    await engine.learn(
        content="Kubernetes pod scheduling with node affinity rules",
        type="pattern",
        confidence="high",
    )
    result = await engine.learn(
        content="Underwater basket weaving techniques",
        type="solution",
        confidence="high",
    )
    assert "candidates" in result
    assert result["candidates"] == []


@pytest.mark.asyncio
async def test_candidates_respect_namespace_isolation(
    engine: IntelligenceLayer,
) -> None:
    """A save in namespace A must not return candidates from namespace B.

    Load-bearing for Phase 3: kn_judge will consume candidates and
    create edges. If isolation breaks, cross-namespace edges sneak in
    and violate the workspace boundary. Must be empirically verified
    end-to-end through `_scan_candidates`, not just at the storage layer.
    """
    await engine.learn(
        content="Redis caching strategy for distributed systems",
        type="pattern",
        confidence="high",
        namespace="evolving",
    )
    result = await engine.learn(
        content="Redis cluster failover for distributed cache",
        type="solution",
        confidence="high",
        namespace="revane",
    )
    assert result["candidates"] == []


@pytest.mark.asyncio
async def test_candidates_excludes_just_created_node(
    engine: IntelligenceLayer,
) -> None:
    """The newly-created high-confidence node must not appear in its own candidates."""
    # Seed with a node that will match the new save (so FTS5 has SOMETHING to return).
    await engine.learn(
        content="Postgres connection pooling for high-throughput services",
        type="pattern",
        confidence="high",
    )
    result = await engine.learn(
        content="Postgres connection pooling best practices in production",
        type="solution",
        confidence="high",
    )
    candidate_ids = {c["id"] for c in result["candidates"]}
    assert result["node_id"] not in candidate_ids


@pytest.mark.asyncio
async def test_with_candidates_false_omits_field(
    engine: IntelligenceLayer,
) -> None:
    """Opt-out: candidates key is absent (not empty list) when scan skipped."""
    await engine.learn(
        content="Pre-existing node about Postgres pooling",
        type="pattern",
        confidence="high",
    )
    result = await engine.learn(
        content="Postgres connection pool sizing",
        type="solution",
        confidence="high",
        with_candidates=False,
    )
    assert "candidates" not in result


@pytest.mark.asyncio
async def test_candidates_works_for_low_confidence_save(
    engine: IntelligenceLayer,
) -> None:
    """Medium / low confidence saves still get candidates (no node created, but scan runs)."""
    await engine.learn(
        content="Established node about Kubernetes pod scheduling",
        type="pattern",
        confidence="high",
    )
    result = await engine.learn(
        content="Kubernetes pod scheduling tweak I tried once",
        type="workaround",
        confidence="low",
    )
    assert result["stored_as"] == "experience"
    assert result["node_id"] is None  # low confidence: no node created
    assert "candidates" in result
    # Should find the high-confidence seed via FTS5 even without creating a node.
    assert len(result["candidates"]) >= 1


@pytest.mark.asyncio
async def test_candidates_respects_max_limit(
    engine: IntelligenceLayer,
) -> None:
    """Returns at most _CANDIDATES_LIMIT entries even if more matches exist."""
    # Seed N+3 matching nodes so over-fetch logic gets exercised.
    for i in range(_CANDIDATES_LIMIT + 3):
        await engine.learn(
            content=f"Database indexing pattern variant {i}",
            type="pattern",
            confidence="high",
            with_candidates=False,  # don't slow this loop with candidate scans
        )

    result = await engine.learn(
        content="Database indexing optimization",
        type="solution",
        confidence="high",
    )
    assert len(result["candidates"]) <= _CANDIDATES_LIMIT
    # sim_rank should be a clean 0..N-1 sequence after exclusion filter.
    ranks = [c["sim_rank"] for c in result["candidates"]]
    assert ranks == list(range(len(result["candidates"])))


@pytest.mark.asyncio
async def test_candidates_default_is_on(
    engine: IntelligenceLayer,
) -> None:
    """Default behaviour: candidates key present (even if empty)."""
    result = await engine.learn(
        content="Some new fact",
        type="decision",
        confidence="high",
    )
    assert "candidates" in result
    assert isinstance(result["candidates"], list)


@pytest.mark.asyncio
async def test_candidates_snippet_truncation(
    engine: IntelligenceLayer,
) -> None:
    """Long descriptions are truncated with ellipsis; short ones are not."""
    long_content = "Kafka exactly-once semantics " * 20  # ~ 600 chars
    short_content = "Kafka basics"

    await engine.learn(content=long_content, type="pattern", confidence="high")
    await engine.learn(content=short_content, type="pattern", confidence="high")

    result = await engine.learn(
        content="Kafka guarantees",
        type="solution",
        confidence="high",
    )

    snippets = {c["snippet"] for c in result["candidates"]}
    # At least one snippet should be ellipsis-truncated (the long one).
    assert any(s.endswith("...") for s in snippets)
