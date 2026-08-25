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
