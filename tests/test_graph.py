"""Tests for the Graph Engine."""

from __future__ import annotations

import pytest

from kairn.core.graph import GraphEngine
from kairn.events.bus import EventBus
from kairn.events.types import EventType
from kairn.storage.sqlite_store import SQLiteStore


@pytest.fixture
async def graph(store: SQLiteStore) -> GraphEngine:
    bus = EventBus()
    return GraphEngine(store, bus)


async def test_add_and_get_node(graph: GraphEngine):
    node = await graph.add_node(name="JWT Auth", type="concept", description="Token-based auth")
    assert node.name == "JWT Auth"
    assert node.id

    fetched = await graph.get_node(node.id)
    assert fetched is not None
    assert fetched.name == "JWT Auth"


async def test_get_nonexistent_node(graph: GraphEngine):
    result = await graph.get_node("nope")
    assert result is None


async def test_update_node(graph: GraphEngine):
    node = await graph.add_node(name="Original", type="concept")
    updated = await graph.update_node(node.id, name="Updated", description="New desc")
    assert updated is not None
    assert updated.name == "Updated"
    assert updated.description == "New desc"
    assert updated.updated_at is not None


async def test_remove_and_restore_node(graph: GraphEngine):
    node = await graph.add_node(name="Removable", type="concept")
    assert await graph.remove_node(node.id) is True
    assert await graph.get_node(node.id) is None
    assert await graph.restore_node(node.id) is True
    assert await graph.get_node(node.id) is not None


async def test_query_by_text(graph: GraphEngine):
    await graph.add_node(name="Redis Caching", type="concept", description="Distributed cache")
    await graph.add_node(name="PostgreSQL", type="concept", description="Relational database")

    results = await graph.query(text="Redis")
    assert len(results) == 1
    assert results[0].name == "Redis Caching"


async def test_query_by_namespace(graph: GraphEngine):
    await graph.add_node(name="Node A", type="concept", namespace="knowledge")
    await graph.add_node(name="Node B", type="concept", namespace="idea")

    results = await graph.query(namespace="idea")
    assert len(results) == 1
    assert results[0].name == "Node B"


async def test_query_by_type(graph: GraphEngine):
    await graph.add_node(name="Pattern A", type="pattern")
    await graph.add_node(name="Concept A", type="concept")

    results = await graph.query(node_type="pattern")
    assert len(results) == 1


async def test_query_by_tags(graph: GraphEngine):
    await graph.add_node(name="Tagged", type="concept", tags=["python", "auth"])
    await graph.add_node(name="Other", type="concept", tags=["javascript"])

    results = await graph.query(tags=["python"])
    assert len(results) == 1
    assert results[0].name == "Tagged"


async def test_query_pagination(graph: GraphEngine):
    for i in range(15):
        await graph.add_node(name=f"Node {i}", type="concept")

    page1 = await graph.query(limit=10, offset=0)
    assert len(page1) == 10

    page2 = await graph.query(limit=10, offset=10)
    assert len(page2) == 5


async def test_connect_nodes(graph: GraphEngine):
    n1 = await graph.add_node(name="Auth", type="concept")
    n2 = await graph.add_node(name="JWT", type="concept")

    edge = await graph.connect(n1.id, n2.id, "uses", weight=0.9)
    assert edge.source_id == n1.id
    assert edge.target_id == n2.id
    assert edge.weight == 0.9


async def test_connect_nonexistent_source(graph: GraphEngine):
    n = await graph.add_node(name="Target", type="concept")
    with pytest.raises(ValueError, match="Source node not found"):
        await graph.connect("nope", n.id, "uses")


async def test_connect_nonexistent_target(graph: GraphEngine):
    n = await graph.add_node(name="Source", type="concept")
    with pytest.raises(ValueError, match="Target node not found"):
        await graph.connect(n.id, "nope", "uses")


async def test_disconnect(graph: GraphEngine):
    n1 = await graph.add_node(name="A", type="concept")
    n2 = await graph.add_node(name="B", type="concept")
    await graph.connect(n1.id, n2.id, "related_to")

    assert await graph.disconnect(n1.id, n2.id, "related_to") is True
    assert await graph.disconnect(n1.id, n2.id, "related_to") is False


async def test_get_edges(graph: GraphEngine):
    n1 = await graph.add_node(name="A", type="concept")
    n2 = await graph.add_node(name="B", type="concept")
    n3 = await graph.add_node(name="C", type="concept")
    await graph.connect(n1.id, n2.id, "related_to")
    await graph.connect(n1.id, n3.id, "uses")

    edges = await graph.get_edges(source_id=n1.id)
    explicit_edges = [e for e in edges if e.type != "auto_related"]
    assert len(explicit_edges) == 2


async def test_get_related_bfs(graph: GraphEngine):
    n1 = await graph.add_node(name="Center", type="concept")
    n2 = await graph.add_node(name="Neighbor1", type="concept")
    n3 = await graph.add_node(name="Neighbor2", type="concept")
    n4 = await graph.add_node(name="Depth2", type="concept")

    await graph.connect(n1.id, n2.id, "related_to")
    await graph.connect(n1.id, n3.id, "related_to")
    await graph.connect(n2.id, n4.id, "related_to")

    depth1 = await graph.get_related(n1.id, depth=1)
    depth1_names = {r["node"]["name"] for r in depth1}
    assert "Neighbor1" in depth1_names
    assert "Neighbor2" in depth1_names
    assert "Depth2" not in depth1_names

    depth2 = await graph.get_related(n1.id, depth=2)
    depth2_names = {r["node"]["name"] for r in depth2}
    assert "Depth2" in depth2_names


async def test_auto_link_on_add(graph: GraphEngine):
    n1 = await graph.add_node(name="Redis caching", type="concept", description="Redis is an in-memory cache for distributed systems")
    n2 = await graph.add_node(name="Redis patterns", type="pattern", description="Common Redis caching patterns and strategies")

    edges = await graph.get_edges(source_id=n2.id)
    auto_edges = [e for e in edges if e.type == "auto_related"]
    assert len(auto_edges) >= 1
    assert any(e.target_id == n1.id for e in auto_edges)


async def test_auto_link_sets_created_by(graph: GraphEngine):
    """Auto-generated edges must carry created_by='auto_link' for provenance."""
    n1 = await graph.add_node(
        name="Database indexing",
        type="concept",
        description="B-tree indexes for query optimization",
    )
    n2 = await graph.add_node(
        name="Database query tuning",
        type="pattern",
        description="Techniques for optimizing database indexing and queries",
    )

    edges = await graph.get_edges(source_id=n2.id)
    auto_edges = [e for e in edges if e.type == "auto_related"]
    assert len(auto_edges) >= 1
    for edge in auto_edges:
        assert edge.created_by == "auto_link"


async def test_connect_propagates_created_by(graph: GraphEngine):
    """Manual connect() must propagate caller-provided created_by."""
    n1 = await graph.add_node(name="Source", type="concept")
    n2 = await graph.add_node(name="Target", type="concept")

    edge = await graph.connect(n1.id, n2.id, "uses", created_by="test_caller")
    assert edge.created_by == "test_caller"


async def test_connect_default_created_by_is_none(graph: GraphEngine):
    """connect() without created_by defaults to None (backward compat)."""
    n1 = await graph.add_node(name="A", type="concept")
    n2 = await graph.add_node(name="B", type="concept")

    edge = await graph.connect(n1.id, n2.id, "related_to")
    assert edge.created_by is None


async def test_auto_link_limits_to_three(graph: GraphEngine):
    """Auto-link creates at most 3 edges per node (was 5)."""
    # Create 6 nodes that will all share keywords with the final one
    for i in range(6):
        await graph.add_node(
            name=f"Caching pattern variant {i}",
            type="concept",
            description=f"Redis caching strategy number {i}",
        )

    n_new = await graph.add_node(
        name="Redis caching overview",
        type="pattern",
        description="Overview of Redis caching strategies",
    )

    edges = await graph.get_edges(source_id=n_new.id)
    auto_edges = [e for e in edges if e.type == "auto_related"]
    assert 1 <= len(auto_edges) <= 3


async def test_stats(graph: GraphEngine):
    await graph.add_node(name="Test", type="concept")
    stats = await graph.stats()
    assert stats["nodes"] >= 1


async def test_events_emitted(store: SQLiteStore):
    bus = EventBus()
    events: list = []

    async def collect(et, data):
        events.append((et, data))

    bus.on_all(collect)

    graph = GraphEngine(store, bus)
    node = await graph.add_node(name="Test", type="concept")

    assert any(e[0] == EventType.NODE_CREATED for e in events)

    await graph.remove_node(node.id)
    assert any(e[0] == EventType.NODE_DELETED for e in events)
