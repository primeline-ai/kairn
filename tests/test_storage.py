"""Tests for SQLite storage backend including FTS5 triggers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from kairn.storage.sqlite_store import SQLiteStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Initialization ---


async def test_initialize_creates_db(tmp_path):
    db_path = tmp_path / "test.db"
    store = SQLiteStore(db_path)
    await store.initialize()
    assert db_path.exists()
    await store.close()


async def test_initialize_wal_mode(store: SQLiteStore):
    cursor = await store.db.execute("PRAGMA journal_mode")
    row = await cursor.fetchone()
    assert row[0] == "wal"


async def test_initialize_foreign_keys(store: SQLiteStore):
    cursor = await store.db.execute("PRAGMA foreign_keys")
    row = await cursor.fetchone()
    assert row[0] == 1


async def test_double_initialize(tmp_path):
    """Initializing twice should not error (IF NOT EXISTS)."""
    db_path = tmp_path / "test.db"
    store = SQLiteStore(db_path)
    await store.initialize()
    await store.initialize()
    await store.close()


async def test_migrate_legacy_experiences_adds_namespace_column(tmp_path):
    """Phase 3 migration: a pre-existing DB without `experiences.namespace`
    gets the column added during initialize(), existing rows default to
    'knowledge', and row count is preserved.

    This simulates the state of older engram databases that were created
    before the namespace field existed (e.g. the production brain DB
    with thousands of experiences).
    """
    import sqlite3

    db_path = tmp_path / "legacy.db"

    # Build a legacy schema DB by hand: experiences table without namespace.
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE experiences (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            context TEXT,
            confidence TEXT DEFAULT 'high',
            score REAL NOT NULL DEFAULT 1.0,
            decay_rate REAL NOT NULL,
            tags JSON,
            properties JSON,
            created_by TEXT,
            access_count INTEGER DEFAULT 0,
            promoted_to_node_id TEXT,
            created_at TEXT NOT NULL,
            last_accessed TEXT
        )
        """
    )
    # Seed legacy rows
    for i in range(3):
        conn.execute(
            "INSERT INTO experiences (id, type, content, decay_rate, created_at) "
            "VALUES (?, 'solution', ?, 0.003, '2026-01-01T00:00:00Z')",
            (f"legacy-{i}", f"Legacy experience {i}"),
        )
    conn.commit()

    # Confirm pre-state: namespace column is absent
    pre_cols = {row[1] for row in conn.execute("PRAGMA table_info(experiences)")}
    assert "namespace" not in pre_cols
    legacy_count = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
    assert legacy_count == 3
    conn.close()

    # Initialize the store — migration must fire
    store = SQLiteStore(db_path)
    await store.initialize()
    try:
        cursor = await store.db.execute("PRAGMA table_info(experiences)")
        post_cols = {row[1] for row in await cursor.fetchall()}
        assert "namespace" in post_cols

        # Row count preserved
        cursor = await store.db.execute("SELECT COUNT(*) FROM experiences")
        row = await cursor.fetchone()
        assert row[0] == 3

        # Legacy rows default to 'knowledge'
        cursor = await store.db.execute(
            "SELECT namespace FROM experiences WHERE id = ?", ("legacy-0",)
        )
        row = await cursor.fetchone()
        assert row[0] == "knowledge"

        # Migration is idempotent: re-initializing on the same DB is a no-op
        await store.close()
        store2 = SQLiteStore(db_path)
        await store2.initialize()
        cursor = await store2.db.execute("SELECT COUNT(*) FROM experiences")
        row = await cursor.fetchone()
        assert row[0] == 3
        await store2.close()
    finally:
        # store may already be closed by the idempotency branch above
        try:
            await store.close()
        except Exception:
            pass


# --- Node CRUD ---


async def test_insert_and_get_node(store: SQLiteStore):
    node = {
        "id": "n1",
        "namespace": "knowledge",
        "type": "concept",
        "name": "JWT Auth",
        "description": "JSON Web Token authentication",
        "properties": {"lang": "python"},
        "tags": ["auth", "security"],
        "created_by": None,
        "visibility": "workspace",
        "source_type": "manual",
        "source_ref": None,
        "created_at": _now(),
        "updated_at": None,
    }
    await store.insert_node(node)
    result = await store.get_node("n1")
    assert result is not None
    assert result["name"] == "JWT Auth"
    assert result["tags"] == ["auth", "security"]
    assert result["properties"] == {"lang": "python"}


async def test_get_node_not_found(store: SQLiteStore):
    result = await store.get_node("nonexistent")
    assert result is None


async def test_update_node(store: SQLiteStore):
    node = {
        "id": "n1",
        "namespace": "knowledge",
        "type": "concept",
        "name": "Original",
        "description": None,
        "properties": None,
        "tags": None,
        "created_by": None,
        "visibility": "workspace",
        "source_type": None,
        "source_ref": None,
        "created_at": _now(),
        "updated_at": None,
    }
    await store.insert_node(node)
    updated = await store.update_node("n1", {"name": "Updated", "updated_at": _now()})
    assert updated is not None
    assert updated["name"] == "Updated"


async def test_update_nonexistent_node(store: SQLiteStore):
    result = await store.update_node("nope", {"name": "x"})
    assert result is None


async def test_soft_delete_and_restore_node(store: SQLiteStore):
    node = {
        "id": "n1",
        "namespace": "knowledge",
        "type": "concept",
        "name": "Deletable",
        "description": None,
        "properties": None,
        "tags": None,
        "created_by": None,
        "visibility": "workspace",
        "source_type": None,
        "source_ref": None,
        "created_at": _now(),
        "updated_at": None,
    }
    await store.insert_node(node)

    assert await store.soft_delete_node("n1") is True
    assert await store.get_node("n1") is None  # Hidden
    assert await store.count_nodes() == 0

    assert await store.restore_node("n1") is True
    assert await store.get_node("n1") is not None
    assert await store.count_nodes() == 1


async def test_soft_delete_nonexistent(store: SQLiteStore):
    assert await store.soft_delete_node("nope") is False


# --- Node queries ---


async def test_query_nodes_by_namespace(store: SQLiteStore):
    for i, ns in enumerate(["knowledge", "knowledge", "idea"]):
        await store.insert_node({
            "id": f"n{i}",
            "namespace": ns,
            "type": "concept",
            "name": f"Node {i}",
            "description": None,
            "properties": None,
            "tags": None,
            "created_by": None,
            "visibility": "workspace",
            "source_type": None,
            "source_ref": None,
            "created_at": _now(),
            "updated_at": None,
        })

    results = await store.query_nodes(namespace="knowledge")
    assert len(results) == 2

    results = await store.query_nodes(namespace="idea")
    assert len(results) == 1


async def test_query_nodes_by_type(store: SQLiteStore):
    for i, t in enumerate(["concept", "concept", "pattern"]):
        await store.insert_node({
            "id": f"n{i}",
            "namespace": "knowledge",
            "type": t,
            "name": f"Node {i}",
            "description": None,
            "properties": None,
            "tags": None,
            "created_by": None,
            "visibility": "workspace",
            "source_type": None,
            "source_ref": None,
            "created_at": _now(),
            "updated_at": None,
        })

    results = await store.query_nodes(node_type="pattern")
    assert len(results) == 1


async def test_query_nodes_by_tags(store: SQLiteStore):
    await store.insert_node({
        "id": "n1",
        "namespace": "knowledge",
        "type": "concept",
        "name": "Tagged",
        "description": None,
        "properties": None,
        "tags": ["python", "auth"],
        "created_by": None,
        "visibility": "workspace",
        "source_type": None,
        "source_ref": None,
        "created_at": _now(),
        "updated_at": None,
    })
    await store.insert_node({
        "id": "n2",
        "namespace": "knowledge",
        "type": "concept",
        "name": "Untagged",
        "description": None,
        "properties": None,
        "tags": ["javascript"],
        "created_by": None,
        "visibility": "workspace",
        "source_type": None,
        "source_ref": None,
        "created_at": _now(),
        "updated_at": None,
    })

    results = await store.query_nodes(tags=["python"])
    assert len(results) == 1
    assert results[0]["name"] == "Tagged"


async def test_query_nodes_pagination(store: SQLiteStore):
    for i in range(15):
        await store.insert_node({
            "id": f"n{i}",
            "namespace": "knowledge",
            "type": "concept",
            "name": f"Node {i}",
            "description": None,
            "properties": None,
            "tags": None,
            "created_by": None,
            "visibility": "workspace",
            "source_type": None,
            "source_ref": None,
            "created_at": _now(),
            "updated_at": None,
        })

    page1 = await store.query_nodes(limit=10, offset=0)
    assert len(page1) == 10

    page2 = await store.query_nodes(limit=10, offset=10)
    assert len(page2) == 5


# --- FTS5 tests ---


async def test_fts5_search_nodes(store: SQLiteStore):
    await store.insert_node({
        "id": "n1",
        "namespace": "knowledge",
        "type": "concept",
        "name": "Redis Caching",
        "description": "Using Redis for distributed caching in microservices",
        "properties": None,
        "tags": None,
        "created_by": None,
        "visibility": "workspace",
        "source_type": None,
        "source_ref": None,
        "created_at": _now(),
        "updated_at": None,
    })
    await store.insert_node({
        "id": "n2",
        "namespace": "knowledge",
        "type": "concept",
        "name": "PostgreSQL Indexing",
        "description": "B-tree and GIN indexes for performance",
        "properties": None,
        "tags": None,
        "created_by": None,
        "visibility": "workspace",
        "source_type": None,
        "source_ref": None,
        "created_at": _now(),
        "updated_at": None,
    })

    results = await store.query_nodes(text="Redis")
    assert len(results) == 1
    assert results[0]["name"] == "Redis Caching"

    results = await store.query_nodes(text="caching")
    assert len(results) == 1


async def test_fts5_update_sync(store: SQLiteStore):
    """FTS5 trigger must update index when node is updated."""
    await store.insert_node({
        "id": "n1",
        "namespace": "knowledge",
        "type": "concept",
        "name": "Old Name",
        "description": "Old description",
        "properties": None,
        "tags": None,
        "created_by": None,
        "visibility": "workspace",
        "source_type": None,
        "source_ref": None,
        "created_at": _now(),
        "updated_at": None,
    })

    # Search finds old name
    results = await store.query_nodes(text="Old")
    assert len(results) == 1

    # Update name
    await store.update_node("n1", {"name": "New Name", "description": "New description"})

    # Old name no longer found
    results = await store.query_nodes(text="Old")
    assert len(results) == 0

    # New name found
    results = await store.query_nodes(text="New")
    assert len(results) == 1


async def test_fts5_delete_sync(store: SQLiteStore):
    """FTS5 trigger must remove from index on delete."""
    await store.insert_node({
        "id": "n1",
        "namespace": "knowledge",
        "type": "concept",
        "name": "Searchable",
        "description": "Find me",
        "properties": None,
        "tags": None,
        "created_by": None,
        "visibility": "workspace",
        "source_type": None,
        "source_ref": None,
        "created_at": _now(),
        "updated_at": None,
    })

    results = await store.query_nodes(text="Searchable")
    assert len(results) == 1

    # Soft-delete hides from normal queries but FTS still has it
    await store.soft_delete_node("n1")
    # Our FTS query joins with deleted_at IS NULL, so it's filtered
    results = await store.query_nodes(text="Searchable")
    assert len(results) == 0


async def test_fts5_namespace_filter(store: SQLiteStore):
    """FTS5 search respects namespace filter."""
    await store.insert_node({
        "id": "n1",
        "namespace": "knowledge",
        "type": "concept",
        "name": "Auth Pattern",
        "description": "Authentication pattern",
        "properties": None,
        "tags": None,
        "created_by": None,
        "visibility": "workspace",
        "source_type": None,
        "source_ref": None,
        "created_at": _now(),
        "updated_at": None,
    })
    await store.insert_node({
        "id": "n2",
        "namespace": "idea",
        "type": "concept",
        "name": "Auth Idea",
        "description": "New authentication approach",
        "properties": None,
        "tags": None,
        "created_by": None,
        "visibility": "workspace",
        "source_type": None,
        "source_ref": None,
        "created_at": _now(),
        "updated_at": None,
    })

    results = await store.query_nodes(text="Auth", namespace="knowledge")
    assert len(results) == 1
    assert results[0]["namespace"] == "knowledge"


# --- Edge CRUD ---


async def test_insert_and_get_edges(store: SQLiteStore):
    # Create nodes first
    for nid in ("n1", "n2"):
        await store.insert_node({
            "id": nid,
            "namespace": "knowledge",
            "type": "concept",
            "name": f"Node {nid}",
            "description": None,
            "properties": None,
            "tags": None,
            "created_by": None,
            "visibility": "workspace",
            "source_type": None,
            "source_ref": None,
            "created_at": _now(),
            "updated_at": None,
        })

    edge = {
        "source_id": "n1",
        "target_id": "n2",
        "type": "related_to",
        "weight": 0.8,
        "properties": {"context": "auth"},
        "created_by": None,
        "created_at": _now(),
    }
    await store.insert_edge(edge)

    edges = await store.get_edges(source_id="n1")
    assert len(edges) == 1
    assert edges[0]["target_id"] == "n2"
    assert edges[0]["weight"] == 0.8

    edges = await store.get_edges(target_id="n2")
    assert len(edges) == 1


async def test_delete_edge(store: SQLiteStore):
    for nid in ("n1", "n2"):
        await store.insert_node({
            "id": nid,
            "namespace": "knowledge",
            "type": "concept",
            "name": f"Node {nid}",
            "description": None,
            "properties": None,
            "tags": None,
            "created_by": None,
            "visibility": "workspace",
            "source_type": None,
            "source_ref": None,
            "created_at": _now(),
            "updated_at": None,
        })

    await store.insert_edge({
        "source_id": "n1",
        "target_id": "n2",
        "type": "related_to",
        "weight": 1.0,
        "properties": None,
        "created_by": None,
        "created_at": _now(),
    })

    assert await store.delete_edge("n1", "n2", "related_to") is True
    assert await store.delete_edge("n1", "n2", "related_to") is False
    assert await store.count_edges() == 0


# --- Experience operations ---


async def test_insert_and_get_experience(store: SQLiteStore):
    exp = {
        "id": "e1",
        "type": "solution",
        "content": "Use Redis for caching",
        "context": "Building API",
        "confidence": "high",
        "score": 1.0,
        "decay_rate": 0.00347,
        "tags": ["caching", "redis"],
        "properties": None,
        "created_by": None,
        "access_count": 0,
        "promoted_to_node_id": None,
        "created_at": _now(),
        "last_accessed": None,
    }
    await store.insert_experience(exp)
    result = await store.get_experience("e1")
    assert result is not None
    assert result["content"] == "Use Redis for caching"
    assert result["confidence"] == "high"


async def test_experience_fts5(store: SQLiteStore):
    await store.insert_experience({
        "id": "e1",
        "type": "solution",
        "content": "Redis is great for distributed caching",
        "context": "Microservice architecture",
        "confidence": "high",
        "score": 1.0,
        "decay_rate": 0.00347,
        "tags": None,
        "properties": None,
        "created_by": None,
        "access_count": 0,
        "promoted_to_node_id": None,
        "created_at": _now(),
        "last_accessed": None,
    })
    await store.insert_experience({
        "id": "e2",
        "type": "pattern",
        "content": "Always validate JWT tokens server-side",
        "context": "Security review",
        "confidence": "high",
        "score": 1.0,
        "decay_rate": 0.00231,
        "tags": None,
        "properties": None,
        "created_by": None,
        "access_count": 0,
        "promoted_to_node_id": None,
        "created_at": _now(),
        "last_accessed": None,
    })

    results = await store.query_experiences(text="Redis")
    assert len(results) == 1
    assert results[0]["id"] == "e1"

    results = await store.query_experiences(text="JWT")
    assert len(results) == 1
    assert results[0]["id"] == "e2"


async def test_increment_access_count(store: SQLiteStore):
    await store.insert_experience({
        "id": "e1",
        "type": "solution",
        "content": "Test",
        "context": None,
        "confidence": "high",
        "score": 1.0,
        "decay_rate": 0.00347,
        "tags": None,
        "properties": None,
        "created_by": None,
        "access_count": 0,
        "promoted_to_node_id": None,
        "created_at": _now(),
        "last_accessed": None,
    })

    for i in range(3):
        result = await store.increment_access_count("e1")
        assert result is not None
        assert result["access_count"] == i + 1
        assert result["last_accessed"] is not None


async def test_auto_promotion_flag(store: SQLiteStore):
    """Trigger sets needs_promotion when access_count >= 5."""
    await store.insert_experience({
        "id": "e1",
        "type": "solution",
        "content": "Frequently accessed pattern",
        "context": None,
        "confidence": "medium",
        "score": 1.0,
        "decay_rate": 0.00347,
        "tags": None,
        "properties": None,
        "created_by": None,
        "access_count": 0,
        "promoted_to_node_id": None,
        "created_at": _now(),
        "last_accessed": None,
    })

    # Access 4 times - should NOT be flagged
    for _ in range(4):
        await store.increment_access_count("e1")

    result = await store.get_experience("e1")
    props = result.get("properties") or {}
    assert props.get("needs_promotion") != 1

    # 5th access - should trigger promotion flag
    await store.increment_access_count("e1")
    result = await store.get_experience("e1")
    props = result.get("properties") or {}
    assert props.get("needs_promotion") == 1

    # Should appear in promotable list
    promotable = await store.get_promotable_experiences()
    assert len(promotable) == 1
    assert promotable[0]["id"] == "e1"


async def test_delete_experience(store: SQLiteStore):
    await store.insert_experience({
        "id": "e1",
        "type": "solution",
        "content": "Deletable",
        "context": None,
        "confidence": "high",
        "score": 1.0,
        "decay_rate": 0.00347,
        "tags": None,
        "properties": None,
        "created_by": None,
        "access_count": 0,
        "promoted_to_node_id": None,
        "created_at": _now(),
        "last_accessed": None,
    })

    assert await store.delete_experience("e1") is True
    assert await store.get_experience("e1") is None
    assert await store.delete_experience("e1") is False


# --- Project operations ---


async def test_project_crud(store: SQLiteStore):
    project = {
        "id": "p1",
        "name": "Test Project",
        "phase": "planning",
        "goals": ["ship v1", "get users"],
        "active": False,
        "created_by": None,
        "stakeholders": None,
        "success_metrics": None,
        "created_at": _now(),
        "updated_at": None,
    }
    await store.insert_project(project)

    result = await store.get_project("p1")
    assert result is not None
    assert result["name"] == "Test Project"
    assert result["goals"] == ["ship v1", "get users"]

    updated = await store.update_project("p1", {"phase": "active", "updated_at": _now()})
    assert updated["phase"] == "active"


async def test_set_active_project(store: SQLiteStore):
    for pid in ("p1", "p2"):
        await store.insert_project({
            "id": pid,
            "name": f"Project {pid}",
            "phase": "planning",
            "goals": None,
            "active": False,
            "created_by": None,
            "stakeholders": None,
            "success_metrics": None,
            "created_at": _now(),
            "updated_at": None,
        })

    assert await store.set_active_project("p1") is True

    projects = await store.list_projects(active_only=True)
    assert len(projects) == 1
    assert projects[0]["id"] == "p1"

    # Switch active
    assert await store.set_active_project("p2") is True
    projects = await store.list_projects(active_only=True)
    assert len(projects) == 1
    assert projects[0]["id"] == "p2"


# --- Progress operations ---


async def test_progress_logging(store: SQLiteStore):
    await store.insert_project({
        "id": "p1",
        "name": "Test",
        "phase": "active",
        "goals": None,
        "active": True,
        "created_by": None,
        "stakeholders": None,
        "success_metrics": None,
        "created_at": _now(),
        "updated_at": None,
    })

    await store.insert_progress({
        "id": "pg1",
        "project_id": "p1",
        "type": "progress",
        "action": "Implemented auth",
        "result": "Working JWT",
        "next_step": "Add refresh tokens",
        "created_by": None,
        "created_at": _now(),
    })

    entries = await store.get_progress("p1")
    assert len(entries) == 1
    assert entries[0]["action"] == "Implemented auth"


# --- Idea operations ---


async def test_idea_crud(store: SQLiteStore):
    idea = {
        "id": "i1",
        "title": "Build a CLI",
        "status": "draft",
        "category": "tooling",
        "score": 8.5,
        "properties": {"priority": "high"},
        "created_by": None,
        "visibility": "private",
        "created_at": _now(),
        "updated_at": None,
    }
    await store.insert_idea(idea)

    result = await store.get_idea("i1")
    assert result is not None
    assert result["title"] == "Build a CLI"
    assert result["score"] == 8.5

    updated = await store.update_idea("i1", {"status": "approved"})
    assert updated["status"] == "approved"


async def test_list_ideas_filtered(store: SQLiteStore):
    for i, status in enumerate(["draft", "draft", "approved"]):
        await store.insert_idea({
            "id": f"i{i}",
            "title": f"Idea {i}",
            "status": status,
            "category": "general",
            "score": float(i),
            "properties": None,
            "created_by": None,
            "visibility": "private",
            "created_at": _now(),
            "updated_at": None,
        })

    drafts = await store.list_ideas(status="draft")
    assert len(drafts) == 2

    approved = await store.list_ideas(status="approved")
    assert len(approved) == 1


# --- Route operations ---


async def test_route_upsert_and_get(store: SQLiteStore):
    await store.upsert_route("auth", ["n1", "n2"], 0.9)
    await store.upsert_route("caching", ["n3"], 0.7)

    routes = await store.get_routes(["auth", "caching"])
    assert len(routes) == 2

    # Upsert updates existing
    await store.upsert_route("auth", ["n1", "n2", "n4"], 0.95)
    routes = await store.get_routes(["auth"])
    assert len(routes) == 1
    assert "n4" in routes[0]["node_ids"]


# --- Activity log ---


async def test_activity_log(store: SQLiteStore):
    await store.log_activity({
        "id": "a1",
        "user_id": None,
        "activity_type": "created",
        "entity_type": "node",
        "entity_id": "n1",
        "description": "Created node",
        "created_at": _now(),
    })

    entries = await store.get_activity_log()
    assert len(entries) == 1
    assert entries[0]["activity_type"] == "created"


# --- Stats ---


async def test_get_stats(store: SQLiteStore):
    await store.insert_node({
        "id": "n1",
        "namespace": "knowledge",
        "type": "concept",
        "name": "Test",
        "description": None,
        "properties": None,
        "tags": None,
        "created_by": None,
        "visibility": "workspace",
        "source_type": None,
        "source_ref": None,
        "created_at": _now(),
        "updated_at": None,
    })

    stats = await store.get_stats()
    assert stats["nodes"] == 1
    assert stats["edges"] == 0
    assert stats["experiences"] == 0
    assert stats["namespaces"]["knowledge"] == 1


# --- Error handling ---


async def test_store_not_initialized():
    store = SQLiteStore(Path("/tmp/never.db"))
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = store.db


async def test_increment_nonexistent_experience(store: SQLiteStore):
    result = await store.increment_access_count("nonexistent")
    assert result is None
