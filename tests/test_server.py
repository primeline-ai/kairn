"""Tests for the FastMCP Server (Gate 3: 18 tools)."""

from __future__ import annotations

import json

import pytest
from fastmcp import Client

from kairn.server import create_server


def _text(result) -> str:
    """Extract text from CallToolResult."""
    return result.content[0].text


def _data(result) -> dict:
    """Extract parsed JSON from CallToolResult."""
    return json.loads(result.content[0].text)


@pytest.fixture
async def client(tmp_path):
    server = create_server(str(tmp_path / "test.db"))
    async with Client(server) as c:
        yield c


async def test_list_tools(client: Client):
    tools = await client.list_tools()
    names = {t.name for t in tools}
    expected = {
        "kn_add", "kn_connect", "kn_query", "kn_remove", "kn_status",
        "kn_project", "kn_projects", "kn_log",
        "kn_save", "kn_memories", "kn_prune",
        "kn_idea", "kn_ideas",
        "kn_learn", "kn_recall", "kn_crossref", "kn_context", "kn_related",
        "kn_promote_pending",
    }
    assert expected.issubset(names)
    assert len(names) == 19


async def test_tool_descriptions_short(client: Client):
    tools = await client.list_tools()
    for tool in tools:
        assert len(tool.description) <= 500, f"{tool.name} description too long ({len(tool.description)} chars)"


async def test_kn_add_node(client: Client):
    result = await client.call_tool("kn_add", {
        "name": "JWT Auth",
        "type": "concept",
        "description": "Token-based authentication",
    })
    data = _data(result)
    assert data["name"] == "JWT Auth"
    assert data["_v"] == "1.0"
    assert "id" in data


async def test_kn_add_with_all_fields(client: Client):
    result = await client.call_tool("kn_add", {
        "name": "Redis Cache",
        "type": "pattern",
        "namespace": "idea",
        "description": "In-memory caching layer",
        "tags": ["cache", "redis"],
    })
    data = _data(result)
    assert data["name"] == "Redis Cache"
    assert data["namespace"] == "idea"
    assert "cache" in data["tags"]


async def test_kn_add_minimal(client: Client):
    result = await client.call_tool("kn_add", {
        "name": "Minimal",
        "type": "concept",
    })
    data = _data(result)
    assert data["name"] == "Minimal"


async def test_kn_query_by_text(client: Client):
    await client.call_tool("kn_add", {
        "name": "PostgreSQL",
        "type": "concept",
        "description": "Relational database",
    })
    await client.call_tool("kn_add", {
        "name": "Redis",
        "type": "concept",
        "description": "In-memory cache",
    })

    result = await client.call_tool("kn_query", {"text": "PostgreSQL"})
    data = _data(result)
    assert data["count"] >= 1
    names = [n["name"] for n in data["nodes"]]
    assert "PostgreSQL" in names


async def test_kn_query_by_type(client: Client):
    await client.call_tool("kn_add", {"name": "Pattern A", "type": "pattern"})
    await client.call_tool("kn_add", {"name": "Concept A", "type": "concept"})

    result = await client.call_tool("kn_query", {"node_type": "pattern"})
    data = _data(result)
    assert data["count"] == 1
    assert data["nodes"][0]["name"] == "Pattern A"


async def test_kn_query_by_namespace(client: Client):
    await client.call_tool("kn_add", {"name": "Idea A", "type": "concept", "namespace": "idea"})
    await client.call_tool("kn_add", {"name": "Knowledge A", "type": "concept", "namespace": "knowledge"})

    result = await client.call_tool("kn_query", {"namespace": "idea"})
    data = _data(result)
    assert data["count"] == 1
    assert data["nodes"][0]["name"] == "Idea A"


async def test_kn_query_by_tags(client: Client):
    await client.call_tool("kn_add", {"name": "Tagged", "type": "concept", "tags": ["python"]})
    await client.call_tool("kn_add", {"name": "Untagged", "type": "concept"})

    result = await client.call_tool("kn_query", {"tags": ["python"]})
    data = _data(result)
    assert data["count"] == 1
    assert data["nodes"][0]["name"] == "Tagged"


async def test_kn_query_pagination(client: Client):
    for i in range(15):
        await client.call_tool("kn_add", {"name": f"Node {i}", "type": "concept"})

    result = await client.call_tool("kn_query", {"limit": 5})
    data = _data(result)
    assert data["count"] == 5


async def test_kn_query_detail_full(client: Client):
    await client.call_tool("kn_add", {
        "name": "Full Detail",
        "type": "concept",
        "description": "Detailed node",
        "tags": ["test"],
    })

    result = await client.call_tool("kn_query", {"text": "Full Detail", "detail": "full"})
    data = _data(result)
    assert data["count"] >= 1
    assert "description" in data["nodes"][0]
    assert data["nodes"][0]["description"] == "Detailed node"


async def test_kn_query_empty_result(client: Client):
    result = await client.call_tool("kn_query", {"text": "nonexistent"})
    data = _data(result)
    assert data["count"] == 0
    assert data["nodes"] == []


async def test_kn_connect(client: Client):
    r1 = _data(await client.call_tool("kn_add", {"name": "Auth", "type": "concept"}))
    r2 = _data(await client.call_tool("kn_add", {"name": "JWT", "type": "concept"}))

    result = await client.call_tool("kn_connect", {
        "source_id": r1["id"],
        "target_id": r2["id"],
        "edge_type": "uses",
        "weight": 0.9,
    })
    data = _data(result)
    assert data["type"] == "uses"
    assert data["weight"] == 0.9


async def test_kn_connect_invalid_source(client: Client):
    n = _data(await client.call_tool("kn_add", {"name": "Target", "type": "concept"}))

    result = await client.call_tool("kn_connect", {
        "source_id": "nonexistent",
        "target_id": n["id"],
        "edge_type": "uses",
    })
    data = _data(result)
    assert "error" in data


async def test_kn_remove_node(client: Client):
    n = _data(await client.call_tool("kn_add", {"name": "Removable", "type": "concept"}))

    result = await client.call_tool("kn_remove", {"node_id": n["id"]})
    data = _data(result)
    assert data["removed"] == "node"

    query = _data(await client.call_tool("kn_query", {"text": "Removable"}))
    assert query["count"] == 0


async def test_kn_remove_nonexistent(client: Client):
    result = await client.call_tool("kn_remove", {"node_id": "nope"})
    data = _data(result)
    assert "error" in data


async def test_kn_remove_edge(client: Client):
    n1 = _data(await client.call_tool("kn_add", {"name": "A", "type": "concept"}))
    n2 = _data(await client.call_tool("kn_add", {"name": "B", "type": "concept"}))

    await client.call_tool("kn_connect", {
        "source_id": n1["id"],
        "target_id": n2["id"],
        "edge_type": "related_to",
    })

    result = await client.call_tool("kn_remove", {
        "source_id": n1["id"],
        "target_id": n2["id"],
        "edge_type": "related_to",
    })
    data = _data(result)
    assert data["removed"] == "edge"


async def test_kn_status(client: Client):
    await client.call_tool("kn_add", {"name": "Test", "type": "concept"})

    result = await client.call_tool("kn_status", {})
    data = _data(result)
    assert data["nodes"] >= 1
    assert data["_v"] == "1.0"


async def test_kn_status_empty(client: Client):
    result = await client.call_tool("kn_status", {})
    data = _data(result)
    assert "nodes" in data


async def test_kn_add_returns_valid_json(client: Client):
    result = await client.call_tool("kn_add", {
        "name": "JSON Test",
        "type": "concept",
    })
    data = _data(result)
    assert "id" in data
    assert data["name"] == "JSON Test"
    assert data["_v"] == "1.0"


# ── Project Memory tool tests ────────────────────────────────


async def test_kn_project_create(client: Client):
    result = await client.call_tool("kn_project", {
        "name": "Alpha",
        "goals": ["Ship V1"],
    })
    data = _data(result)
    assert data["_v"] == "1.0"
    assert data["name"] == "Alpha"
    assert data["phase"] == "planning"
    assert "id" in data


async def test_kn_project_update(client: Client):
    created = _data(await client.call_tool("kn_project", {"name": "Beta"}))
    pid = created["id"]

    updated = _data(await client.call_tool("kn_project", {
        "name": "Beta v2",
        "project_id": pid,
        "phase": "active",
    }))
    assert updated["name"] == "Beta v2"
    assert updated["phase"] == "active"


async def test_kn_project_invalid_phase(client: Client):
    created = _data(await client.call_tool("kn_project", {"name": "Gamma"}))
    pid = created["id"]

    result = _data(await client.call_tool("kn_project", {
        "name": "Gamma",
        "project_id": pid,
        "phase": "invalid",
    }))
    assert "error" in result


async def test_kn_project_not_found(client: Client):
    result = _data(await client.call_tool("kn_project", {
        "name": "Ghost",
        "project_id": "nonexistent",
    }))
    assert "error" in result


async def test_kn_project_phase_rejected_on_create(client: Client):
    result = _data(await client.call_tool("kn_project", {
        "name": "Eager",
        "phase": "active",
    }))
    assert "error" in result
    assert "phase" in result["error"].lower()


async def test_kn_projects_list(client: Client):
    await client.call_tool("kn_project", {"name": "P1"})
    await client.call_tool("kn_project", {"name": "P2"})

    result = _data(await client.call_tool("kn_projects", {}))
    assert result["_v"] == "1.0"
    assert result["count"] == 2
    names = [p["name"] for p in result["projects"]]
    assert "P1" in names
    assert "P2" in names


async def test_kn_projects_set_active(client: Client):
    p = _data(await client.call_tool("kn_project", {"name": "Activate Me"}))

    result = _data(await client.call_tool("kn_projects", {
        "set_active": p["id"],
    }))
    assert result["_v"] == "1.0"
    active = [pr for pr in result["projects"] if pr["active"]]
    assert len(active) == 1
    assert active[0]["id"] == p["id"]


async def test_kn_projects_set_active_not_found(client: Client):
    result = _data(await client.call_tool("kn_projects", {
        "set_active": "nonexistent",
    }))
    assert "error" in result


async def test_kn_projects_active_only(client: Client):
    p1 = _data(await client.call_tool("kn_project", {"name": "Active"}))
    await client.call_tool("kn_project", {"name": "Inactive"})
    await client.call_tool("kn_projects", {"set_active": p1["id"]})

    result = _data(await client.call_tool("kn_projects", {"active_only": True}))
    assert result["count"] == 1
    assert result["projects"][0]["name"] == "Active"


async def test_kn_log_progress(client: Client):
    p = _data(await client.call_tool("kn_project", {"name": "Logged"}))

    result = _data(await client.call_tool("kn_log", {
        "project_id": p["id"],
        "action": "Implemented auth",
        "result": "Working",
        "next_step": "Add tests",
    }))
    assert result["_v"] == "1.0"
    assert result["type"] == "progress"
    assert result["action"] == "Implemented auth"


async def test_kn_log_failure(client: Client):
    p = _data(await client.call_tool("kn_project", {"name": "Failed"}))

    result = _data(await client.call_tool("kn_log", {
        "project_id": p["id"],
        "action": "Deploy crashed",
        "type": "failure",
        "result": "OOM error",
    }))
    assert result["type"] == "failure"


async def test_kn_log_empty_action(client: Client):
    p = _data(await client.call_tool("kn_project", {"name": "X"}))
    result = _data(await client.call_tool("kn_log", {
        "project_id": p["id"],
        "action": "",
    }))
    assert "error" in result


# ── Experience Memory tool tests ─────────────────────────────


async def test_kn_save(client: Client):
    result = _data(await client.call_tool("kn_save", {
        "content": "Use WAL mode for concurrent SQLite access",
        "type": "solution",
        "confidence": "high",
        "tags": ["sqlite", "performance"],
    }))
    assert result["_v"] == "1.0"
    assert result["type"] == "solution"
    assert result["confidence"] == "high"
    assert result["score"] == 1.0
    assert "id" in result
    assert result["decay_rate"] > 0


async def test_kn_save_low_confidence(client: Client):
    result = _data(await client.call_tool("kn_save", {
        "content": "Maybe try Redis for caching",
        "type": "workaround",
        "confidence": "low",
    }))
    assert result["confidence"] == "low"
    # Low confidence should decay 4x faster
    assert result["decay_rate"] > 0


async def test_kn_save_invalid_type(client: Client):
    result = _data(await client.call_tool("kn_save", {
        "content": "Something",
        "type": "invalid_type",
    }))
    assert "error" in result


async def test_kn_save_empty_content(client: Client):
    result = _data(await client.call_tool("kn_save", {
        "content": "",
        "type": "solution",
    }))
    assert "error" in result


async def test_kn_memories_search(client: Client):
    await client.call_tool("kn_save", {
        "content": "Always validate JWT tokens on the server side",
        "type": "pattern",
        "tags": ["auth"],
    })
    await client.call_tool("kn_save", {
        "content": "SQLite WAL mode improves concurrency",
        "type": "solution",
        "tags": ["database"],
    })

    result = _data(await client.call_tool("kn_memories", {}))
    assert result["_v"] == "1.0"
    assert result["count"] == 2


async def test_kn_memories_empty(client: Client):
    result = _data(await client.call_tool("kn_memories", {}))
    assert result["count"] == 0
    assert result["experiences"] == []


async def test_kn_memories_with_limit(client: Client):
    for i in range(5):
        await client.call_tool("kn_save", {
            "content": f"Experience {i}",
            "type": "solution",
        })

    result = _data(await client.call_tool("kn_memories", {"limit": 3}))
    assert result["count"] == 3


async def test_kn_memories_relevance_fields(client: Client):
    await client.call_tool("kn_save", {
        "content": "Fresh experience",
        "type": "decision",
    })

    result = _data(await client.call_tool("kn_memories", {}))
    exp = result["experiences"][0]
    assert "relevance" in exp
    assert exp["relevance"] > 0.9  # Just created, should be near 1.0


async def test_kn_prune_nothing(client: Client):
    # Fresh experience should not be pruned
    await client.call_tool("kn_save", {
        "content": "Keep me",
        "type": "solution",
    })

    result = _data(await client.call_tool("kn_prune", {}))
    assert result["_v"] == "1.0"
    assert result["pruned_count"] == 0


async def test_kn_prune_empty(client: Client):
    result = _data(await client.call_tool("kn_prune", {}))
    assert result["pruned_count"] == 0
    assert result["pruned_ids"] == []


# ── Idea tool tests ──────────────────────────────────────────


async def test_kn_idea_create(client: Client):
    result = _data(await client.call_tool("kn_idea", {
        "title": "Build a CLI",
        "category": "feature",
        "score": 8.5,
    }))
    assert result["_v"] == "1.0"
    assert result["title"] == "Build a CLI"
    assert result["status"] == "draft"
    assert result["category"] == "feature"
    assert result["score"] == 8.5
    assert "id" in result


async def test_kn_idea_update(client: Client):
    created = _data(await client.call_tool("kn_idea", {"title": "Original"}))

    updated = _data(await client.call_tool("kn_idea", {
        "title": "Updated Title",
        "idea_id": created["id"],
        "status": "evaluating",
    }))
    assert updated["title"] == "Updated Title"
    assert updated["status"] == "evaluating"


async def test_kn_idea_invalid_transition(client: Client):
    created = _data(await client.call_tool("kn_idea", {"title": "Stuck"}))

    # draft -> done is not valid (must go through evaluating, approved, implementing)
    result = _data(await client.call_tool("kn_idea", {
        "title": "Stuck",
        "idea_id": created["id"],
        "status": "done",
    }))
    assert "error" in result


async def test_kn_idea_not_found(client: Client):
    result = _data(await client.call_tool("kn_idea", {
        "title": "Ghost",
        "idea_id": "nonexistent",
    }))
    assert "error" in result


async def test_kn_idea_empty_title(client: Client):
    result = _data(await client.call_tool("kn_idea", {"title": ""}))
    assert "error" in result


async def test_kn_idea_with_link(client: Client):
    # Create a node to link to
    node = _data(await client.call_tool("kn_add", {
        "name": "Auth System",
        "type": "concept",
    }))

    result = _data(await client.call_tool("kn_idea", {
        "title": "Improve Auth",
        "link_to": node["id"],
    }))
    assert result["_v"] == "1.0"
    assert result["title"] == "Improve Auth"
    assert result["linked_to"] == node["id"]


async def test_kn_idea_link_to_nonexistent_node(client: Client):
    result = _data(await client.call_tool("kn_idea", {
        "title": "Orphan Idea",
        "link_to": "nonexistent",
    }))
    assert result["_v"] == "1.0"
    assert result["title"] == "Orphan Idea"
    assert result["linked_to"] is None
    assert "link_error" in result


async def test_kn_ideas_list(client: Client):
    await client.call_tool("kn_idea", {"title": "Idea A", "category": "feature"})
    await client.call_tool("kn_idea", {"title": "Idea B", "category": "bug"})

    result = _data(await client.call_tool("kn_ideas", {}))
    assert result["_v"] == "1.0"
    assert result["count"] == 2


async def test_kn_ideas_filter_by_status(client: Client):
    created = _data(await client.call_tool("kn_idea", {"title": "Advancing"}))
    await client.call_tool("kn_idea", {"title": "Static"})

    # Advance first idea to evaluating
    await client.call_tool("kn_idea", {
        "title": "Advancing",
        "idea_id": created["id"],
        "status": "evaluating",
    })

    result = _data(await client.call_tool("kn_ideas", {"status": "evaluating"}))
    assert result["count"] == 1
    assert result["ideas"][0]["title"] == "Advancing"


async def test_kn_ideas_filter_by_category(client: Client):
    await client.call_tool("kn_idea", {"title": "Feature X", "category": "feature"})
    await client.call_tool("kn_idea", {"title": "Bug Y", "category": "bug"})

    result = _data(await client.call_tool("kn_ideas", {"category": "feature"}))
    assert result["count"] == 1
    assert result["ideas"][0]["title"] == "Feature X"


async def test_kn_ideas_pagination(client: Client):
    for i in range(8):
        await client.call_tool("kn_idea", {"title": f"Idea {i}"})

    result = _data(await client.call_tool("kn_ideas", {"limit": 3}))
    assert result["count"] == 3


async def test_kn_ideas_empty(client: Client):
    result = _data(await client.call_tool("kn_ideas", {}))
    assert result["count"] == 0
    assert result["ideas"] == []


# ── Intelligence tool tests (5 tools) ────────────────────────


async def test_kn_learn_high_confidence(client: Client):
    result = _data(await client.call_tool("kn_learn", {
        "content": "Use JWT for API authentication",
        "type": "decision",
        "confidence": "high",
        "context": "Architecture review",
        "tags": ["auth", "jwt"],
    }))
    assert result["_v"] == "1.0"
    assert result["stored_as"] == "node"
    assert result["node_id"] is not None
    assert result["experience_id"] is not None


async def test_kn_learn_medium_confidence(client: Client):
    result = _data(await client.call_tool("kn_learn", {
        "content": "Redis might be good for caching",
        "type": "pattern",
        "confidence": "medium",
    }))
    assert result["stored_as"] == "experience"
    assert result["node_id"] is None
    assert result["experience_id"] is not None


async def test_kn_learn_low_confidence(client: Client):
    result = _data(await client.call_tool("kn_learn", {
        "content": "Maybe try GraphQL",
        "type": "decision",
        "confidence": "low",
    }))
    assert result["stored_as"] == "experience"
    assert result["node_id"] is None


async def test_kn_learn_invalid_type(client: Client):
    result = _data(await client.call_tool("kn_learn", {
        "content": "Something",
        "type": "invalid_type",
    }))
    assert "error" in result


async def test_kn_learn_empty_content(client: Client):
    result = _data(await client.call_tool("kn_learn", {
        "content": "",
        "type": "decision",
    }))
    assert "error" in result


async def test_kn_recall_basic(client: Client):
    await client.call_tool("kn_learn", {
        "content": "Token bucket algorithm for rate limiting",
        "type": "solution",
        "confidence": "high",
    })

    result = _data(await client.call_tool("kn_recall", {
        "topic": "rate limiting",
    }))
    assert result["_v"] == "1.0"
    assert result["count"] >= 1


async def test_kn_recall_empty_topic(client: Client):
    await client.call_tool("kn_learn", {
        "content": "Testing is important",
        "type": "pattern",
        "confidence": "high",
    })

    result = _data(await client.call_tool("kn_recall", {}))
    assert result["_v"] == "1.0"
    assert result["count"] >= 1


async def test_kn_recall_no_results(client: Client):
    result = _data(await client.call_tool("kn_recall", {
        "topic": "nonexistent_xyz_abc_123",
    }))
    assert result["count"] == 0


async def test_kn_crossref_basic(client: Client):
    await client.call_tool("kn_learn", {
        "content": "Implemented rate limiting with Redis",
        "type": "solution",
        "confidence": "high",
    })

    result = _data(await client.call_tool("kn_crossref", {
        "problem": "Need rate limiting for API endpoints",
    }))
    assert result["_v"] == "1.0"
    assert result["count"] >= 1


async def test_kn_crossref_empty_problem(client: Client):
    result = _data(await client.call_tool("kn_crossref", {
        "problem": "",
    }))
    assert "error" in result


async def test_kn_context_basic(client: Client):
    await client.call_tool("kn_learn", {
        "content": "FastAPI uses Pydantic for validation",
        "type": "pattern",
        "confidence": "high",
    })

    result = _data(await client.call_tool("kn_context", {
        "keywords": "FastAPI validation",
    }))
    assert result["_v"] == "1.0"
    assert "nodes" in result
    assert "experiences" in result


async def test_kn_context_empty_keywords(client: Client):
    result = _data(await client.call_tool("kn_context", {
        "keywords": "",
    }))
    assert result["count"] == 0


async def test_kn_context_detail_levels(client: Client):
    await client.call_tool("kn_learn", {
        "content": "SQLite FTS5 for search",
        "type": "pattern",
        "confidence": "high",
    })

    summary = _data(await client.call_tool("kn_context", {
        "keywords": "SQLite search",
        "detail": "summary",
    }))
    full = _data(await client.call_tool("kn_context", {
        "keywords": "SQLite search",
        "detail": "full",
    }))
    assert summary["_v"] == "1.0"
    assert full["_v"] == "1.0"


async def test_kn_related_basic(client: Client):
    n1 = _data(await client.call_tool("kn_add", {
        "name": "Authentication",
        "type": "concept",
    }))
    n2 = _data(await client.call_tool("kn_add", {
        "name": "JWT Tokens",
        "type": "pattern",
    }))
    await client.call_tool("kn_connect", {
        "source_id": n1["id"],
        "target_id": n2["id"],
        "edge_type": "uses",
    })

    result = _data(await client.call_tool("kn_related", {
        "node_id": n1["id"],
        "depth": 1,
    }))
    assert result["_v"] == "1.0"
    assert result["count"] >= 1


async def test_kn_related_empty_node_id(client: Client):
    result = _data(await client.call_tool("kn_related", {
        "node_id": "",
    }))
    assert "error" in result


async def test_kn_related_nonexistent(client: Client):
    result = _data(await client.call_tool("kn_related", {
        "node_id": "nonexistent_id",
    }))
    assert result["count"] == 0


async def test_learn_then_recall_workflow(client: Client):
    """End-to-end: learn something, then recall it."""
    await client.call_tool("kn_learn", {
        "content": "Always use parameterized queries to prevent SQL injection",
        "type": "pattern",
        "confidence": "high",
        "tags": ["security", "sql"],
    })

    result = _data(await client.call_tool("kn_recall", {
        "topic": "SQL injection prevention",
    }))
    assert result["count"] >= 1


async def test_learn_then_crossref_workflow(client: Client):
    """End-to-end: learn a solution, then crossref a similar problem."""
    await client.call_tool("kn_learn", {
        "content": "Circuit breaker pattern for external API resilience",
        "type": "solution",
        "confidence": "high",
    })

    result = _data(await client.call_tool("kn_crossref", {
        "problem": "External API keeps failing, need resilience pattern",
    }))
    assert result["count"] >= 1


# ── Resource tests ───────────────────────────────────────────


async def test_list_resources(client: Client):
    resources = await client.list_resources()
    uris = {str(r.uri) for r in resources}
    assert "kn://status" in uris
    assert "kn://projects" in uris
    assert "kn://memories" in uris
    assert len(uris) == 3


def _res(content) -> dict:
    """Extract parsed JSON from resource read result (list of content items)."""
    return json.loads(content[0].text if isinstance(content, list) else content)


async def test_resource_status_empty(client: Client):
    content = await client.read_resource("kn://status")
    data = _res(content)
    assert data["_v"] == "1.0"
    assert "graph" in data
    assert data["active_project"] is None


async def test_resource_status_with_project(client: Client):
    p = _data(await client.call_tool("kn_project", {"name": "Res Test"}))
    await client.call_tool("kn_projects", {"set_active": p["id"]})

    content = await client.read_resource("kn://status")
    data = _res(content)
    assert data["active_project"] is not None
    assert data["active_project"]["name"] == "Res Test"


async def test_resource_projects(client: Client):
    p = _data(await client.call_tool("kn_project", {"name": "Listed"}))
    await client.call_tool("kn_log", {
        "project_id": p["id"],
        "action": "Setup done",
    })

    content = await client.read_resource("kn://projects")
    data = _res(content)
    assert data["count"] == 1
    assert data["projects"][0]["name"] == "Listed"
    assert len(data["projects"][0]["recent_progress"]) == 1


async def test_resource_projects_empty(client: Client):
    content = await client.read_resource("kn://projects")
    data = _res(content)
    assert data["count"] == 0


async def test_resource_memories(client: Client):
    await client.call_tool("kn_save", {
        "content": "Resource test memory",
        "type": "solution",
    })

    content = await client.read_resource("kn://memories")
    data = _res(content)
    assert data["count"] == 1
    assert "Resource test" in data["experiences"][0]["content"]


async def test_resource_memories_empty(client: Client):
    content = await client.read_resource("kn://memories")
    data = _res(content)
    assert data["count"] == 0


# ── Prompt tests ─────────────────────────────────────────────


async def test_list_prompts(client: Client):
    prompts = await client.list_prompts()
    names = {p.name for p in prompts}
    assert "kn_bootup" in names
    assert "kn_review" in names
    assert len(names) == 2


async def test_prompt_bootup_empty(client: Client):
    result = await client.get_prompt("kn_bootup")
    text = result.messages[0].content.text
    assert "Kairn Session Context" in text
    assert "No active project" in text


async def test_prompt_bootup_with_project(client: Client):
    p = _data(await client.call_tool("kn_project", {"name": "Boot Project"}))
    await client.call_tool("kn_projects", {"set_active": p["id"]})
    await client.call_tool("kn_log", {
        "project_id": p["id"],
        "action": "Initial setup",
    })

    result = await client.get_prompt("kn_bootup")
    text = result.messages[0].content.text
    assert "Boot Project" in text
    assert "Initial setup" in text


async def test_prompt_review_empty(client: Client):
    result = await client.get_prompt("kn_review")
    text = result.messages[0].content.text
    assert "Session Review" in text
    assert "No active project" in text


async def test_prompt_review_with_progress(client: Client):
    p = _data(await client.call_tool("kn_project", {"name": "Review Project"}))
    await client.call_tool("kn_projects", {"set_active": p["id"]})
    await client.call_tool("kn_log", {
        "project_id": p["id"],
        "action": "DB migration failed",
        "type": "failure",
        "result": "Schema mismatch",
    })
    # Log progress last so most recent entry has next_step
    await client.call_tool("kn_log", {
        "project_id": p["id"],
        "action": "Added auth module",
        "result": "Working",
        "next_step": "Add tests",
    })

    result = await client.get_prompt("kn_review")
    text = result.messages[0].content.text
    assert "Review Project" in text
    assert "Added auth module" in text
    assert "DB migration failed" in text
    assert "Add tests" in text
