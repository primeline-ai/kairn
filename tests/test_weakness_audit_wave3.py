"""Regression tests for the 2026-07-11 weakness-audit Wave-3 fixes.

Covers: the route-index lost-update race (rank 14), soft-deleted ids starving
route() result slots (rank 62), git-import redaction (rank 16), JWT fallback
secret (rank 37), RBAC honesty tripwire (rank 17), namespace present in the
recall/context/crossref item shapes (downstream namespace-based access filters
depend on it), config bool coercion (rank 61), idea status-transition CAS
(rank 65), and the CLI version source (rank 96).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from kairn.core.experience import ExperienceEngine
from kairn.core.graph import GraphEngine
from kairn.core.ideas import IdeaEngine
from kairn.core.intelligence import IntelligenceLayer
from kairn.core.memory import ProjectMemory
from kairn.core.router import ContextRouter
from kairn.events.bus import EventBus


@pytest.fixture
async def engines(store):
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
    return {
        "store": store,
        "graph": graph,
        "router": router,
        "experience": experience,
        "ideas": ideas,
        "intel": intel,
    }


# ---------------------------------------------------------------------------
# rank 14 - route-index lost update under concurrent writers
# ---------------------------------------------------------------------------


async def test_concurrent_route_updates_lose_no_node_ids(engines):
    """N concurrent update_routes_for_node calls sharing one keyword must all
    land in the route's node_ids. The pre-fix read-modify-write interleaves at
    its await points under gather and drops ids (last-write-wins)."""
    router = engines["router"]
    store = engines["store"]
    n = 24
    node_ids = [f"node-{i:04d}" for i in range(n)]

    await asyncio.gather(
        *(
            router.update_routes_for_node(nid, f"gearbox telemetry {nid}", None)
            for nid in node_ids
        )
    )

    routes = await store.get_routes(["gearbox"])
    assert routes, "route for shared keyword must exist"
    stored = routes[0]["node_ids"]
    if isinstance(stored, str):
        stored = json.loads(stored)
    missing = [nid for nid in node_ids if nid not in stored]
    assert not missing, f"lost {len(missing)}/{n} node_ids in route index: {missing[:5]}"


# ---------------------------------------------------------------------------
# rank 62 - soft-deleted nodes must not starve route() result slots
# ---------------------------------------------------------------------------


async def test_soft_deleted_nodes_do_not_starve_route_limit(engines):
    graph = engines["graph"]
    router = engines["router"]

    # Routes are populated by callers of update_routes_for_node (the MCP
    # server's kn_learn path), not by add_node itself - mimic that here.
    live = []
    for i in range(4):
        node = await graph.add_node(
            name=f"turbine blade inspection pattern {i}",
            type="learned_pattern",
        )
        await router.update_routes_for_node(node.id, node.name, node.description)
        live.append(node)
    doomed = []
    for i in range(4):
        node = await graph.add_node(
            name=f"turbine blade inspection obsolete {i}",
            type="learned_pattern",
        )
        await router.update_routes_for_node(node.id, node.name, node.description)
        doomed.append(node)
    for node in doomed:
        assert await graph.remove_node(node.id)

    results = await router.route("turbine blade inspection", limit=4)
    got_ids = {r["node"]["id"] for r in results}
    assert len(results) == 4, (
        f"soft-deleted ids starved the limit: got {len(results)}/4 results"
    )
    assert got_ids == {n.id for n in live}


# ---------------------------------------------------------------------------
# rank 16 - git importer redacts commit messages
# ---------------------------------------------------------------------------


def _make_repo_with_secret(tmp_path: Path) -> Path:
    import subprocess

    repo = tmp_path / "secretrepo"
    repo.mkdir()
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=env)
    (repo / "a.txt").write_text("x")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True, env=env)
    msg = (
        "fix: rotate service credentials\n\n"
        "old value was password=hunter2-super-secret and the header\n"
        "Authorization: Bearer abc123def456ghi789jkl012mno345pqr678\n"
    )
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=repo, check=True, env=env)
    return repo


async def test_git_import_redacts_commit_messages(store, tmp_path):
    from kairn.importers.git import import_git_repo

    repo = _make_repo_with_secret(tmp_path)
    result = await import_git_repo(store, repo)
    assert result["imported"] == 1

    rows = await store.query_experiences(limit=50)
    joined = " ".join(r["content"] for r in rows)
    assert "hunter2-super-secret" not in joined
    assert "abc123def456ghi789jkl012mno345pqr678" not in joined
    assert "[REDACTED:" in joined
    assert result.get("redactions", 0) >= 1


async def test_git_import_dry_run_preview_is_redacted(store, tmp_path):
    from kairn.importers.git import import_git_repo

    repo = _make_repo_with_secret(tmp_path)
    result = await import_git_repo(store, repo, dry_run=True)
    joined = " ".join(p["content"] for p in result["preview"])
    assert "hunter2-super-secret" not in joined


# ---------------------------------------------------------------------------
# rank 37 - JWT must fail closed without a configured secret
# ---------------------------------------------------------------------------


def test_create_token_requires_env_secret(monkeypatch):
    from kairn.auth import jwt as kjwt

    monkeypatch.delenv("KAIRN_JWT_SECRET", raising=False)
    with pytest.raises(RuntimeError):
        kjwt.create_token("u1", "o1")


def test_create_token_with_env_secret(monkeypatch):
    from kairn.auth import jwt as kjwt

    monkeypatch.setenv("KAIRN_JWT_SECRET", "unit-test-secret")
    token = kjwt.create_token("u1", "o1")
    payload = kjwt.verify_token(token, "unit-test-secret")
    assert payload["sub"] == "u1"


# ---------------------------------------------------------------------------
# rank 17 - RBAC is documented-unenforced; tripwire if someone wires it
# ---------------------------------------------------------------------------


def test_rbac_unenforced_tripwire():
    """permissions.py is not consulted by any execution path (v0.2). Its
    module docstring says so. If this test fails because a caller appeared,
    UPDATE the docstring + this test - do not delete the test."""
    import kairn

    src_root = Path(kairn.__file__).parent
    callers = []
    for py in src_root.rglob("*.py"):
        if py.name == "permissions.py":
            continue
        text = py.read_text(encoding="utf-8")
        for needle in ("can_read(", "can_write(", "can_admin(", "check_permission("):
            if needle in text:
                callers.append((str(py.relative_to(src_root)), needle))
    assert not callers, (
        "RBAC checks now have callers - update auth/permissions.py docstring "
        f"and this tripwire: {callers}"
    )
    from kairn.auth import permissions

    assert "NOT wired" in (permissions.__doc__ or ""), (
        "permissions.py docstring must state the unenforced status"
    )


# ---------------------------------------------------------------------------
# bridge dependency - namespace present in recall/context/crossref shapes
# ---------------------------------------------------------------------------


async def test_recall_context_crossref_items_carry_namespace(engines):
    graph = engines["graph"]
    experience = engines["experience"]
    intel = engines["intel"]

    await graph.add_node(
        name="restricted tenant node",
        type="learned_decision",
        namespace="private-tenant",
        description="nsprobe wave3",
    )
    await experience.save(
        content="nsprobe wave3 experience",
        type="decision",
        namespace="private-tenant",
    )

    recall = await intel.recall(topic="nsprobe wave3", limit=10)
    assert recall, "recall returned nothing for planted content"
    assert all("namespace" in item for item in recall), (
        f"recall items missing namespace: {sorted(recall[0].keys())}"
    )
    assert any(item["namespace"] == "private-tenant" for item in recall)

    ctx = await intel.context(keywords="nsprobe wave3", limit=10)
    for item in ctx["nodes"] + ctx["experiences"]:
        assert "namespace" in item, f"context item missing namespace: {sorted(item.keys())}"

    xref = await intel.crossref(problem="nsprobe wave3")
    assert xref, "crossref returned nothing for planted content"
    assert all("namespace" in item for item in xref)


# ---------------------------------------------------------------------------
# rank 61 - config bool coercion ('false' string must not become True)
# ---------------------------------------------------------------------------


def test_config_yaml_bool_string_false(tmp_path, monkeypatch):
    from kairn.config import Config

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "config.yaml").write_text('wal_mode: "false"\nfts5_enabled: "0"\n')
    monkeypatch.setenv("KAIRN_WORKSPACE", str(ws))
    cfg = Config.load()
    assert cfg.wal_mode is False
    assert cfg.fts5_enabled is False


# ---------------------------------------------------------------------------
# rank 65 - idea status transition uses compare-and-set
# ---------------------------------------------------------------------------


async def test_idea_status_update_is_cas_guarded(engines):
    ideas = engines["ideas"]
    store = engines["store"]
    idea = await ideas.create(title="wave3 cas probe")

    updated = await ideas.update(idea.id, status="evaluating")
    assert updated.status == "evaluating"

    # Simulate the race: the row status changed after the engine read its
    # snapshot. The CAS-guarded store update must refuse the stale write.
    stale_won = await store.update_idea(
        idea.id, {"status": "approved"}, expected_status="draft"
    )
    assert not stale_won, "stale status transition must not win"
    current = await ideas.get(idea.id)
    assert current.status == "evaluating"


# ---------------------------------------------------------------------------
# rc-wave3 round-1 CRITICAL - cli `join` must not verify against a hardcoded
# fallback secret when KAIRN_JWT_SECRET is unset
# ---------------------------------------------------------------------------


def test_cli_join_fails_closed_without_jwt_secret(monkeypatch, tmp_path):
    from click.testing import CliRunner

    from kairn.cli import main as cli_main

    monkeypatch.delenv("KAIRN_JWT_SECRET", raising=False)
    monkeypatch.setenv("KAIRN_WORKSPACE", str(tmp_path))
    # A token forged with the old public constant must NOT authenticate.
    import jwt as pyjwt

    forged = pyjwt.encode(
        {"sub": "admin", "org": "x", "exp": 9999999999},
        "test-secret-key-do-not-use",
        algorithm="HS256",
    )
    result = CliRunner().invoke(cli_main, ["workspace", "join", "ws1", "--token", forged])
    assert result.exit_code != 0
    assert "test-secret-key-do-not-use" not in result.output
    # It must refuse for lack of a configured secret, not accept the forgery.
    assert "KAIRN_JWT_SECRET" in result.output or "secret" in result.output.lower()


# ---------------------------------------------------------------------------
# rc-wave3 / rc-wave2b round-1 MEDIUM - kn_related item shape must carry
# namespace too (the 4th kn_* tool the first pass missed)
# ---------------------------------------------------------------------------


async def test_get_related_items_carry_namespace(engines):
    graph = engines["graph"]

    a = await graph.add_node(name="hub node nsprobe related", type="learned_pattern")
    b = await graph.add_node(
        name="restricted neighbor nsprobe related",
        type="learned_decision",
        namespace="private-tenant",
    )
    await graph.connect(source_id=a.id, target_id=b.id, edge_type="related")

    related = await graph.get_related(a.id, depth=1)
    assert related, "expected the neighbor via BFS"
    for item in related:
        assert "namespace" in item["node"], (
            f"kn_related node dict missing namespace: {sorted(item['node'].keys())}"
        )
    assert any(item["node"]["namespace"] == "private-tenant" for item in related)


# ---------------------------------------------------------------------------
# rc-wave3 round-1 HIGH - merge_route_node_id must not swallow a real
# database-locked OperationalError as if it were malformed JSON
# ---------------------------------------------------------------------------


async def test_merge_route_lock_error_propagates_but_malformed_json_is_skipped(store):
    import sqlite3

    # Malformed JSON in a route -> logged + skipped, never crashes.
    await store.db.execute(
        "INSERT INTO routes (keyword, node_ids, confidence) VALUES (?, ?, ?)",
        ("corrupt", "not-json", 0.5),
    )
    await store.db.commit()
    await store.merge_route_node_id("corrupt", "node-x")  # must not raise

    # A genuine 'database is locked' OperationalError must propagate, not be
    # mislabeled as corrupted JSON and silently swallowed (the exact merge-loss
    # the fix targets, just moved to a lock timeout).
    orig_execute = store.db.execute

    async def _locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    store.db.execute = _locked
    try:
        import pytest as _pytest

        with _pytest.raises(sqlite3.OperationalError, match="locked"):
            await store.merge_route_node_id("anything", "node-y")
    finally:
        store.db.execute = orig_execute


# ---------------------------------------------------------------------------
# rc-wave3 round-2 - the namespace invariant must hold across ALL kn_* MCP
# tools that return node/experience data (kn_query + kn_memories were the two
# remaining holes in the same wall as kn_related)
# ---------------------------------------------------------------------------


async def test_kn_query_and_kn_memories_shapes_carry_namespace(tmp_path):
    from fastmcp import Client

    from kairn.server import create_server

    server = create_server(str(tmp_path / "srv.db"))
    async with Client(server) as c:
        await c.call_tool(
            "kn_add",
            {"name": "nsprobe node", "type": "concept", "namespace": "private-tenant"},
        )
        # kn_save has no namespace param (defaults to 'knowledge'); the shape
        # must still echo whatever namespace the experience carries.
        await c.call_tool(
            "kn_save",
            {"content": "nsprobe experience", "type": "pattern"},
        )

        q = await c.call_tool("kn_query", {"text": "nsprobe"})
        q_text = "".join(getattr(b, "text", "") for b in q.content)
        q_data = json.loads(q_text)
        assert q_data["nodes"], "kn_query returned no nodes for planted content"
        for node in q_data["nodes"]:
            assert "namespace" in node, f"kn_query node missing namespace: {sorted(node)}"

        m = await c.call_tool("kn_memories", {"text": "nsprobe"})
        m_text = "".join(getattr(b, "text", "") for b in m.content)
        m_data = json.loads(m_text)
        assert m_data["experiences"], "kn_memories returned nothing"
        for exp in m_data["experiences"]:
            assert "namespace" in exp, f"kn_memories exp missing namespace: {sorted(exp)}"


# ---------------------------------------------------------------------------
# rank 96 - CLI --version reports the source-tree version
# ---------------------------------------------------------------------------


def test_cli_version_matches_source():
    import kairn
    from click.testing import CliRunner

    from kairn.cli import main as cli_main

    result = CliRunner().invoke(cli_main, ["--version"])
    assert result.exit_code == 0
    assert kairn.__version__ in result.output
