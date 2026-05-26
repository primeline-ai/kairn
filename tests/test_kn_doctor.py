"""Tests for the kn_doctor MCP tool + `kairn doctor` CLI subcommand.

Phase 4 of `.claude/plans/2026-05-26-kairn-judgment-envelope-and-doctor.md`.

The diagnostic registry runs 5 read-only health checks:
- check_lock_mode (SQLite WAL + busy_timeout)
- check_fts_index_health (nodes vs nodes_fts parity)
- check_promoted_experience_consistency (sweeper backlog)
- check_namespace_distribution (namespace sprawl detection)
- check_orphan_edges (edges referencing missing nodes)

Each check returns the standard envelope shape; the orchestrator
wraps them in a roll-up summary. MCP and CLI surfaces emit the same
JSON (envelope parity is a Phase 4 gate criterion).
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner
from fastmcp import Client

from kairn.cli import main as cli_main
from kairn.diagnostic import CHECK_REGISTRY, run_checks
from kairn.events.bus import EventBus
from kairn.server import create_server
from kairn.storage.sqlite_store import SQLiteStore


def _data(call_result) -> dict:
    text = call_result.content[0].text
    return json.loads(text)


@pytest.fixture
async def client(tmp_path):
    server = create_server(str(tmp_path / "doctor.db"))
    async with Client(server) as c:
        yield c


@pytest.fixture
async def store(tmp_path):
    s = SQLiteStore(tmp_path / "doctor_unit.db")
    await s.initialize()
    yield s
    await s.close()


# ----------------------------------------------------------------------
# Registry contract
# ----------------------------------------------------------------------


def test_registry_has_five_checks() -> None:
    """Plan SC: 5 MVP checks registered."""
    assert len(CHECK_REGISTRY) == 5
    expected = {
        "check_lock_mode",
        "check_fts_index_health",
        "check_promoted_experience_consistency",
        "check_namespace_distribution",
        "check_orphan_edges",
    }
    assert set(CHECK_REGISTRY) == expected


def test_each_check_id_matches_function_name() -> None:
    """Stability invariant: dict key equals __name__ so callers can
    pass the function name as a string to --check / only=."""
    for check_id, fn in CHECK_REGISTRY.items():
        assert fn.__name__ == check_id


# ----------------------------------------------------------------------
# Individual checks on a clean (healthy) workspace
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lock_mode_ok_on_clean_db(store: SQLiteStore) -> None:
    envelope = await CHECK_REGISTRY["check_lock_mode"](store)
    assert envelope["status"] == "ok"
    assert "wal" in envelope["evidence"].lower()


@pytest.mark.asyncio
async def test_fts_index_health_ok_on_clean_db(store: SQLiteStore) -> None:
    envelope = await CHECK_REGISTRY["check_fts_index_health"](store)
    assert envelope["status"] == "ok"


@pytest.mark.asyncio
async def test_orphan_edges_ok_on_clean_db(store: SQLiteStore) -> None:
    envelope = await CHECK_REGISTRY["check_orphan_edges"](store)
    assert envelope["status"] == "ok"
    assert "orphans=0" in envelope["evidence"]


@pytest.mark.asyncio
async def test_namespace_distribution_ok_on_clean_db(store: SQLiteStore) -> None:
    envelope = await CHECK_REGISTRY["check_namespace_distribution"](store)
    assert envelope["status"] == "ok"


@pytest.mark.asyncio
async def test_promoted_experience_consistency_ok_on_clean_db(
    store: SQLiteStore,
) -> None:
    envelope = await CHECK_REGISTRY["check_promoted_experience_consistency"](store)
    assert envelope["status"] == "ok"


# ----------------------------------------------------------------------
# Synthetic broken state detection
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fts_index_health_ok_after_soft_delete(store: SQLiteStore) -> None:
    """Soft-deleted nodes stay in nodes_fts via the `nodes_fts_au` UPDATE
    trigger (schema/triggers.sql), so the count formula in
    check_fts_index_health must hold: nodes_fts == live_nodes + soft_deleted.
    Closes RC#4 Finding #5: previously the FTS-soft-delete invariant was
    only asserted in a docstring comment, never empirically verified by a
    test. Triggers can be silently dropped or schema-migrations can break
    the invariant; this test pins it down.
    """
    bus = EventBus()
    from kairn.core.graph import GraphEngine
    graph = GraphEngine(store, bus)

    # Seed 3 nodes, soft-delete 1. Formula expects nodes_fts to still
    # contain all 3 rows (the soft-delete is an UPDATE setting deleted_at,
    # not a DELETE, so the trigger keeps the FTS row).
    a = await graph.add_node(name="Survive A", type="concept")
    b = await graph.add_node(name="Survive B", type="concept")
    c = await graph.add_node(name="Soft-delete me", type="concept")
    await graph.remove_node(c.id)

    envelope = await CHECK_REGISTRY["check_fts_index_health"](store)
    assert envelope["status"] == "ok", (
        f"Expected ok but got {envelope['status']}: {envelope['evidence']}"
    )
    # Evidence should show nodes=2 live, deleted=1, fts=3 (formula holds).
    assert "nodes=2" in envelope["evidence"]
    assert "deleted=1" in envelope["evidence"]
    assert "fts=3" in envelope["evidence"]


@pytest.mark.asyncio
async def test_orphan_edges_detects_orphan(store: SQLiteStore) -> None:
    """Insert an edge to a non-existent target; check must warn.

    The orphan check exists for workspaces where FK enforcement was
    disabled or skipped (legacy DBs predating FK constraints, manual
    DELETE bypasses). We simulate that by temporarily disabling FKs
    in the SQLite connection before the synthetic INSERT.
    """
    bus = EventBus()
    from kairn.core.graph import GraphEngine
    graph = GraphEngine(store, bus)
    real = await graph.add_node(name="Real node", type="concept")
    # Disable FK enforcement to simulate the corrupted-state scenario
    # the orphan-edges check is designed to detect.
    await store.db.execute("PRAGMA foreign_keys = OFF")
    try:
        await store.db.execute(
            "INSERT INTO edges (source_id, target_id, type, weight, properties, "
            "created_by, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (real.id, "ghost-node-id", "related", 1.0, None, "test", "2026-05-26"),
        )
        await store.db.commit()
    finally:
        await store.db.execute("PRAGMA foreign_keys = ON")
    envelope = await CHECK_REGISTRY["check_orphan_edges"](store)
    assert envelope["status"] == "warn"
    assert "orphans=1" in envelope["evidence"]


# ----------------------------------------------------------------------
# run_checks orchestrator
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_checks_returns_envelope(store: SQLiteStore) -> None:
    report = await run_checks(store)
    assert report["_v"] == "1.0"
    assert set(report["summary"]) == {"ok", "warn", "fail", "error"}
    assert len(report["checks"]) == 5
    # Clean workspace -> all 5 ok.
    assert report["summary"]["ok"] == 5


@pytest.mark.asyncio
async def test_run_checks_only_filter(store: SQLiteStore) -> None:
    report = await run_checks(store, only="check_lock_mode")
    assert len(report["checks"]) == 1
    assert report["checks"][0]["check_id"] == "check_lock_mode"


@pytest.mark.asyncio
async def test_run_checks_invalid_only_raises(store: SQLiteStore) -> None:
    with pytest.raises(ValueError, match="Unknown check_id"):
        await run_checks(store, only="does_not_exist")


# ----------------------------------------------------------------------
# MCP tool surface
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kn_doctor_appears_in_tool_list(client: Client) -> None:
    """Plan SC: MCP tool list shows kn_doctor (now 21 tools, was 19)."""
    tools = await client.list_tools()
    names = {t.name for t in tools}
    assert "kn_doctor" in names
    assert "kn_judge" in names  # Phase 3 still there
    assert "kn_connect" in names  # legacy still there


@pytest.mark.asyncio
async def test_kn_doctor_full_run_returns_envelope(client: Client) -> None:
    """MCP envelope parity with run_checks helper."""
    result = _data(await client.call_tool("kn_doctor", {}))
    assert result["_v"] == "1.0"
    assert "summary" in result
    assert "checks" in result
    assert len(result["checks"]) == 5


@pytest.mark.asyncio
async def test_kn_doctor_single_check(client: Client) -> None:
    result = _data(await client.call_tool(
        "kn_doctor", {"check": "check_fts_index_health"},
    ))
    assert len(result["checks"]) == 1
    assert result["checks"][0]["check_id"] == "check_fts_index_health"


@pytest.mark.asyncio
async def test_kn_doctor_invalid_check_returns_error(client: Client) -> None:
    result = _data(await client.call_tool(
        "kn_doctor", {"check": "nope"},
    ))
    assert "error" in result
    assert "Unknown check_id" in result["error"]


# ----------------------------------------------------------------------
# CLI surface (envelope parity gate)
# ----------------------------------------------------------------------


def test_kairn_doctor_cli_on_clean_db_exits_zero(tmp_path) -> None:
    runner = CliRunner()
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    init = runner.invoke(cli_main, ["init", str(ws_dir)])
    assert init.exit_code == 0, init.output

    result = runner.invoke(cli_main, ["doctor", str(ws_dir)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    # Envelope parity: identical shape to MCP tool output.
    assert payload["_v"] == "1.0"
    assert "summary" in payload
    assert "checks" in payload
    assert len(payload["checks"]) == 5
