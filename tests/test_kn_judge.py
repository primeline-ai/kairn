"""Tests for the kn_judge MCP tool + `kairn judge` CLI subcommand.

Phase 3 of `.claude/plans/2026-05-26-kairn-judgment-envelope-and-doctor.md`.

`kn_judge` is the strict-mode wrapper around kn_connect. It only
accepts the 5 canonical relation verbs (Phase 1's RELATION_VERBS) and
is the only path that should produce judgment edges in production
(legacy / system-generated edges continue via the lax-mode kn_connect).
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner
from fastmcp import Client

from kairn.cli import main as cli_main
from kairn.server import create_server


@pytest.fixture
async def client(tmp_path):
    """MCP client wired to a per-test SQLite workspace."""
    server = create_server(str(tmp_path / "judge.db"))
    async with Client(server) as c:
        yield c


def _data(call_result) -> dict:
    """Extract JSON payload from an MCP CallToolResult."""
    text = call_result.content[0].text
    return json.loads(text)


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kn_judge_happy_path_persists_canonical_edge(client: Client) -> None:
    """A judgment with a canonical verb persists with strict mode + provenance."""
    a = _data(await client.call_tool(
        "kn_learn",
        {"content": "Source for judgment test A", "type": "decision", "confidence": "high"},
    ))
    b = _data(await client.call_tool(
        "kn_learn",
        {"content": "Target for judgment test B", "type": "decision", "confidence": "high"},
    ))

    result = _data(await client.call_tool(
        "kn_judge",
        {
            "source_id": a["node_id"],
            "target_id": b["node_id"],
            "relation": "supersedes",
            "reason": "B replaces A per session 2026-05-26",
            "confidence": 0.9,
        },
    ))

    assert result["_v"] == "1.0"
    assert result["source_id"] == a["node_id"]
    assert result["target_id"] == b["node_id"]
    assert result["type"] == "supersedes"
    assert result["weight"] == 0.9
    assert result["created_by"] == "kn_judge"
    # Reason persisted into edge.properties.
    assert "reason" in (result.get("properties") or {})


# ----------------------------------------------------------------------
# Strict-mode rejection of legacy / unknown verbs
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kn_judge_rejects_legacy_verb(client: Client) -> None:
    """Phase 1 strict mode kicks in: legacy `auto_related` raises ValueError."""
    a = _data(await client.call_tool(
        "kn_learn", {"content": "A node strict reject", "type": "decision", "confidence": "high"},
    ))
    b = _data(await client.call_tool(
        "kn_learn", {"content": "B node strict reject", "type": "decision", "confidence": "high"},
    ))

    result = _data(await client.call_tool(
        "kn_judge",
        {
            "source_id": a["node_id"],
            "target_id": b["node_id"],
            "relation": "auto_related",
        },
    ))
    assert "error" in result
    assert "Invalid relation verb" in result["error"]


@pytest.mark.asyncio
async def test_kn_judge_rejects_empty_relation(client: Client) -> None:
    """Empty / whitespace relation is rejected with a clear error."""
    a = _data(await client.call_tool(
        "kn_learn", {"content": "A node empty rel", "type": "decision", "confidence": "high"},
    ))
    b = _data(await client.call_tool(
        "kn_learn", {"content": "B node empty rel", "type": "decision", "confidence": "high"},
    ))

    result = _data(await client.call_tool(
        "kn_judge",
        {"source_id": a["node_id"], "target_id": b["node_id"], "relation": "  "},
    ))
    assert "error" in result
    assert "relation is required" in result["error"]


# ----------------------------------------------------------------------
# Missing source / target
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kn_judge_missing_source_returns_error(client: Client) -> None:
    """Non-existent source_id surfaces as an error envelope, not a tool exception."""
    b = _data(await client.call_tool(
        "kn_learn", {"content": "B node missing src", "type": "decision", "confidence": "high"},
    ))

    result = _data(await client.call_tool(
        "kn_judge",
        {
            "source_id": "does-not-exist",
            "target_id": b["node_id"],
            "relation": "related",
        },
    ))
    assert "error" in result
    assert "Source node not found" in result["error"]


@pytest.mark.asyncio
async def test_kn_judge_missing_target_returns_error(client: Client) -> None:
    """Non-existent target_id surfaces as an error envelope."""
    a = _data(await client.call_tool(
        "kn_learn", {"content": "A node missing tgt", "type": "decision", "confidence": "high"},
    ))

    result = _data(await client.call_tool(
        "kn_judge",
        {
            "source_id": a["node_id"],
            "target_id": "does-not-exist",
            "relation": "related",
        },
    ))
    assert "error" in result
    assert "Target node not found" in result["error"]


# ----------------------------------------------------------------------
# Idempotency on duplicate (same source / target / relation)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kn_judge_duplicate_same_verb_raises_integrity_error(client: Client) -> None:
    """Same (source, target, relation) triple is the edge primary key.

    Second call must error rather than silently create a phantom duplicate.
    The error envelope is the right surface (vs a tool exception); callers
    that need upsert semantics can disconnect-then-judge.
    """
    a = _data(await client.call_tool(
        "kn_learn", {"content": "A dup", "type": "decision", "confidence": "high"},
    ))
    b = _data(await client.call_tool(
        "kn_learn", {"content": "B dup", "type": "decision", "confidence": "high"},
    ))

    first = _data(await client.call_tool(
        "kn_judge",
        {"source_id": a["node_id"], "target_id": b["node_id"], "relation": "compatible"},
    ))
    assert first["type"] == "compatible"

    second = _data(await client.call_tool(
        "kn_judge",
        {"source_id": a["node_id"], "target_id": b["node_id"], "relation": "compatible"},
    ))
    # SQLite UNIQUE PK violation surfaces as an error envelope - the
    # MCP tool catches sqlite3.IntegrityError and converts it (rather
    # than letting it become a tool exception, which would force every
    # caller to handle dual surfaces). Caller-driven upsert is out of
    # scope for Phase 3.
    assert "error" in second
    assert "duplicate" in second["error"].lower()


@pytest.mark.asyncio
async def test_kn_judge_accepts_case_insensitive_relation(client: Client) -> None:
    """MCP path must match CLI Choice(case_sensitive=False) - capitalized
    verbs work end-to-end. Prevents the cross-surface inconsistency
    where `kairn judge --relation Compatible` succeeds but
    kn_judge(relation="Compatible") errors."""
    a = _data(await client.call_tool(
        "kn_learn", {"content": "A case", "type": "decision", "confidence": "high"},
    ))
    b = _data(await client.call_tool(
        "kn_learn", {"content": "B case", "type": "decision", "confidence": "high"},
    ))
    result = _data(await client.call_tool(
        "kn_judge",
        {
            "source_id": a["node_id"],
            "target_id": b["node_id"],
            "relation": " Compatible ",  # leading/trailing space + capital letter
        },
    ))
    assert result["type"] == "compatible"


@pytest.mark.asyncio
async def test_kn_judge_different_verbs_same_pair_coexist(client: Client) -> None:
    """Same (source, target) with DIFFERENT verbs is allowed (composite PK)."""
    a = _data(await client.call_tool(
        "kn_learn", {"content": "A multi-verb", "type": "decision", "confidence": "high"},
    ))
    b = _data(await client.call_tool(
        "kn_learn", {"content": "B multi-verb", "type": "decision", "confidence": "high"},
    ))

    r1 = _data(await client.call_tool(
        "kn_judge",
        {"source_id": a["node_id"], "target_id": b["node_id"], "relation": "related"},
    ))
    r2 = _data(await client.call_tool(
        "kn_judge",
        {"source_id": a["node_id"], "target_id": b["node_id"], "relation": "compatible"},
    ))
    assert r1["type"] == "related"
    assert r2["type"] == "compatible"


# ----------------------------------------------------------------------
# Tool list contract
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kn_judge_appears_in_tool_list(client: Client) -> None:
    """MCP tool list now contains kn_judge (plan SC: 19 -> 20 tools)."""
    tools = await client.list_tools()
    names = {t.name for t in tools}
    assert "kn_judge" in names
    # Sanity: kn_connect still there too (Phase 3 is additive).
    assert "kn_connect" in names


# ----------------------------------------------------------------------
# CLI subcommand
# ----------------------------------------------------------------------


def test_kairn_judge_cli_happy_path(tmp_path) -> None:
    """`kairn judge` subcommand exists and persists a canonical-verb edge."""
    runner = CliRunner()
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()

    # Initialize the workspace DB.
    init_result = runner.invoke(cli_main, ["init", str(ws_dir)])
    assert init_result.exit_code == 0, init_result.output

    # Seed two nodes via CLI `learn`.
    r1 = runner.invoke(cli_main, [
        "learn", str(ws_dir),
        "--content", "Source for CLI judge",
        "--type", "decision",
        "--confidence", "high",
    ])
    assert r1.exit_code == 0, r1.output
    a = json.loads(r1.output)["node_id"]

    r2 = runner.invoke(cli_main, [
        "learn", str(ws_dir),
        "--content", "Target for CLI judge",
        "--type", "decision",
        "--confidence", "high",
    ])
    assert r2.exit_code == 0, r2.output
    b = json.loads(r2.output)["node_id"]

    result = runner.invoke(cli_main, [
        "judge", str(ws_dir),
        "--source-id", a,
        "--target-id", b,
        "--relation", "compatible",
        "--reason", "From the CLI test",
        "--confidence", "0.8",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["type"] == "compatible"
    assert payload["weight"] == 0.8
    assert payload["created_by"] == "kn_judge"


def test_kairn_judge_cli_rejects_legacy_verb(tmp_path) -> None:
    """CLI Choice validation blocks legacy verbs at the click layer."""
    runner = CliRunner()
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    init_result = runner.invoke(cli_main, ["init", str(ws_dir)])
    assert init_result.exit_code == 0, init_result.output
    result = runner.invoke(cli_main, [
        "judge", str(ws_dir),
        "--source-id", "x", "--target-id", "y",
        "--relation", "auto_related",
    ])
    assert result.exit_code != 0
    assert "Invalid value" in result.output or "auto_related" in result.output
