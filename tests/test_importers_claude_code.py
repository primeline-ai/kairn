"""Tests for the `kairn import claude-code` transcript importer.

WOW-9 Phase 3 (2026-07-08, plan: .claude/plans/2026-07-07-kairn-wow9-import.md).

Ships in COARSE mode (one experience per session, session-summary only) per
the plan's FAILED-condition fallback: the Phase 3 precision spike measured
fine-grained per-signal extraction at ~38% junk on a real N=68 holdout, above
the plan's <20% bar (Kairn decision 21985368). Coarse mode is ~100% precision -
the content is a deterministic pair of fields (aiTitle + first genuine user
prompt), not marker-matched prose. Zero-LLM, zero-network throughout; every
stored string is routed through the Phase 2 redaction module.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from kairn.config import Config
from kairn.importers.claude_code import (
    _MAX_CONTENT,
    IMPORT_NAMESPACE,
    SchemaError,
    assert_transcript_schema,
    discover_transcripts,
    extract_session_summary,
    import_claude_code,
)
from kairn.storage.sqlite_store import SQLiteStore

# --- fixtures -------------------------------------------------------------


# A realistic transcript shape (matches the live Claude Code JSONL schema
# probed 2026-07-08): ai-title record, isMeta slash-command expansion, a
# genuine user prompt carrying metadata, an assistant turn with thinking +
# text + a kn_learn tool_use, and a user turn whose content list is a
# tool_result (tool output fed back, NOT the user's words).
def _good_records() -> list[dict]:
    return [
        {
            "type": "ai-title",
            "aiTitle": "Fix the socket leak in server.py",
            "timestamp": "2026-01-02T09:59:00.000Z",
        },
        {
            "type": "user",
            "isMeta": True,
            "timestamp": "2026-01-02T09:59:30.000Z",
            "message": {"role": "user", "content": "# /session-end\nfull ritual ..."},
        },
        {
            "type": "user",
            "timestamp": "2026-01-02T10:00:00.000Z",
            "cwd": "/home/u/proj",
            "gitBranch": "main",
            "sessionId": "sess-abc",
            "message": {
                "role": "user",
                "content": "Help me fix the socket leak in server.py. "
                "My key is AKIAIOSFODNN7EXAMPLE do not commit it.",
            },
        },
        {
            "type": "assistant",
            "timestamp": "2026-01-02T10:00:20.000Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "The leak is the unclosed socket."},
                    {"type": "text", "text": "The root cause is the missing close()."},
                    {
                        "type": "tool_use",
                        "name": "mcp__kairn__kn_learn",
                        "input": {"type": "gotcha", "content": "sockets need explicit close"},
                    },
                ],
            },
        },
        {
            "type": "user",
            "timestamp": "2026-01-02T10:00:25.000Z",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": "secret-in-tool-output SECRETVAL123"},
                ],
            },
        },
        {"type": "file-history-snapshot", "timestamp": "2026-01-02T10:00:30.000Z"},
    ]


def _write_transcript(path: Path, records: list[dict], *, trailing_garbage: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
        if trailing_garbage:
            fh.write("this is not valid json{{{\n")
            fh.write("\n")  # blank line


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(workspace_path=tmp_path)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A transcript root laid out like ~/.claude/projects/<proj>/<uuid>.jsonl."""
    r = tmp_path / "projects"
    _write_transcript(r / "-home-u-proj" / "sess-abc.jsonl", _good_records())
    return r


async def _imported(store: SQLiteStore) -> list[dict]:
    return await store.query_experiences_since("2000-01-01T00:00:00", namespace=IMPORT_NAMESPACE)


# --- extract_session_summary ---------------------------------------------


def test_extract_uses_aititle_and_first_genuine_prompt():
    summary = extract_session_summary(_good_records(), source_ref="cc:x")
    assert summary is not None
    assert "Fix the socket leak in server.py" in summary["content"]
    assert "Help me fix the socket leak" in summary["content"]


def test_extract_skips_ismeta_and_tool_result_and_assistant():
    summary = extract_session_summary(_good_records(), source_ref="cc:x")
    # isMeta slash-command expansion must not be the chosen prompt
    assert "/session-end" not in summary["content"]
    # tool_result (tool output, appears in a user turn) is not user prose
    assert "secret-in-tool-output" not in summary["content"]
    # assistant text/thinking/tool_use are not part of the coarse summary
    assert "root cause is the missing" not in summary["content"]


def test_extract_captures_metadata():
    summary = extract_session_summary(_good_records(), source_ref="cc:x")
    assert summary["cwd"] == "/home/u/proj"
    assert summary["gitBranch"] == "main"
    assert summary["created_at"].startswith("2026-01-02T09:59:00")
    assert summary["created_at"].endswith("+00:00")  # normalized to UTC


def test_extract_returns_none_when_no_genuine_user_prose():
    records = [
        {
            "type": "user",
            "isMeta": True,
            "timestamp": "2026-01-02T10:00:00.000Z",
            "message": {"role": "user", "content": "# /auto-run"},
        },
        {
            "type": "assistant",
            "timestamp": "2026-01-02T10:00:20.000Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        },
    ]
    assert extract_session_summary(records, source_ref="cc:x") is None


# --- assert_transcript_schema (drift canary) -----------------------------


def test_schema_check_passes_on_good_shape():
    assert_transcript_schema(_good_records())  # must not raise


def test_schema_check_raises_when_core_fields_missing():
    # Simulate a drifted release: message.role renamed, no timestamps.
    drifted = [
        {"kind": "user", "msg": {"speaker": "user", "body": "hi"}},
        {"kind": "assistant", "msg": {"speaker": "assistant", "body": "yo"}},
        {"kind": "user", "msg": {"speaker": "user", "body": "again"}},
    ]
    with pytest.raises(SchemaError):
        assert_transcript_schema(drifted)


def test_schema_check_tolerates_empty_sample():
    # Nothing to import - must not raise (no false drift alarm).
    assert_transcript_schema([])


# --- discover_transcripts ------------------------------------------------


def test_discover_finds_session_files_excludes_subagents(tmp_path: Path):
    r = tmp_path / "projects"
    _write_transcript(r / "-proj-a" / "s1.jsonl", _good_records())
    _write_transcript(r / "-proj-b" / "s2.jsonl", _good_records())
    _write_transcript(r / "-proj-a" / "s1" / "subagents" / "agent-x.jsonl", _good_records())
    found = discover_transcripts([r])
    names = {p.name for p in found}
    assert names == {"s1.jsonl", "s2.jsonl"}  # subagent transcript excluded


def test_discover_scans_multiple_roots(tmp_path: Path):
    r1 = tmp_path / "primary"
    r2 = tmp_path / "secondary"
    _write_transcript(r1 / "-proj" / "s1.jsonl", _good_records())
    _write_transcript(r2 / "-proj" / "s2.jsonl", _good_records())
    found = discover_transcripts([r1, r2])
    assert len(found) == 2


# --- import_claude_code --------------------------------------------------


async def test_import_writes_one_experience_per_session(
    root: Path, store: SQLiteStore, config: Config
):
    result = await import_claude_code(store, [root], config=config)
    assert result["imported"] == 1
    experiences = await _imported(store)
    assert len(experiences) == 1
    assert experiences[0]["namespace"] == IMPORT_NAMESPACE


async def test_import_redacts_secrets_in_content(root: Path, store: SQLiteStore, config: Config):
    await import_claude_code(store, [root], config=config)
    exp = (await _imported(store))[0]
    assert "AKIAIOSFODNN7EXAMPLE" not in exp["content"]
    assert "[REDACTED:" in exp["content"]


async def test_import_sets_source_ref_and_metadata(root: Path, store: SQLiteStore, config: Config):
    await import_claude_code(store, [root], config=config)
    exp = (await _imported(store))[0]
    assert exp["properties"]["source_ref"].startswith("claude-code:")
    assert exp["created_at"].startswith("2026-01-02")


async def test_import_is_idempotent(root: Path, store: SQLiteStore, config: Config):
    first = await import_claude_code(store, [root], config=config)
    second = await import_claude_code(store, [root], config=config)
    assert first["imported"] == 1
    assert second["imported"] == 0
    assert second["skipped_duplicate"] == 1
    assert len(await _imported(store)) == 1


async def test_import_dry_run_writes_nothing(root: Path, store: SQLiteStore, config: Config):
    result = await import_claude_code(store, [root], config=config, dry_run=True)
    assert result["dry_run"] is True
    assert result["imported"] == 1
    assert len(await _imported(store)) == 0


async def test_import_skips_sessions_without_prose(
    tmp_path: Path, store: SQLiteStore, config: Config
):
    r = tmp_path / "projects"
    _write_transcript(
        r / "-proj" / "empty.jsonl",
        [
            {
                "type": "user",
                "isMeta": True,
                "timestamp": "2026-01-02T10:00:00.000Z",
                "message": {"role": "user", "content": "# /auto-run"},
            }
        ],
    )
    result = await import_claude_code(store, [r], config=config)
    assert result["imported"] == 0
    assert result["skipped_empty"] == 1


async def test_import_defensive_against_malformed_lines(
    tmp_path: Path, store: SQLiteStore, config: Config
):
    r = tmp_path / "projects"
    _write_transcript(r / "-proj" / "s.jsonl", _good_records(), trailing_garbage=True)
    # Malformed JSONL lines are skipped, not fatal.
    result = await import_claude_code(store, [r], config=config)
    assert result["imported"] == 1


async def test_import_since_filters_older_sessions(
    tmp_path: Path, store: SQLiteStore, config: Config
):
    r = tmp_path / "projects"
    old = _good_records()
    for rec in old:
        if "timestamp" in rec:
            rec["timestamp"] = "2025-06-01T10:00:00.000Z"
    _write_transcript(r / "-proj" / "old.jsonl", old)
    _write_transcript(r / "-proj" / "new.jsonl", _good_records())

    result = await import_claude_code(store, [r], config=config, since="2026-01-01")
    assert result["imported"] == 1  # only the 2026 session


# --- RC-gate regression tests --------------------------------------------


async def test_import_redacts_secret_straddling_truncation_boundary(
    tmp_path: Path, store: SQLiteStore, config: Config
):
    """RC-gate BLOCKER: a secret must be redacted at FULL length BEFORE the
    _MAX_CONTENT truncation. Otherwise a secret straddling the cut is shortened
    below its redaction rule's minimum match length, isn't matched, and its raw
    tail is stored."""
    r = tmp_path / "projects"
    key = "AKIAIOSFODNN7EXAMPLE"  # AKIA + exactly 16 chars; the rule needs all 16
    # "x " * 395 = 790 chars ending in a space, so the key sits at a word boundary
    # (the redaction rule ignores mid-identifier matches) AND straddles char 800.
    long_prompt = "x " * 395 + key + "b" * 60
    _write_transcript(
        r / "-proj" / "s.jsonl",
        [
            {
                "type": "user",
                "timestamp": "2026-01-02T10:00:00.000Z",
                "message": {"role": "user", "content": long_prompt},
            }
        ],
    )
    result = await import_claude_code(store, [r], config=config)
    assert result["redactions"] >= 1
    exp = (await _imported(store))[0]
    assert key not in exp["content"]
    assert "AKIAIOSFO" not in exp["content"]  # not even a truncated fragment survives
    assert len(exp["content"]) <= _MAX_CONTENT + len(" ...")  # truncation still applied


def test_schema_error_is_valueerror():
    """RC-gate: SchemaError must subclass ValueError so the CLI's _run_json emits
    a clean JSON error envelope on real drift, not a raw traceback."""
    assert issubclass(SchemaError, ValueError)


async def test_import_schema_canary_checks_newest_file(
    tmp_path: Path, store: SQLiteStore, config: Config
):
    """RC-gate: the drift canary must sample the NEWEST transcript (where a CC
    release's schema change surfaces first), not the path-first (old) one."""
    r = tmp_path / "projects"
    old = r / "-proj" / "aaa_old.jsonl"  # sorts FIRST alphabetically (clean shape)
    new = r / "-proj" / "zzz_new.jsonl"  # sorts LAST alphabetically (drifted shape)
    _write_transcript(old, _good_records())
    _write_transcript(
        new,
        [
            {"kind": "user", "msg": {"speaker": "user", "body": "hi"}},
            {"kind": "assistant", "msg": {"speaker": "assistant", "body": "yo"}},
        ],
    )
    # Make the drifted file the most recently modified.
    os.utime(old, (1_600_000_000, 1_600_000_000))
    os.utime(new, (1_700_000_000, 1_700_000_000))
    with pytest.raises(SchemaError):
        await import_claude_code(store, [r], config=config)
