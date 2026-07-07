"""CLI tests for `kairn import claude-code` - WOW-9 Phase 3.

Exercises the CLI as a shell user would (subprocess), matching test_cli_import.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run_kairn(*args: str, check: bool = True, stdin: str | None = None) -> tuple[int, str, str]:
    cmd = [sys.executable, "-m", "kairn.cli", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, input=stdin)
    if check and result.returncode != 0:
        raise AssertionError(
            f"CLI command failed: {' '.join(cmd)}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result.returncode, result.stdout, result.stderr


def _write_transcript(root: Path, project: str, name: str) -> None:
    path = root / project / name
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "type": "ai-title",
            "aiTitle": "Fix the socket leak",
            "timestamp": "2026-01-02T09:59:00.000Z",
        },
        {
            "type": "user",
            "timestamp": "2026-01-02T10:00:00.000Z",
            "cwd": "/home/u/proj",
            "gitBranch": "main",
            "sessionId": name,
            "message": {"role": "user", "content": "Help me fix the socket leak in server.py."},
        },
    ]
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    _run_kairn("init", str(ws))
    return ws


@pytest.fixture
def transcript_root(tmp_path: Path) -> Path:
    r = tmp_path / "projects"
    _write_transcript(r, "-home-u-proj", "sess-1.jsonl")
    return r


def test_import_claude_code_writes_experiences(workspace: Path, transcript_root: Path):
    _, stdout, _ = _run_kairn(
        "import", "claude-code", str(workspace), "--root", str(transcript_root), "--yes"
    )
    data = json.loads(stdout)
    assert data["result"]["imported"] == 1
    assert data["result"]["namespace"] == "imported-claude-code"

    _, stdout2, _ = _run_kairn("status", str(workspace))
    assert json.loads(stdout2)["experiences"] >= 1


def test_import_claude_code_dry_run_writes_nothing(workspace: Path, transcript_root: Path):
    _, stdout, _ = _run_kairn(
        "import", "claude-code", str(workspace), "--root", str(transcript_root), "--dry-run"
    )
    assert json.loads(stdout)["result"]["dry_run"] is True

    _, stdout2, _ = _run_kairn("status", str(workspace))
    assert json.loads(stdout2)["experiences"] == 0


def test_import_claude_code_multiple_roots(workspace: Path, tmp_path: Path):
    r1 = tmp_path / "primary"
    r2 = tmp_path / "secondary"
    _write_transcript(r1, "-proj", "a.jsonl")
    _write_transcript(r2, "-proj", "b.jsonl")
    _, stdout, _ = _run_kairn(
        "import",
        "claude-code",
        str(workspace),
        "--root",
        str(r1),
        "--root",
        str(r2),
        "--yes",
    )
    assert json.loads(stdout)["result"]["imported"] == 2


def test_import_claude_code_confirmation_gate_declines(workspace: Path, transcript_root: Path):
    """No --yes and a 'n' answer must write nothing (no silent first-run writes)."""
    rc, _stdout, stderr = _run_kairn(
        "import",
        "claude-code",
        str(workspace),
        "--root",
        str(transcript_root),
        check=False,
        stdin="n\n",
    )
    assert rc == 0
    assert "Aborted" in stderr

    _, stdout2, _ = _run_kairn("status", str(workspace))
    assert json.loads(stdout2)["experiences"] == 0
