"""CLI tests for `kairn import git` - WOW-9 Phase 1.

Uses subprocess to exercise the CLI as a shell user would, matching the
convention in test_cli.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run_kairn(*args: str, check: bool = True) -> tuple[int, str, str]:
    cmd = [sys.executable, "-m", "kairn.cli", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise AssertionError(
            f"CLI command failed: {' '.join(cmd)}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result.returncode, result.stdout, result.stderr


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    _run_kairn("init", str(ws))
    return ws


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.name", "Test User")
    _git(r, "config", "user.email", "test@example.com")
    (r / "a.txt").write_text("1")
    _git(r, "add", "a.txt")
    _git(r, "commit", "-q", "-m", "fix: a real bug")
    return r


def test_import_git_writes_experiences(workspace: Path, git_repo: Path):
    _, stdout, _ = _run_kairn("import", "git", str(workspace), str(git_repo))
    data = json.loads(stdout)
    assert data["results"][0]["imported"] == 1
    assert data["results"][0]["namespace"] == "imported-git"

    _, stdout2, _ = _run_kairn("status", str(workspace))
    stats = json.loads(stdout2)
    assert stats["experiences"] >= 1


def test_import_git_dry_run_flag(workspace: Path, git_repo: Path):
    _, stdout, _ = _run_kairn("import", "git", str(workspace), str(git_repo), "--dry-run")
    data = json.loads(stdout)
    assert data["results"][0]["dry_run"] is True

    _, stdout2, _ = _run_kairn("status", str(workspace))
    stats = json.loads(stdout2)
    assert stats["experiences"] == 0


def test_import_git_multiple_repos(workspace: Path, git_repo: Path, tmp_path: Path):
    r2 = tmp_path / "repo2"
    r2.mkdir()
    _git(r2, "init", "-q", "-b", "main")
    _git(r2, "config", "user.name", "Test User")
    _git(r2, "config", "user.email", "test@example.com")
    (r2 / "b.txt").write_text("1")
    _git(r2, "add", "b.txt")
    _git(r2, "commit", "-q", "-m", "feat: second repo commit")

    _, stdout, _ = _run_kairn("import", "git", str(workspace), str(git_repo), str(r2))
    data = json.loads(stdout)
    assert len(data["results"]) == 2


def test_import_git_missing_repo_errors(workspace: Path, tmp_path: Path):
    rc, _, _stderr = _run_kairn(
        "import", "git", str(workspace), str(tmp_path / "nope"), check=False
    )
    assert rc != 0


def test_import_git_empty_repo_errors_cleanly(workspace: Path, tmp_path: Path):
    """RC-gate finding: an unborn-HEAD repo must produce a clean JSON error
    + non-zero exit, not a raw Python traceback."""
    empty_repo = tmp_path / "empty_repo"
    empty_repo.mkdir()
    _git(empty_repo, "init", "-q", "-b", "main")

    rc, stdout, stderr = _run_kairn("import", "git", str(workspace), str(empty_repo), check=False)

    assert rc != 0
    assert "Traceback" not in stderr
    if stdout.strip():
        data = json.loads(stdout)
        assert "error" in data
