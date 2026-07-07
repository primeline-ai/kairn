"""Tests for the `kairn import git` deterministic commit-metadata importer.

WOW-9 Phase 1 (2026-07-08, plan: .claude/plans/2026-07-07-kairn-wow9-import.md).
Zero-LLM: conventional-commit prefixes map to Kairn experience types via
plain string matching, no model calls anywhere in this module.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kairn.config import Config
from kairn.importers.git import classify_commit_type, import_git_repo
from kairn.storage.sqlite_store import SQLiteStore


def _git(repo: Path, *args: str, date: str | None = None) -> None:
    env = None
    if date is not None:
        import os

        env = {**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=env)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.name", "Test User")
    _git(r, "config", "user.email", "test@example.com")
    (r / "a.txt").write_text("1")
    _git(r, "add", "a.txt")
    _git(r, "commit", "-q", "-m", "fix: close the socket leak", date="2026-01-01T10:00:00")
    (r / "a.txt").write_text("2")
    _git(r, "add", "a.txt")
    _git(
        r,
        "commit",
        "-q",
        "-m",
        "feat: add retry with backoff\n\nHandles transient network errors.",
        date="2026-01-02T10:00:00",
    )
    (r / "a.txt").write_text("3")
    _git(r, "add", "a.txt")
    _git(r, "commit", "-q", "-m", "chore: bump version", date="2026-01-03T10:00:00")
    return r


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(workspace_path=tmp_path)


async def _imported(store: SQLiteStore) -> list[dict]:
    return await store.query_experiences_since("2020-01-01T00:00:00", namespace="imported-git")


# --- classify_commit_type ---


def test_classify_fix_as_solution():
    assert classify_commit_type("fix: close the socket leak") == "solution"


def test_classify_feat_as_pattern():
    assert classify_commit_type("feat: add retry with backoff") == "pattern"


def test_classify_refactor_as_pattern():
    assert classify_commit_type("refactor: extract helper") == "pattern"


def test_classify_chore_as_decision():
    assert classify_commit_type("chore: bump version") == "decision"


def test_classify_no_prefix_as_decision():
    assert classify_commit_type("bump version") == "decision"


# --- import_git_repo ---


async def test_import_writes_experiences_with_correct_types(
    repo: Path, store: SQLiteStore, config: Config
):
    result = await import_git_repo(store, repo, config=config)

    assert result["imported"] == 3  # 3 non-merge commits
    experiences = await _imported(store)
    types_by_content = {e["content"].splitlines()[0]: e["type"] for e in experiences}
    assert types_by_content["fix: close the socket leak"] == "solution"
    assert types_by_content["feat: add retry with backoff"] == "pattern"
    assert types_by_content["chore: bump version"] == "decision"


async def test_import_includes_commit_body_in_content(
    repo: Path, store: SQLiteStore, config: Config
):
    await import_git_repo(store, repo, config=config)
    experiences = await _imported(store)
    feat = next(e for e in experiences if e["content"].startswith("feat:"))
    assert "Handles transient network errors." in feat["content"]


async def test_import_sets_namespace_and_source_ref(repo: Path, store: SQLiteStore, config: Config):
    await import_git_repo(store, repo, config=config)
    experiences = await _imported(store)
    assert len(experiences) == 3
    for e in experiences:
        assert e["namespace"] == "imported-git"
        assert e["properties"]["source_ref"].startswith("git:")


async def test_import_uses_commit_date_as_created_at(
    repo: Path, store: SQLiteStore, config: Config
):
    await import_git_repo(store, repo, config=config)
    experiences = await _imported(store)
    fix_exp = next(e for e in experiences if e["content"].startswith("fix:"))
    assert fix_exp["created_at"].startswith("2026-01-01")
    assert fix_exp["valid_from"].startswith("2026-01-01")


async def test_import_is_idempotent(repo: Path, store: SQLiteStore, config: Config):
    first = await import_git_repo(store, repo, config=config)
    second = await import_git_repo(store, repo, config=config)

    assert first["imported"] == 3
    assert second["imported"] == 0
    assert second["skipped_duplicate"] == 3
    experiences = await _imported(store)
    assert len(experiences) == 3


async def test_import_dry_run_writes_nothing(repo: Path, store: SQLiteStore, config: Config):
    result = await import_git_repo(store, repo, config=config, dry_run=True)

    assert result["dry_run"] is True
    assert result["imported"] == 3
    experiences = await _imported(store)
    assert len(experiences) == 0


async def test_import_since_filters_older_commits(repo: Path, store: SQLiteStore, config: Config):
    result = await import_git_repo(store, repo, config=config, since="2026-01-02")

    assert result["imported"] == 2  # feat + chore, fix is before --since
    experiences = await _imported(store)
    contents = {e["content"].splitlines()[0] for e in experiences}
    assert "fix: close the socket leak" not in contents


async def test_import_skips_merge_commits(store: SQLiteStore, config: Config, tmp_path: Path):
    r = tmp_path / "repo_with_merge"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.name", "Test User")
    _git(r, "config", "user.email", "test@example.com")
    (r / "a.txt").write_text("1")
    _git(r, "add", "a.txt")
    _git(r, "commit", "-q", "-m", "fix: base commit", date="2026-01-01T10:00:00")
    _git(r, "checkout", "-q", "-b", "feature")
    (r / "b.txt").write_text("1")
    _git(r, "add", "b.txt")
    _git(r, "commit", "-q", "-m", "feat: feature work", date="2026-01-02T10:00:00")
    _git(r, "checkout", "-q", "main")
    _git(
        r,
        "merge",
        "-q",
        "--no-ff",
        "-m",
        "Merge branch 'feature'",
        "feature",
        date="2026-01-03T10:00:00",
    )

    result = await import_git_repo(store, r, config=config)

    assert result["imported"] == 2
    assert result["skipped_merge"] == 1


async def test_import_does_not_false_positive_on_merge_worded_subject(
    store: SQLiteStore, config: Config, tmp_path: Path
):
    """RC-gate finding: subject-text regex would wrongly drop a genuine,
    single-parent commit whose message happens to start with 'Merge'.
    Merge-ness must be structural (parent count), not text-based."""
    r = tmp_path / "repo_merge_worded"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.name", "Test User")
    _git(r, "config", "user.email", "test@example.com")
    (r / "a.txt").write_text("1")
    _git(r, "add", "a.txt")
    _git(
        r,
        "commit",
        "-q",
        "-m",
        "Merge duplicate customer records",
        date="2026-01-01T10:00:00",
    )

    result = await import_git_repo(store, r, config=config)

    assert result["imported"] == 1
    assert result["skipped_merge"] == 0


async def test_import_normalizes_commit_date_to_utc(
    store: SQLiteStore, config: Config, tmp_path: Path
):
    """RC-gate finding: %aI preserves the commit's original offset, but
    created_at/valid_from must be UTC - query_experiences_since does raw
    string comparison and every other writer stores UTC."""
    r = tmp_path / "repo_tz"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.name", "Test User")
    _git(r, "config", "user.email", "test@example.com")
    (r / "a.txt").write_text("1")
    _git(r, "add", "a.txt")
    _git(r, "commit", "-q", "-m", "fix: tz commit", date="2026-01-01T10:00:00+07:00")

    await import_git_repo(store, r, config=config)
    experiences = await _imported(store)
    exp = experiences[0]

    assert exp["created_at"].endswith("+00:00")
    assert exp["created_at"].startswith("2026-01-01T03:00:00")  # 10:00+07:00 -> 03:00 UTC
    assert exp["valid_from"].startswith("2026-01-01T03:00:00")


async def test_import_id_is_16_hex_chars(repo: Path, store: SQLiteStore, config: Config):
    """RC-gate finding: 8 hex chars (32 bits) has a real birthday-collision
    risk at the scale this feature is designed for (repeated re-imports
    over a repo's lifetime)."""
    await import_git_repo(store, repo, config=config)
    experiences = await _imported(store)
    assert all(len(e["id"]) == 16 for e in experiences)


async def test_import_detects_id_collision_without_clobbering_existing_row(
    repo: Path, store: SQLiteStore, config: Config
):
    """RC-gate finding: a bare IntegrityError was assumed to always mean
    'already imported' - if a different source_ref ever collided on id,
    the second commit was silently dropped and misreported as a dup.
    Now the existing row's source_ref is checked before counting it as
    a duplicate; a genuine mismatch is reported as a collision instead."""
    commits = await import_git_repo(store, repo, config=config)
    assert commits["imported"] == 3
    experiences = await _imported(store)
    victim = experiences[0]

    # Poison: delete the real row and reinsert the SAME id with a
    # different source_ref, simulating a hash collision with some other
    # (hypothetical) commit.
    await store.delete_experience(victim["id"])
    poisoned = dict(victim)
    poisoned["properties"] = {"source_ref": "git:/some/other/repo:deadbeef"}
    await store.insert_experience(poisoned)

    result = await import_git_repo(store, repo, config=config)

    assert result["collisions"] == 1
    assert result["skipped_duplicate"] == 2
    # The poisoned row must survive untouched - never silently overwritten.
    still_there = await store.get_experience(victim["id"])
    assert still_there["properties"]["source_ref"] == "git:/some/other/repo:deadbeef"


async def test_import_empty_repo_raises_clean_error(
    store: SQLiteStore, config: Config, tmp_path: Path
):
    """RC-gate finding: an unborn-HEAD repo (git init, zero commits) made
    `git log` exit 128, which propagated as a raw CalledProcessError
    instead of a clean, catchable error."""
    from kairn.importers.git import GitLogError

    r = tmp_path / "empty_repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")

    with pytest.raises(GitLogError):
        await import_git_repo(store, r, config=config)


async def test_import_non_git_directory_raises_clean_error(
    store: SQLiteStore, config: Config, tmp_path: Path
):
    from kairn.importers.git import GitLogError

    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()

    with pytest.raises(GitLogError):
        await import_git_repo(store, not_a_repo, config=config)


async def test_import_handles_non_ascii_commit_content(
    store: SQLiteStore, config: Config, tmp_path: Path
):
    """RC-gate finding: bare text=True without an explicit encoding risks
    UnicodeDecodeError under a restrictive locale (LANG=C, common in CI /
    minimal containers)."""
    r = tmp_path / "repo_unicode"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.name", "Test User")
    _git(r, "config", "user.email", "test@example.com")
    (r / "a.txt").write_text("1")
    _git(r, "add", "a.txt")
    _git(
        r,
        "commit",
        "-q",
        "-m",
        "fix: résumé parsing für Käse 🧀",
        date="2026-01-01T10:00:00",
    )

    result = await import_git_repo(store, r, config=config)

    assert result["imported"] == 1
    experiences = await _imported(store)
    assert "résumé parsing für Käse 🧀" in experiences[0]["content"]
