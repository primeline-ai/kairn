"""Deterministic git-log importer - WOW-9 Phase 1.

Zero-LLM: conventional-commit prefixes map to Kairn experience types via
plain string matching. No model calls, no network calls - `git log` runs
against a user-named local repo path only, never auto-discovered.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kairn.config import Config
from kairn.storage.base import StorageBackend

IMPORT_NAMESPACE = "imported-git"

_RECORD_SEP = "\x1e"
_FIELD_SEP = "\x1f"

# Conventional Commits (v1.0.0) type -> Kairn experience type. Types not
# listed here (docs, chore, style, test, build, ci, revert, or no
# recognized prefix) fall through to "decision" - lower-priority context,
# still worth importing.
_TYPE_MAP = {
    "fix": "solution",
    "feat": "pattern",
    "refactor": "pattern",
    "perf": "pattern",
}

_CONVENTIONAL_RE = re.compile(r"^([a-z]+)(\([^)]*\))?!?:\s")


class GitLogError(ValueError):
    """Raised when `git log` fails - empty repo (unborn HEAD), not a git
    repository, etc. A subclass of ValueError so it flows through the
    CLI's existing clean-error path (_run_json) without special-casing."""


def classify_commit_type(subject: str) -> str:
    """Map a commit subject line to a Kairn experience type.

    Pure prefix mapping - merge-ness is NOT decided here. A commit whose
    subject happens to start with the word "Merge" (e.g. "Merge duplicate
    customer records") is not a git merge commit and must not be treated
    as one; see `_iter_commits`, which uses git's own parent count.
    """
    match = _CONVENTIONAL_RE.match(subject)
    if match:
        return _TYPE_MAP.get(match.group(1), "decision")
    return "decision"


def _to_utc_iso(date_str: str) -> str:
    """Normalize an ISO-8601 date (any offset) to a UTC ISO-8601 string.

    Kairn's created_at/valid_from convention is always UTC (see
    Experience.created_at's default_factory) - storage-layer raw string
    comparisons (query_experiences_since) implicitly depend on this, so a
    commit authored in a non-UTC timezone must be converted, not stored
    with its original offset.
    """
    dt = datetime.fromisoformat(date_str)
    return dt.astimezone(UTC).isoformat()


def _iter_commits(repo_path: Path, *, since: str | None = None) -> list[dict[str, Any]]:
    """Run `git log` and parse commits via a control-character-delimited
    format, robust against arbitrary punctuation/newlines in messages."""
    fmt = f"%H{_FIELD_SEP}%P{_FIELD_SEP}%aI{_FIELD_SEP}%s{_FIELD_SEP}%b{_RECORD_SEP}"
    args = ["git", "log", f"--format={fmt}"]
    if since:
        args.append(f"--since={since}")
    result = subprocess.run(
        args,
        cwd=repo_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise GitLogError(
            f"git log failed in {repo_path}: {result.stderr.strip() or 'unknown error'}"
        )
    commits = []
    for record in result.stdout.split(_RECORD_SEP):
        record = record.strip("\n")
        if not record:
            continue
        sha, parents, author_date, subject, body = record.split(_FIELD_SEP, 4)
        commits.append(
            {
                "sha": sha,
                "is_merge": len(parents.split()) > 1,
                "date": _to_utc_iso(author_date),
                "subject": subject,
                "body": body.strip(),
            }
        )
    return commits


async def import_git_repo(
    store: StorageBackend,
    repo_path: Path,
    *,
    config: Config | None = None,
    since: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Import a local git repo's commit history as Kairn experiences.

    Idempotent: the experience id is derived deterministically from the
    commit SHA, so re-running is a no-op for already-imported commits -
    the duplicate-primary-key insert is caught and counted, not errored.
    A collision (a different commit landing on the same id) is verified
    against the existing row's source_ref and reported separately, never
    silently miscounted as a duplicate or allowed to clobber the row.
    """
    config = config or Config()
    repo_path = Path(repo_path).expanduser().resolve()
    commits = _iter_commits(repo_path, since=since)

    imported = 0
    skipped_merge = 0
    skipped_duplicate = 0
    collisions = 0
    preview: list[dict[str, str]] = []

    for commit in commits:
        if commit["is_merge"]:
            skipped_merge += 1
            continue

        exp_type = classify_commit_type(commit["subject"])
        content = commit["subject"]
        if commit["body"]:
            content = f"{commit['subject']}\n\n{commit['body']}"

        source_ref = f"git:{repo_path}:{commit['sha']}"
        exp_id = hashlib.sha256(source_ref.encode()).hexdigest()[:16]
        created_at = commit["date"]

        if dry_run:
            imported += 1
            preview.append({"id": exp_id, "type": exp_type, "content": content})
            continue

        payload = {
            "id": exp_id,
            "namespace": IMPORT_NAMESPACE,
            "type": exp_type,
            "content": content,
            "context": f"Imported from git commit {commit['sha'][:8]} in {repo_path.name}",
            "confidence": "high",
            "score": 1.0,
            "decay_rate": config.decay_rate_for_type(exp_type),
            "tags": None,
            "properties": {"source_ref": source_ref},
            "created_by": None,
            "access_count": 0,
            "promoted_to_node_id": None,
            "created_at": created_at,
            "last_accessed": None,
            "valid_from": created_at,
            "valid_to": None,
        }
        try:
            await store.insert_experience(payload)
            imported += 1
        except sqlite3.IntegrityError:
            existing = await store.get_experience(exp_id)
            existing_ref = (existing or {}).get("properties", {}).get("source_ref")
            if existing_ref == source_ref:
                skipped_duplicate += 1
            else:
                collisions += 1

    result: dict[str, Any] = {
        "repo": str(repo_path),
        "namespace": IMPORT_NAMESPACE,
        "imported": imported,
        "skipped_merge": skipped_merge,
        "skipped_duplicate": skipped_duplicate,
        "collisions": collisions,
        "dry_run": dry_run,
    }
    if dry_run:
        result["preview"] = preview
    return result
