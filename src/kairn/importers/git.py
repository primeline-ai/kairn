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

_MERGE_RE = re.compile(r"^Merge\b", re.IGNORECASE)
_CONVENTIONAL_RE = re.compile(r"^([a-z]+)(\([^)]*\))?!?:\s")


def classify_commit_type(subject: str) -> str | None:
    """Map a commit subject line to a Kairn experience type.

    Returns None for merge commits - filtered out per the import plan,
    they carry branch-topology noise, not independent decision signal.
    """
    if _MERGE_RE.match(subject):
        return None
    match = _CONVENTIONAL_RE.match(subject)
    if match:
        return _TYPE_MAP.get(match.group(1), "decision")
    return "decision"


def _iter_commits(repo_path: Path, *, since: str | None = None) -> list[dict[str, str]]:
    """Run `git log` and parse commits via a control-character-delimited
    format, robust against arbitrary punctuation/newlines in messages."""
    fmt = f"%H{_FIELD_SEP}%aI{_FIELD_SEP}%s{_FIELD_SEP}%b{_RECORD_SEP}"
    args = ["git", "log", f"--format={fmt}"]
    if since:
        args.append(f"--since={since}")
    result = subprocess.run(args, cwd=repo_path, check=True, capture_output=True, text=True)
    commits = []
    for record in result.stdout.split(_RECORD_SEP):
        record = record.strip("\n")
        if not record:
            continue
        sha, author_date, subject, body = record.split(_FIELD_SEP, 3)
        commits.append({"sha": sha, "date": author_date, "subject": subject, "body": body.strip()})
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
    """
    config = config or Config()
    repo_path = Path(repo_path).expanduser().resolve()
    commits = _iter_commits(repo_path, since=since)

    imported = 0
    skipped_merge = 0
    skipped_duplicate = 0
    preview: list[dict[str, str]] = []

    for commit in commits:
        exp_type = classify_commit_type(commit["subject"])
        if exp_type is None:
            skipped_merge += 1
            continue

        content = commit["subject"]
        if commit["body"]:
            content = f"{commit['subject']}\n\n{commit['body']}"

        source_ref = f"git:{repo_path}:{commit['sha']}"
        exp_id = hashlib.sha256(source_ref.encode()).hexdigest()[:8]
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
            skipped_duplicate += 1

    result: dict[str, Any] = {
        "repo": str(repo_path),
        "namespace": IMPORT_NAMESPACE,
        "imported": imported,
        "skipped_merge": skipped_merge,
        "skipped_duplicate": skipped_duplicate,
        "dry_run": dry_run,
    }
    if dry_run:
        result["preview"] = preview
    return result
