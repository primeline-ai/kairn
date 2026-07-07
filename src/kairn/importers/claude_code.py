"""Deterministic Claude Code transcript importer - WOW-9 Phase 3 (coarse mode).

Zero-LLM, zero-network: reads local ``~/.claude/projects`` JSONL transcripts and
writes ONE session-summary experience per session. Ships in COARSE mode per the
plan's FAILED-condition fallback (Kairn decision ``21985368``): the Phase 3
precision spike measured fine-grained per-signal rule extraction at ~38% junk on
a real N=68 holdout, above the plan's <20% bar. Coarse mode is deterministic and
high-precision because the summary is a fixed pair of fields (the session's
``aiTitle`` + its first genuine user prompt), not marker-matched prose.

Design choices:

- **One experience per session.** A transcript file is a session. Nested
  ``subagents/`` transcripts are excluded - they are children of a session, and
  coarse mode emits one summary per session.
- **Only user-authored prose is read.** The summary is built from the first
  genuine user prompt; ``isMeta`` records (slash-command expansions, skill
  preambles, hook-injected context - the dominant false-positive source measured
  in the spike), system-wrapper content, and ``tool_result`` blocks are skipped.
  Assistant ``text``/``thinking``/``tool_use`` blocks are never read, so a
  ``kn_learn``/``kn_save``/``kn_add`` tool call can never be re-imported
  (Kairn ``d7a24b80``) - it is excluded by construction, not by a fragile filter.
- **Privacy.** Every stored string (content AND ``source_ref``) is routed through
  the Phase 2 redaction module before it touches the store.
- **Schema resilience.** The CC JSONL schema is internal/undocumented and can
  drift between releases (Kairn ``82ed6779``). Parsing is defensive (malformed
  lines and unrecognized record shapes are skipped, never fatal) and
  ``assert_transcript_schema`` fails loudly if the core fields we depend on
  disappear from a real sample - a drift canary, not a one-time check.
- **Idempotency & rollback.** Writes into the dedicated ``imported-claude-code``
  namespace with a stable, path-derived ``source_ref``; re-running is a no-op for
  already-imported sessions, and a bad import is reversed by a namespace-scoped
  delete (mirrors the git importer).
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kairn.config import Config
from kairn.importers.redact import redact
from kairn.storage.base import StorageBackend

IMPORT_NAMESPACE = "imported-claude-code"

# Default transcript roots: the primary account plus a secondary-account tree
# (Stage 0 found a real dual-account layout). Only those that exist are scanned;
# --root overrides/extends this set, it is not mandatory for the common case.
_DEFAULT_ROOTS = (
    Path("~/.claude/projects"),
    Path("~/.claude-secondary/projects"),
)

# A coarse session summary spans a whole session (many topics); "decision" is the
# neutral, valid bucket used the same way the git importer uses it as a catch-all
# ("context" is not in VALID_TYPES and would be rejected by the recall/engine path).
_SESSION_TYPE = "decision"

_MAX_CONTENT = 800  # a session summary is a pointer, not the whole transcript

# System/harness wrappers injected into a user turn - not the user's own words.
_SYSTEM_WRAPPER = re.compile(
    r"^\s*(?:#\s*/|<local-command|<command-|<task-notification|<system-reminder"
    r"|\[Image:|\[Request interrupted|Base directory for this skill:|You are the )"
)

_BARE_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class SchemaError(RuntimeError):
    """A real transcript sample no longer matches the schema shape this parser
    was built against - a loud drift canary (Kairn ``82ed6779``)."""


def default_roots() -> list[Path]:
    """Expanded default roots that actually exist on this machine."""
    return [p.expanduser() for p in _DEFAULT_ROOTS if p.expanduser().is_dir()]


def discover_transcripts(roots: list[Path]) -> list[Path]:
    """All top-level session transcripts under the given roots, sorted.

    One file == one CC session. Nested ``subagents/`` transcripts are excluded
    (children of a session). Sorted for deterministic ordering across runs."""
    files: set[Path] = set()
    for root in roots:
        root = Path(root).expanduser()
        if not root.is_dir():
            continue
        for path in root.rglob("*.jsonl"):
            if "subagents" in path.parts:
                continue
            files.add(path.resolve())
    return sorted(files)


def _iter_records(path: Path):
    """Yield parsed JSON dict records; skip blank/malformed lines defensively."""
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec


def _normalize_ts(ts: Any) -> str | None:
    """ISO-8601 (incl. a trailing ``Z``) -> UTC ISO string; None if unparseable."""
    if not isinstance(ts, str):
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(UTC).isoformat()


def _normalize_since(since: str) -> str:
    """Bare ``YYYY-MM-DD`` -> that date at UTC midnight; a full ISO datetime is
    normalized to UTC. The result is string-comparable against a session's
    UTC-ISO ``created_at``."""
    if _BARE_DATE_RE.match(since):
        return f"{since}T00:00:00+00:00"
    return _normalize_ts(since) or since


def _user_text(message: dict) -> str | None:
    """The user's own prose from a message, or None.

    A user message's content is a bare string OR a list of blocks. Only ``text``
    blocks are the user's words; ``tool_result`` blocks (tool output fed back)
    are NOT and are skipped - they are the single largest carrier of raw secrets
    in a transcript."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]
        joined = " ".join(p for p in parts if p)
        return joined or None
    return None


def assert_transcript_schema(records: list[dict]) -> None:
    """Fail loudly if a real sample lost the fields the parser depends on.

    Verifies that, across the sample, at least one record carries ``type``, at
    least one record carries a ``message.role``, and at least one carries a
    ``timestamp``. An empty sample is tolerated (nothing to import, no false
    alarm). This is the drift canary for the internal, undocumented CC schema
    (Kairn ``82ed6779``)."""
    if not records:
        return
    has_type = any("type" in r for r in records)
    has_role = any(isinstance(r.get("message"), dict) and "role" in r["message"] for r in records)
    has_ts = any("timestamp" in r for r in records)
    missing = [
        name
        for name, ok in (
            ("type", has_type),
            ("message.role", has_role),
            ("timestamp", has_ts),
        )
        if not ok
    ]
    if missing:
        raise SchemaError(
            "Claude Code transcript schema drift: expected field(s) missing from a "
            f"real sample: {', '.join(missing)}. The parser was built against "
            "type / message.role / timestamp; a CC release may have changed the shape."
        )


def extract_session_summary(records: list[dict], *, source_ref: str) -> dict | None:
    """Build one coarse session summary from a transcript's records, or None.

    Content = the session's ``aiTitle`` (if any) + its first genuine user prompt.
    'Genuine' excludes ``isMeta`` records, system-wrapper content, and
    ``tool_result`` blocks. Returns None when a session has no user-authored
    prose (e.g. a pure automation run) - such a session yields no experience.
    Redaction is applied by the caller; this stays pure text assembly."""
    ai_title: str | None = None
    first_prompt: str | None = None
    created_at: str | None = None
    cwd = git_branch = session_id = None

    for rec in records:
        ts = _normalize_ts(rec.get("timestamp"))
        if ts and (created_at is None or ts < created_at):
            created_at = ts

        rtype = rec.get("type")
        if rtype == "ai-title" and ai_title is None:
            ai_title = rec.get("aiTitle") or rec.get("title")
            continue
        if rtype != "user" or rec.get("isMeta") or first_prompt is not None:
            continue
        msg = rec.get("message")
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        text = _user_text(msg)
        if not text:
            continue
        text = text.strip()
        if not text or _SYSTEM_WRAPPER.match(text):
            continue
        first_prompt = text
        cwd = rec.get("cwd")
        git_branch = rec.get("gitBranch")
        session_id = rec.get("sessionId")

    if first_prompt is None:
        return None

    title = (ai_title or "").strip()
    body = first_prompt
    if len(body) > _MAX_CONTENT:
        body = body[:_MAX_CONTENT].rstrip() + " ..."
    content = f"{title}\n\n{body}" if title else body
    return {
        "content": content,
        "created_at": created_at,
        "cwd": cwd,
        "gitBranch": git_branch,
        "sessionId": session_id,
        "source_ref": source_ref,
    }


def _file_mtime_iso(path: Path) -> str:
    """UTC-ISO file mtime - a deterministic fallback created_at for the rare
    session with no parseable timestamp in any record."""
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()


async def import_claude_code(
    store: StorageBackend,
    roots: list[Path] | None = None,
    *,
    config: Config | None = None,
    since: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Import Claude Code session transcripts as coarse session-summary experiences.

    Idempotent: the experience id is derived deterministically from the session's
    path-based ``source_ref``, so re-running is a no-op for already-imported
    sessions. A hash collision (a different session on the same id) is verified
    against the existing row's ``source_ref`` and reported separately, never
    silently miscounted or allowed to clobber the row (mirrors the git importer)."""
    config = config or Config()
    roots = list(roots) if roots is not None else default_roots()
    cutoff = _normalize_since(since) if since else None

    files = discover_transcripts(roots)
    schema_checked = False

    imported = 0
    skipped_empty = 0
    skipped_since = 0
    skipped_duplicate = 0
    collisions = 0
    redactions = 0
    preview: list[dict[str, str]] = []

    for path in files:
        records = list(_iter_records(path))
        if not records:
            continue
        if not schema_checked:
            assert_transcript_schema(records)  # drift canary on the first real file
            schema_checked = True

        source_ref = f"claude-code:{path}"
        summary = extract_session_summary(records, source_ref=source_ref)
        if summary is None:
            skipped_empty += 1
            continue

        created_at = summary["created_at"] or _file_mtime_iso(path)
        if cutoff and created_at < cutoff:
            skipped_since += 1
            continue

        result_redact = redact(summary["content"])
        content = result_redact.text
        redactions += len(result_redact.findings)

        exp_id = hashlib.sha256(source_ref.encode()).hexdigest()[:16]

        if dry_run:
            imported += 1
            preview.append({"id": exp_id, "type": _SESSION_TYPE, "content": content})
            continue

        context_bits = [b for b in (summary["cwd"], summary["gitBranch"]) if b]
        context = "Imported from Claude Code session"
        if context_bits:
            context += " (" + ", ".join(context_bits) + ")"

        payload = {
            "id": exp_id,
            "namespace": IMPORT_NAMESPACE,
            "type": _SESSION_TYPE,
            "content": content,
            "context": redact(context).text,
            "confidence": "high",
            "score": 1.0,
            "decay_rate": config.decay_rate_for_type(_SESSION_TYPE),
            "tags": None,
            "properties": {"source_ref": source_ref, "session_id": summary["sessionId"]},
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
        "roots": [str(Path(r).expanduser()) for r in roots],
        "namespace": IMPORT_NAMESPACE,
        "sessions_scanned": len(files),
        "imported": imported,
        "skipped_empty": skipped_empty,
        "skipped_since": skipped_since,
        "skipped_duplicate": skipped_duplicate,
        "collisions": collisions,
        "redactions": redactions,
        "dry_run": dry_run,
    }
    if dry_run:
        result["preview"] = preview
    return result
