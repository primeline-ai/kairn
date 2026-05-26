"""Read-only Kairn health checks.

Each check is an async function `(store: StorageBackend) -> dict` that
returns the standard envelope. The functions reach into `store.db`
directly because the storage layer's `query_nodes` / `get_stats`
abstractions don't surface the low-level signals diagnostics need
(PRAGMA results, virtual-table counts, COUNT-with-JOIN). Read-only
SQL only.
"""

from __future__ import annotations

from typing import Any

from kairn.storage.base import StorageBackend
from kairn.storage.sqlite_store import SQLiteStore


def _envelope(
    check_id: str,
    *,
    severity: str,
    status: str,
    evidence: str,
    safe_next_step: str,
) -> dict[str, Any]:
    """Build the standard diagnostic envelope.

    severity: info | warn | error (intrinsic importance of the check)
    status:   ok | warn | fail    (the verdict of this specific run)
    """
    return {
        "check_id": check_id,
        "severity": severity,
        "status": status,
        "evidence": evidence,
        "safe_next_step": safe_next_step,
    }


async def _db(store: StorageBackend):
    """Extract the live aiosqlite connection. Only SQLiteStore is supported."""
    if not isinstance(store, SQLiteStore):
        raise RuntimeError(
            "Diagnostic checks require SQLiteStore; got "
            f"{type(store).__name__}"
        )
    return store.db


# ----------------------------------------------------------------------
# Check 1: SQLite journal mode + busy_timeout sanity
# ----------------------------------------------------------------------


async def check_lock_mode(store: StorageBackend) -> dict[str, Any]:
    """SQLite must be in WAL mode for concurrent reads. A corrupted DB
    or external `journal_mode=DELETE` write would silently degrade
    concurrency. This check catches the regression.

    NOTE: busy_timeout is observed but NOT a gate condition. The
    current SQLiteStore.initialize() does not set it, so requiring
    it would produce a false fail on every healthy workspace. If
    initialize() starts setting busy_timeout, tighten this check
    to require both. Severity uses `warn` (consistent with the
    other 4 checks) so the summary `error` counter is reserved
    for the registry-level "the check itself crashed" sentinel.
    """
    db = await _db(store)
    journal_row = await (await db.execute("PRAGMA journal_mode")).fetchone()
    timeout_row = await (await db.execute("PRAGMA busy_timeout")).fetchone()
    journal = (journal_row[0] if journal_row else "unknown").lower()
    busy_timeout = int(timeout_row[0]) if timeout_row else 0

    if journal == "wal":
        return _envelope(
            "check_lock_mode",
            severity="warn",
            status="ok",
            evidence=f"journal_mode={journal} busy_timeout={busy_timeout}ms",
            safe_next_step="No action.",
        )

    return _envelope(
        "check_lock_mode",
        severity="warn",
        status="fail",
        evidence=f"journal_mode={journal} (expected wal); busy_timeout={busy_timeout}ms",
        safe_next_step=(
            "Re-initialize the workspace (kairn init) or run "
            "PRAGMA journal_mode=WAL;"
        ),
    )


# ----------------------------------------------------------------------
# Check 2: FTS5 index parity (nodes vs nodes_fts)
# ----------------------------------------------------------------------


async def check_fts_index_health(store: StorageBackend) -> dict[str, Any]:
    """nodes_fts is an external-content FTS5 table synced via triggers
    on the nodes table. If the row counts diverge (e.g. a manual
    INSERT bypassed the trigger, or the trigger was dropped), candidate
    scans in kn_learn (Phase 2) will return stale or missing matches.
    """
    db = await _db(store)
    nodes_n = (await (await db.execute(
        "SELECT COUNT(*) FROM nodes WHERE deleted_at IS NULL"
    )).fetchone())[0]
    fts_n = (await (await db.execute(
        "SELECT COUNT(*) FROM nodes_fts"
    )).fetchone())[0]
    # FTS5 also indexes soft-deleted nodes. The `nodes_fts_au` trigger
    # (schema/triggers.sql line 21-26) fires on every UPDATE - including
    # the soft-delete UPDATE that sets deleted_at - and does
    # `INSERT 'delete'` + `INSERT new`, so the row stays in nodes_fts
    # with deleted_at populated. The matching `nodes_fts_ad` trigger
    # only fires on hard DELETE (which Kairn does not perform via the
    # storage layer). Therefore: nodes_fts row count == count of nodes
    # not hard-deleted == live nodes + soft-deleted nodes.
    deleted_n = (await (await db.execute(
        "SELECT COUNT(*) FROM nodes WHERE deleted_at IS NOT NULL"
    )).fetchone())[0]

    expected_fts_n = nodes_n + deleted_n
    if fts_n == expected_fts_n:
        return _envelope(
            "check_fts_index_health",
            severity="warn",
            status="ok",
            evidence=f"nodes={nodes_n} deleted={deleted_n} fts={fts_n}",
            safe_next_step="No action.",
        )

    drift = abs(fts_n - expected_fts_n)
    return _envelope(
        "check_fts_index_health",
        severity="warn",
        status="warn",
        evidence=(
            f"nodes_fts row count drift: nodes={nodes_n} deleted={deleted_n} "
            f"expected_fts={expected_fts_n} actual_fts={fts_n} (delta={drift})"
        ),
        safe_next_step=(
            "Rebuild the FTS5 index: INSERT INTO nodes_fts(nodes_fts) "
            "VALUES('rebuild'); (read-only diagnostic so we cannot do "
            "this for you)"
        ),
    )


# ----------------------------------------------------------------------
# Check 3: promoted-experience consistency
# ----------------------------------------------------------------------


async def check_promoted_experience_consistency(
    store: StorageBackend,
) -> dict[str, Any]:
    """Experiences flagged needs_promotion=1 should eventually have
    promoted_to_node_id set by promote_pending. A growing backlog of
    flagged-but-not-promoted experiences indicates the sweeper isn't
    running or is failing. Threshold: > 50 flagged unpromoted = warn.
    """
    db = await _db(store)
    flagged_unpromoted_n = (await (await db.execute(
        "SELECT COUNT(*) FROM experiences "
        "WHERE json_extract(properties, '$.needs_promotion') = 1 "
        "AND promoted_to_node_id IS NULL"
    )).fetchone())[0]
    promoted_n = (await (await db.execute(
        "SELECT COUNT(*) FROM experiences WHERE promoted_to_node_id IS NOT NULL"
    )).fetchone())[0]

    if flagged_unpromoted_n <= 50:
        return _envelope(
            "check_promoted_experience_consistency",
            severity="warn",
            status="ok",
            evidence=(
                f"flagged_unpromoted={flagged_unpromoted_n} "
                f"already_promoted={promoted_n}"
            ),
            safe_next_step="No action.",
        )
    return _envelope(
        "check_promoted_experience_consistency",
        severity="warn",
        status="warn",
        evidence=(
            f"flagged_unpromoted={flagged_unpromoted_n} > 50; "
            f"already_promoted={promoted_n}"
        ),
        safe_next_step=(
            "Run promote_pending sweeper: `kairn promote-pending <path>` "
            "or call IntelligenceLayer.promote_pending(limit=100)."
        ),
    )


# ----------------------------------------------------------------------
# Check 4: namespace distribution sanity
# ----------------------------------------------------------------------


async def check_namespace_distribution(store: StorageBackend) -> dict[str, Any]:
    """Catches accidental namespace pollution (e.g. a save with
    `namespace=""` or a typo creating an unintended namespace). The
    workspace should have a small finite set of namespaces; > 20
    distinct namespaces is suspicious.
    """
    db = await _db(store)
    rows = await (await db.execute(
        "SELECT namespace, COUNT(*) AS n FROM nodes "
        "WHERE deleted_at IS NULL GROUP BY namespace ORDER BY n DESC"
    )).fetchall()
    distinct_n = len(rows)
    summary = ", ".join(f"{row[0]}={row[1]}" for row in rows[:5])
    if distinct_n <= 20:
        return _envelope(
            "check_namespace_distribution",
            severity="info",
            status="ok",
            evidence=f"distinct_namespaces={distinct_n} ({summary or 'empty'})",
            safe_next_step="No action.",
        )
    return _envelope(
        "check_namespace_distribution",
        severity="warn",
        status="warn",
        evidence=(
            f"distinct_namespaces={distinct_n} > 20; top: {summary}. "
            f"Likely typos or accidental pollution."
        ),
        safe_next_step=(
            "Review SELECT DISTINCT namespace FROM nodes; consolidate "
            "or rename mistyped namespaces."
        ),
    )


# ----------------------------------------------------------------------
# Check 5: orphan edges (edges referencing missing nodes)
# ----------------------------------------------------------------------


async def check_orphan_edges(store: StorageBackend) -> dict[str, Any]:
    """Edges have a foreign key on nodes(id) but SQLite enforces FKs
    only when PRAGMA foreign_keys is ON. Soft-deleted nodes are still
    rows (deleted_at IS NOT NULL), so they don't break referential
    integrity. True orphans (source_id / target_id not in nodes at all)
    indicate either a hard-DELETE bypass or pre-FK-enforcement legacy.
    """
    db = await _db(store)
    orphan_n = (await (await db.execute(
        "SELECT COUNT(*) FROM edges e "
        "WHERE NOT EXISTS (SELECT 1 FROM nodes n WHERE n.id = e.source_id) "
        "   OR NOT EXISTS (SELECT 1 FROM nodes n WHERE n.id = e.target_id)"
    )).fetchone())[0]
    total_n = (await (await db.execute("SELECT COUNT(*) FROM edges")).fetchone())[0]

    if orphan_n == 0:
        return _envelope(
            "check_orphan_edges",
            severity="warn",
            status="ok",
            evidence=f"edges={total_n} orphans=0",
            safe_next_step="No action.",
        )
    return _envelope(
        "check_orphan_edges",
        severity="warn",
        status="warn",
        evidence=f"edges={total_n} orphans={orphan_n}",
        safe_next_step=(
            "DELETE FROM edges WHERE source_id NOT IN (SELECT id FROM nodes) "
            "OR target_id NOT IN (SELECT id FROM nodes); (read-only diagnostic, "
            "manual cleanup required)"
        ),
    )
