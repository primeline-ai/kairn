"""Bi-temporal validity + cross-session recall tests (LongMemEval level-up).

Phase 2: valid_from / valid_to (valid-time window), orthogonal to decay.
Phase 3: entity_key (cross-session grouping).
Phase 4: session/valid-time-diverse recall mode.

valid-time answers "what was true at T"; it is a SEPARATE axis from
transaction-time (created_at) and from decay (relevance()). The orthogonality
test enforces that valid_to never feeds relevance().
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from kairn.core.experience import ExperienceEngine
from kairn.events.bus import EventBus
from kairn.models.experience import Experience
from kairn.storage.sqlite_store import SQLiteStore


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Phase 2: schema migration + validity-window storage
# ---------------------------------------------------------------------------

async def test_migrate_legacy_experiences_adds_bitemporal_columns(tmp_path):
    """A legacy experiences table without valid_from/valid_to gets them on
    initialize(); existing rows default to NULL; row count preserved; idempotent.
    """
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE experiences (
            id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL DEFAULT 'knowledge',
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            context TEXT,
            confidence TEXT DEFAULT 'high',
            score REAL NOT NULL DEFAULT 1.0,
            decay_rate REAL NOT NULL,
            tags JSON,
            properties JSON,
            created_by TEXT,
            access_count INTEGER DEFAULT 0,
            promoted_to_node_id TEXT,
            created_at TEXT NOT NULL,
            last_accessed TEXT
        )
        """
    )
    for i in range(3):
        conn.execute(
            "INSERT INTO experiences (id, type, content, decay_rate, created_at) "
            "VALUES (?, 'solution', ?, 0.003, '2026-01-01T00:00:00Z')",
            (f"legacy-{i}", f"Legacy {i}"),
        )
    conn.commit()
    pre = {r[1] for r in conn.execute("PRAGMA table_info(experiences)")}
    assert "valid_from" not in pre and "valid_to" not in pre
    conn.close()

    store = SQLiteStore(db_path)
    await store.initialize()
    try:
        cur = await store.db.execute("PRAGMA table_info(experiences)")
        post = {r[1] for r in await cur.fetchall()}
        assert "valid_from" in post
        assert "valid_to" in post
        cur = await store.db.execute("SELECT COUNT(*) FROM experiences")
        assert (await cur.fetchone())[0] == 3
        cur = await store.db.execute(
            "SELECT valid_from, valid_to FROM experiences WHERE id='legacy-0'")
        row = await cur.fetchone()
        assert row[0] is None and row[1] is None
    finally:
        await store.close()

    # Idempotent re-init
    store2 = SQLiteStore(db_path)
    await store2.initialize()
    try:
        cur = await store2.db.execute("SELECT COUNT(*) FROM experiences")
        assert (await cur.fetchone())[0] == 3
    finally:
        await store2.close()


async def test_save_and_read_validity_window(store: SQLiteStore):
    """save() accepts valid_from/valid_to and the store round-trips them."""
    engine = ExperienceEngine(store, EventBus())
    exp = await engine.save(
        content="User prefers Adobe Premiere Pro",
        type="pattern",
        valid_from="2023/01/08 (Sun) 12:49",
        valid_to=None,
        namespace="bench",
    )
    assert exp.valid_from == "2023/01/08 (Sun) 12:49"
    assert exp.valid_to is None
    got = await store.get_experience(exp.id)
    assert got["valid_from"] == "2023/01/08 (Sun) 12:49"
    assert got["valid_to"] is None


async def test_save_defaults_validity_to_none(store: SQLiteStore):
    """Existing callers that do not pass validity get NULL windows (no decay
    impact, fully backward compatible)."""
    engine = ExperienceEngine(store, EventBus())
    exp = await engine.save(content="x", type="solution")
    assert exp.valid_from is None and exp.valid_to is None


# ---------------------------------------------------------------------------
# Phase 2: orthogonality - valid_to NEVER feeds relevance() (Kairn GP2 7ff6120c)
# ---------------------------------------------------------------------------

def test_relevance_ignores_valid_to():
    """Two experiences identical except for a closed validity window must have
    byte-identical relevance: valid-time is orthogonal to decay (the temporal-
    index vs decay orthogonality principle)."""
    base = dict(type="solution", content="c", decay_rate=0.003,
                created_at="2026-01-01T00:00:00+00:00", score=1.0)
    open_window = Experience(**base, valid_from="2026-01-01T00:00:00+00:00", valid_to=None)
    closed_window = Experience(
        **base, valid_from="2026-01-01T00:00:00+00:00",
        valid_to="2026-01-02T00:00:00+00:00")
    at = datetime(2026, 6, 1, tzinfo=UTC)
    assert open_window.relevance(at=at) == closed_window.relevance(at=at)


def test_relevance_unchanged_for_no_window_case():
    """relevance() for an experience with NO validity window equals the legacy
    formula score*exp(-decay_rate*days) exactly."""
    import math
    exp = Experience(type="solution", content="c", decay_rate=0.003,
                     created_at="2026-01-01T00:00:00+00:00", score=1.0)
    at = datetime(2026, 1, 31, tzinfo=UTC)
    days = (at - datetime(2026, 1, 1, tzinfo=UTC)).total_seconds() / 86400
    assert exp.relevance(at=at) == pytest.approx(1.0 * math.exp(-0.003 * days))


def test_closed_validity_still_decays_on_own_schedule():
    """A fact whose validity is closed (superseded) still decays by created_at,
    not by valid_to - decay and validity are independent code paths."""
    superseded = Experience(
        type="decision", content="old fact", decay_rate=0.01,
        created_at="2026-01-01T00:00:00+00:00",
        valid_from="2026-01-01T00:00:00+00:00",
        valid_to="2026-01-05T00:00:00+00:00")
    # Decay is positive and based on age from created_at, regardless of valid_to.
    r_early = superseded.relevance(at=datetime(2026, 1, 2, tzinfo=UTC))
    r_late = superseded.relevance(at=datetime(2026, 3, 1, tzinfo=UTC))
    assert r_early > r_late > 0
