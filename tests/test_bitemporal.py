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

from kairn.core.experience import ExperienceEngine, normalize_date_prefix
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


# ---------------------------------------------------------------------------
# Phase 3: rule-based cross-session entity grouping (no LLM, no embeddings)
# ---------------------------------------------------------------------------

def test_derive_entity_key_is_deterministic():
    """Same (content, tags) -> same key, every call. Seed-stable, no randomness."""
    from kairn.core.experience import derive_entity_key
    k1 = derive_entity_key("the user prefers Premiere Pro", ["video-editing", "tools"])
    k2 = derive_entity_key("a totally different sentence", ["Video-Editing"])
    assert k1 == derive_entity_key("the user prefers Premiere Pro", ["video-editing", "tools"])
    # normalization: case-insensitive, same tag -> same key regardless of content
    assert k1 == k2


def test_derive_entity_key_none_without_tags():
    """No tags -> no entity key (recall falls back to session/valid-time)."""
    from kairn.core.experience import derive_entity_key
    assert derive_entity_key("some content", None) is None
    assert derive_entity_key("some content", []) is None


async def test_save_sets_entity_key_from_tags(store: SQLiteStore):
    engine = ExperienceEngine(store, EventBus())
    exp = await engine.save(content="x", type="pattern", tags=["Model-Kits", "hobby"])
    assert exp.entity_key == "model-kits"
    got = await store.get_experience(exp.id)
    assert got["entity_key"] == "model-kits"


async def test_group_by_entity_aggregates_across_sessions(store: SQLiteStore):
    """Two saves in different namespaces (sessions) about the same subject group
    together; a different subject does not merge in."""
    engine = ExperienceEngine(store, EventBus())
    await engine.save(content="bought an F-15 kit", type="pattern",
                      tags=["model-kits"], namespace="sess-a")
    await engine.save(content="started a Spitfire kit", type="pattern",
                      tags=["Model-Kits"], namespace="sess-b")
    await engine.save(content="unrelated", type="pattern",
                      tags=["cooking"], namespace="sess-a")
    grouped = await engine.group_by_entity("model-kits")
    assert len(grouped) == 2
    contents = {e.content for e in grouped}
    assert contents == {"bought an F-15 kit", "started a Spitfire kit"}


async def test_group_by_entity_no_false_merge(store: SQLiteStore):
    engine = ExperienceEngine(store, EventBus())
    await engine.save(content="a", type="pattern", tags=["aviation"])
    await engine.save(content="b", type="pattern", tags=["automobiles"])
    assert len(await engine.group_by_entity("aviation")) == 1
    assert len(await engine.group_by_entity("automobiles")) == 1
    assert await engine.group_by_entity("nonexistent") == []


# ---------------------------------------------------------------------------
# Phase 4: bi-temporal-aware recall (session/valid-time diversification + as-of)
# ---------------------------------------------------------------------------

async def test_search_bitemporal_fallback_identity_without_windows(store: SQLiteStore):
    """With no valid_from and no entity_key on any row, bitemporal recall returns
    the SAME ordered set as the baseline engine search (pure BM25 fallback)."""
    engine = ExperienceEngine(store, EventBus())
    for i in range(5):
        await engine.save(content=f"alpha beta gamma item {i}", type="pattern")
    base = await engine.search(text="alpha beta", limit=8)
    bit = await engine.search_bitemporal(text="alpha beta", limit=8)
    assert [e.id for e in base] == [e.id for e in bit]


async def test_search_bitemporal_diversifies_by_session(store: SQLiteStore):
    """When one session has many matching turns and another has one, baseline
    top-k can miss the second session; bitemporal diversification surfaces it."""
    engine = ExperienceEngine(store, EventBus())
    # Session A (valid_from = day 1): 8 matching, verbose turns
    for i in range(8):
        await engine.save(content=f"project alpha update {i}", type="pattern",
                          valid_from="2023/01/01 (Sun) 10:00")
    # Session B (valid_from = day 2): a single matching turn
    await engine.save(content="project alpha kickoff", type="pattern",
                      valid_from="2023/01/02 (Mon) 10:00")
    base = await engine.search(text="project alpha", limit=3)
    bit = await engine.search_bitemporal(text="project alpha", limit=3)
    base_sessions = {e.valid_from for e in base}
    bit_sessions = {e.valid_from for e in bit}
    # bitemporal surfaces BOTH sessions in the top-3; baseline may miss day-2
    assert "2023/01/02 (Mon) 10:00" in bit_sessions
    assert len(bit_sessions) >= len(base_sessions)


async def test_search_bitemporal_as_of_filters_future_validity(store: SQLiteStore):
    """as_of=T excludes facts whose valid_from is AFTER T (not yet true);
    NULL-valid_from facts are always eligible."""
    engine = ExperienceEngine(store, EventBus())
    await engine.save(content="price was ten dollars", type="pattern",
                      valid_from="2023/01/01 (Sun) 10:00")
    await engine.save(content="price became twenty dollars", type="pattern",
                      valid_from="2023/03/01 (Wed) 10:00")
    # As of February, only the January fact is valid.
    res = await engine.search_bitemporal(text="price dollars", limit=8,
                                         as_of="2023/02/01 (Wed) 00:00")
    froms = {e.valid_from for e in res}
    assert "2023/01/01 (Sun) 10:00" in froms
    assert "2023/03/01 (Wed) 10:00" not in froms


async def test_search_bitemporal_as_of_is_day_granular(store: SQLiteStore):
    """Regression repro for LongMemEval-S question gpt4_76048e76 - see
    ExperienceEngine.search_bitemporal's docstring (criterion 1) for the
    full day-vs-minute-granularity rationale.
    """
    engine = ExperienceEngine(store, EventBus())
    await engine.save(content="took the bike to the shop", type="pattern",
                      valid_from="2023/03/10 (Fri) 07:55")
    await engine.save(content="took the car to the shop", type="pattern",
                      valid_from="2023/03/11 (Sat) 07:55")
    res = await engine.search_bitemporal(text="shop", limit=8,
                                         as_of="2023/03/10 (Fri) 03:39")
    froms = {e.valid_from for e in res}
    assert "2023/03/10 (Fri) 07:55" in froms
    assert "2023/03/11 (Sat) 07:55" not in froms


def test_normalize_date_prefix_conventions():
    """Both real-world date conventions canonicalize to the same ISO day;
    non-date strings carry no validity signal (None), never an error."""
    assert normalize_date_prefix("2023/05/20 (Sat) 02:21") == "2023-05-20"
    assert normalize_date_prefix("2023-05-20T03:39:00+00:00") == "2023-05-20"
    assert normalize_date_prefix("2023-05-20 03:39") == "2023-05-20"
    assert normalize_date_prefix("  2023/05/20") == "2023-05-20"
    assert normalize_date_prefix("unknown date") is None
    assert normalize_date_prefix("") is None
    assert normalize_date_prefix(None) is None


async def test_search_bitemporal_as_of_mixed_date_conventions(store: SQLiteStore):
    """as-of filtering is correct when valid_from and as_of use DIFFERENT
    date-separator conventions (previously a documented silent-breakage
    caveat of the raw string-slice compare)."""
    engine = ExperienceEngine(store, EventBus())
    # Slash-form valid_from, ISO as_of
    await engine.save(content="price was ten dollars", type="pattern",
                      valid_from="2023/01/01 (Sun) 10:00")
    await engine.save(content="price became twenty dollars", type="pattern",
                      valid_from="2023/03/01 (Wed) 10:00")
    res = await engine.search_bitemporal(text="price dollars", limit=8,
                                         as_of="2023-02-01T00:00:00")
    froms = {e.valid_from for e in res}
    assert "2023/01/01 (Sun) 10:00" in froms
    assert "2023/03/01 (Wed) 10:00" not in froms

    # ISO valid_from, slash-form as_of
    await engine.save(content="office moved to Berlin", type="pattern",
                      valid_from="2023-01-01T10:00:00")
    await engine.save(content="office moved to Hamburg", type="pattern",
                      valid_from="2023-03-01T10:00:00")
    res = await engine.search_bitemporal(text="office moved", limit=8,
                                         as_of="2023/02/01 (Wed) 00:00")
    froms = {e.valid_from for e in res}
    assert "2023-01-01T10:00:00" in froms
    assert "2023-03-01T10:00:00" not in froms


async def test_search_bitemporal_unparseable_valid_from_stays_eligible(store: SQLiteStore):
    """A valid_from with no recognizable calendar day carries no validity
    signal: the experience must never be silently dropped by the as-of filter."""
    engine = ExperienceEngine(store, EventBus())
    await engine.save(content="ordered a standing desk", type="pattern",
                      valid_from="unknown date")
    res = await engine.search_bitemporal(text="standing desk", limit=8,
                                         as_of="2023/02/01 (Wed) 00:00")
    assert any(e.valid_from == "unknown date" for e in res)
