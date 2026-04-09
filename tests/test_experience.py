"""Tests for Experience/Decay Memory engine."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from kairn.core.experience import (
    CONFIDENCE_MULTIPLIERS,
    HALF_LIVES,
    ExperienceEngine,
    decay_rate_from_half_life,
)
from kairn.events.bus import EventBus
from kairn.events.types import EventType
from kairn.models.experience import VALID_CONFIDENCES, VALID_TYPES


@pytest.fixture
async def engine(store):
    """Create ExperienceEngine with event bus."""
    bus = EventBus()
    return ExperienceEngine(store, bus)


@pytest.mark.asyncio
async def test_decay_rate_calculation():
    """Verify decay_rate = ln(2) / half_life."""
    half_life = 200.0
    expected = math.log(2) / 200.0
    actual = decay_rate_from_half_life(half_life)
    assert abs(actual - expected) < 1e-10


@pytest.mark.asyncio
async def test_save_experience_all_types(engine):
    """Test saving experience with each valid type."""
    for exp_type in VALID_TYPES:
        exp = await engine.save(
            content=f"Test {exp_type}",
            type=exp_type,
            context="test context",
            confidence="high",
            tags=["test"],
        )
        assert exp.type == exp_type
        assert exp.content == f"Test {exp_type}"
        assert exp.confidence == "high"
        assert exp.score == 1.0
        assert exp.access_count == 0
        assert exp.tags == ["test"]


@pytest.mark.asyncio
async def test_save_experience_all_confidences(engine):
    """Test saving experience with each confidence level."""
    for confidence in VALID_CONFIDENCES:
        exp = await engine.save(
            content=f"Test {confidence}",
            type="solution",
            confidence=confidence,
        )
        assert exp.confidence == confidence

        # Verify decay_rate calculation
        half_life = HALF_LIVES["solution"]
        multiplier = CONFIDENCE_MULTIPLIERS[confidence]
        expected_rate = math.log(2) * multiplier / half_life
        assert abs(exp.decay_rate - expected_rate) < 1e-10


@pytest.mark.asyncio
async def test_save_invalid_type(engine):
    """Test that invalid type raises error."""
    with pytest.raises(ValueError, match="Invalid experience type"):
        await engine.save(content="Test", type="invalid_type")


@pytest.mark.asyncio
async def test_save_invalid_confidence(engine):
    """Test that invalid confidence raises error."""
    with pytest.raises(ValueError, match="Invalid confidence level"):
        await engine.save(content="Test", type="solution", confidence="invalid")


@pytest.mark.asyncio
async def test_save_empty_content(engine):
    """Test that empty content raises error."""
    with pytest.raises(ValueError, match="Content cannot be empty"):
        await engine.save(content="", type="solution")


@pytest.mark.asyncio
async def test_save_emits_event(engine):
    """Test that save emits EXPERIENCE_CREATED event."""
    events = []

    async def handler(event_type, data):
        events.append((event_type, data))

    engine.event_bus.on(EventType.EXPERIENCE_CREATED, handler)

    exp = await engine.save(content="Test", type="solution")

    assert len(events) == 1
    assert events[0][0] == EventType.EXPERIENCE_CREATED
    assert events[0][1]["exp_id"] == exp.id
    assert events[0][1]["type"] == "solution"
    assert events[0][1]["confidence"] == "high"


@pytest.mark.asyncio
async def test_get_experience(engine):
    """Test retrieving experience by ID."""
    exp = await engine.save(content="Test", type="solution")
    retrieved = await engine.get(exp.id)

    assert retrieved is not None
    assert retrieved.id == exp.id
    assert retrieved.content == "Test"


@pytest.mark.asyncio
async def test_get_nonexistent(engine):
    """Test retrieving non-existent experience returns None."""
    result = await engine.get("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_decay_math_half_life_high_confidence(engine):
    """Test that after half_life days, relevance ≈ 0.5 for high confidence."""
    # Create experience 200 days ago (solution half-life)
    now = datetime.now(timezone.utc)
    past = now - timedelta(days=200)

    exp = await engine.save(
        content="Test solution",
        type="solution",
        confidence="high",
    )

    # Update created_at to simulate old experience
    await engine.store.db.execute(
        "UPDATE experiences SET created_at = ? WHERE id = ?",
        (past.isoformat(), exp.id),
    )
    await engine.store.db.commit()

    # Retrieve and check relevance
    exp = await engine.get(exp.id)
    relevance = exp.relevance(at=now)

    # After 1 half-life, relevance should be ~0.5
    assert abs(relevance - 0.5) < 0.01


@pytest.mark.asyncio
async def test_decay_math_low_confidence_4x_faster(engine):
    """Test that low confidence decays 4x faster than high."""
    now = datetime.now(timezone.utc)

    # Create high confidence experience
    exp_high = await engine.save(
        content="High confidence",
        type="solution",
        confidence="high",
    )

    # Create low confidence experience
    exp_low = await engine.save(
        content="Low confidence",
        type="solution",
        confidence="low",
    )

    # Set both to same creation time (50 days ago)
    past = now - timedelta(days=50)
    await engine.store.db.execute(
        "UPDATE experiences SET created_at = ? WHERE id IN (?, ?)",
        (past.isoformat(), exp_high.id, exp_low.id),
    )
    await engine.store.db.commit()

    # Retrieve and check relevance
    exp_high = await engine.get(exp_high.id)
    exp_low = await engine.get(exp_low.id)

    relevance_high = exp_high.relevance(at=now)
    relevance_low = exp_low.relevance(at=now)

    # Low confidence should have decayed to ~0.5 after 50 days (4x faster than 200)
    # High confidence should be at ~0.84 after 50 days
    assert abs(relevance_low - 0.5) < 0.01
    assert relevance_high > 0.84


@pytest.mark.asyncio
async def test_decay_math_medium_confidence_2x_faster(engine):
    """Test that medium confidence decays 2x faster than high."""
    now = datetime.now(timezone.utc)

    # Create high confidence experience
    exp_high = await engine.save(
        content="High confidence",
        type="solution",
        confidence="high",
    )

    # Create medium confidence experience
    exp_medium = await engine.save(
        content="Medium confidence",
        type="solution",
        confidence="medium",
    )

    # Set both to 100 days ago
    past = now - timedelta(days=100)
    await engine.store.db.execute(
        "UPDATE experiences SET created_at = ? WHERE id IN (?, ?)",
        (past.isoformat(), exp_high.id, exp_medium.id),
    )
    await engine.store.db.commit()

    # Retrieve and check relevance
    exp_high = await engine.get(exp_high.id)
    exp_medium = await engine.get(exp_medium.id)

    relevance_high = exp_high.relevance(at=now)
    relevance_medium = exp_medium.relevance(at=now)

    # Medium should decay 2x faster
    # High at 100 days: exp(-0.00347 * 100) ≈ 0.707
    # Medium at 100 days: exp(-0.00693 * 100) ≈ 0.5
    assert abs(relevance_medium - 0.5) < 0.01
    assert relevance_high > 0.7


@pytest.mark.asyncio
async def test_decay_different_types(engine):
    """Test that different types have different half-lives."""
    now = datetime.now(timezone.utc)

    # Create solution (200 day half-life) and workaround (50 day half-life)
    exp_solution = await engine.save(
        content="Solution",
        type="solution",
        confidence="high",
    )

    exp_workaround = await engine.save(
        content="Workaround",
        type="workaround",
        confidence="high",
    )

    # Set both to 50 days ago
    past = now - timedelta(days=50)
    await engine.store.db.execute(
        "UPDATE experiences SET created_at = ? WHERE id IN (?, ?)",
        (past.isoformat(), exp_solution.id, exp_workaround.id),
    )

    # Retrieve and check relevance
    exp_solution = await engine.get(exp_solution.id)
    exp_workaround = await engine.get(exp_workaround.id)

    relevance_solution = exp_solution.relevance(at=now)
    relevance_workaround = exp_workaround.relevance(at=now)

    # Workaround at 50 days (1 half-life) ≈ 0.5
    # Solution at 50 days (0.25 half-life) ≈ 0.84
    assert abs(relevance_workaround - 0.5) < 0.01
    assert relevance_solution > 0.83


@pytest.mark.asyncio
async def test_access_increments_count(engine):
    """Test that access increments access_count."""
    exp = await engine.save(content="Test", type="solution")
    assert exp.access_count == 0

    # Access once
    exp = await engine.access(exp.id)
    assert exp.access_count == 1

    # Access again
    exp = await engine.access(exp.id)
    assert exp.access_count == 2


@pytest.mark.asyncio
async def test_access_emits_event(engine):
    """Test that access emits EXPERIENCE_ACCESSED event."""
    events = []

    async def handler(event_type, data):
        events.append((event_type, data))

    engine.event_bus.on(EventType.EXPERIENCE_ACCESSED, handler)

    exp = await engine.save(content="Test", type="solution")
    await engine.access(exp.id)

    assert len(events) == 1
    assert events[0][0] == EventType.EXPERIENCE_ACCESSED
    assert events[0][1]["exp_id"] == exp.id
    assert events[0][1]["access_count"] == 1


@pytest.mark.asyncio
async def test_access_nonexistent(engine):
    """Test accessing non-existent experience returns None."""
    result = await engine.access("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_auto_promotion_at_5_accesses(engine):
    """Test that experience is promoted to node at 5 accesses."""
    exp = await engine.save(content="Important pattern", type="pattern")

    # Access 5 times to trigger auto-promotion
    for _ in range(5):
        await engine.store.increment_access_count(exp.id)

    # Check if flagged for promotion
    promotable = await engine.get_promotable()
    assert len(promotable) == 1
    assert promotable[0].id == exp.id

    # Access again - should trigger promotion
    exp = await engine.access(exp.id)

    # Should have node_id now
    assert exp.promoted_to_node_id is not None

    # Verify node was created
    node = await engine.store.get_node(exp.promoted_to_node_id)
    assert node is not None
    assert node["type"] == "promoted_experience"
    assert node["namespace"] == "knowledge"


@pytest.mark.asyncio
async def test_promotion_emits_event(engine):
    """Test that promotion emits EXPERIENCE_PROMOTED event."""
    events = []

    async def handler(event_type, data):
        events.append((event_type, data))

    engine.event_bus.on(EventType.EXPERIENCE_PROMOTED, handler)

    exp = await engine.save(content="Important", type="pattern")

    # Trigger promotion
    for _ in range(5):
        await engine.store.increment_access_count(exp.id)

    await engine.access(exp.id)

    # Find promotion event
    promotion_events = [e for e in events if e[0] == EventType.EXPERIENCE_PROMOTED]
    assert len(promotion_events) == 1
    assert promotion_events[0][1]["exp_id"] == exp.id
    assert "node_id" in promotion_events[0][1]


@pytest.mark.asyncio
async def test_search_by_text(engine):
    """Test searching experiences by text."""
    await engine.save(content="Python debugging tips", type="solution")
    await engine.save(content="JavaScript error handling", type="solution")
    await engine.save(content="Python performance optimization", type="pattern")

    # Search for Python
    results = await engine.search(text="Python")
    assert len(results) == 2
    assert all("Python" in r.content for r in results)


@pytest.mark.asyncio
async def test_search_by_type(engine):
    """Test searching experiences by type."""
    await engine.save(content="Solution 1", type="solution")
    await engine.save(content="Solution 2", type="solution")
    await engine.save(content="Pattern 1", type="pattern")

    results = await engine.search(exp_type="solution")
    assert len(results) == 2
    assert all(r.type == "solution" for r in results)


@pytest.mark.asyncio
async def test_search_with_min_relevance(engine):
    """Test searching with minimum relevance threshold."""
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=400)  # Very old, low relevance

    # Create fresh experience
    exp_fresh = await engine.save(content="Fresh solution", type="solution")

    # Create old experience
    exp_old = await engine.save(content="Old solution", type="solution")
    await engine.store.db.execute(
        "UPDATE experiences SET created_at = ? WHERE id = ?",
        (old.isoformat(), exp_old.id),
    )

    # Search with min_relevance that filters out old one
    results = await engine.search(text="solution", min_relevance=0.3)

    # Only fresh one should be returned
    assert len(results) == 1
    assert results[0].id == exp_fresh.id


@pytest.mark.asyncio
async def test_search_sorted_by_relevance(engine):
    """Test that search results are sorted by relevance descending."""
    now = datetime.now(timezone.utc)

    # Create experiences with different ages
    exp1 = await engine.save(content="Test 1", type="solution")
    exp2 = await engine.save(content="Test 2", type="solution")
    exp3 = await engine.save(content="Test 3", type="solution")

    # Make exp2 old, exp3 very old
    await engine.store.db.execute(
        "UPDATE experiences SET created_at = ? WHERE id = ?",
        ((now - timedelta(days=100)).isoformat(), exp2.id),
    )
    await engine.store.db.execute(
        "UPDATE experiences SET created_at = ? WHERE id = ?",
        ((now - timedelta(days=200)).isoformat(), exp3.id),
    )

    results = await engine.search(text="Test")

    # Should be sorted by relevance: exp1 (fresh), exp2 (100 days), exp3 (200 days)
    assert results[0].id == exp1.id
    assert results[1].id == exp2.id
    assert results[2].id == exp3.id


@pytest.mark.asyncio
async def test_search_with_limit_offset(engine):
    """Test search pagination with limit and offset."""
    for i in range(10):
        await engine.save(content=f"Test {i}", type="solution")

    # Get first page
    page1 = await engine.search(text="Test", limit=3, offset=0)
    assert len(page1) == 3

    # Get second page
    page2 = await engine.search(text="Test", limit=3, offset=3)
    assert len(page2) == 3

    # Ensure different results
    page1_ids = {r.id for r in page1}
    page2_ids = {r.id for r in page2}
    assert len(page1_ids & page2_ids) == 0


@pytest.mark.asyncio
async def test_prune_expired_experiences(engine):
    """Test pruning experiences below threshold."""
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=700)  # Very old (> 3.5 half-lives)

    # Create fresh and old experiences
    exp_fresh = await engine.save(content="Fresh", type="solution")
    exp_old = await engine.save(content="Old", type="solution")

    await engine.store.db.execute(
        "UPDATE experiences SET created_at = ? WHERE id = ?",
        (old.isoformat(), exp_old.id),
    )
    await engine.store.db.commit()

    # Prune with threshold 0.1
    pruned_ids = await engine.prune(threshold=0.1)

    # Old one should be pruned
    assert exp_old.id in pruned_ids
    assert exp_fresh.id not in pruned_ids

    # Verify deletion
    assert await engine.get(exp_old.id) is None
    assert await engine.get(exp_fresh.id) is not None


@pytest.mark.asyncio
async def test_prune_emits_events(engine):
    """Test that prune emits EXPERIENCE_PRUNED events."""
    events = []

    async def handler(event_type, data):
        events.append((event_type, data))

    engine.event_bus.on(EventType.EXPERIENCE_PRUNED, handler)

    now = datetime.now(timezone.utc)
    old = now - timedelta(days=700)  # Very old (> 3.5 half-lives)

    exp = await engine.save(content="Old", type="solution")
    await engine.store.db.execute(
        "UPDATE experiences SET created_at = ? WHERE id = ?",
        (old.isoformat(), exp.id),
    )
    await engine.store.db.commit()

    await engine.prune(threshold=0.1)

    # Should have pruned event
    pruned_events = [e for e in events if e[0] == EventType.EXPERIENCE_PRUNED]
    assert len(pruned_events) == 1
    assert pruned_events[0][1]["exp_id"] == exp.id


@pytest.mark.asyncio
async def test_get_promotable(engine):
    """Test getting all promotable experiences."""
    exp1 = await engine.save(content="Experience 1", type="pattern")
    exp2 = await engine.save(content="Experience 2", type="solution")
    exp3 = await engine.save(content="Experience 3", type="pattern")

    # Flag exp1 and exp3 for promotion
    for _ in range(5):
        await engine.store.increment_access_count(exp1.id)
        await engine.store.increment_access_count(exp3.id)

    promotable = await engine.get_promotable()

    assert len(promotable) == 2
    promotable_ids = {e.id for e in promotable}
    assert exp1.id in promotable_ids
    assert exp3.id in promotable_ids
    assert exp2.id not in promotable_ids


@pytest.mark.asyncio
async def test_promotion_creates_correct_node(engine):
    """Test that promoted experience creates node with correct properties."""
    exp = await engine.save(
        content="Important pattern about async handling",
        type="pattern",
        context="Found while debugging async/await issues",
        tags=["async", "python"],
    )

    # Trigger promotion
    for _ in range(5):
        await engine.store.increment_access_count(exp.id)

    exp = await engine.access(exp.id)

    # Verify node properties
    node = await engine.store.get_node(exp.promoted_to_node_id)
    assert node["type"] == "promoted_experience"
    assert node["namespace"] == "knowledge"
    assert "pattern" in node["name"].lower() or "async" in node["name"].lower()
    assert node["description"] is not None


@pytest.mark.asyncio
async def test_promoted_experience_not_promoted_again(engine):
    """Test that already promoted experience is not promoted again."""
    exp = await engine.save(content="Test", type="pattern")

    # Trigger promotion
    for _ in range(5):
        await engine.store.increment_access_count(exp.id)

    exp = await engine.access(exp.id)
    first_node_id = exp.promoted_to_node_id

    # Access again - should not create new node
    exp = await engine.access(exp.id)
    assert exp.promoted_to_node_id == first_node_id


# ──────────────────────────────────────────────────────────────────────
# Namespace tests (Phase 3: multi-tenant isolation)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_namespace_default(engine):
    """Saving without namespace arg defaults to 'knowledge'."""
    exp = await engine.save(content="Default namespace check", type="solution")
    assert exp.namespace == "knowledge"

    # Round-trip: reload from store and verify the column persisted.
    retrieved = await engine.get(exp.id)
    assert retrieved is not None
    assert retrieved.namespace == "knowledge"


@pytest.mark.asyncio
async def test_save_with_namespace(engine):
    """Explicit namespace persists round-trip via SQLite store."""
    exp = await engine.save(
        content="Primeline-scoped knowledge",
        type="pattern",
        namespace="primeline",
    )
    assert exp.namespace == "primeline"

    retrieved = await engine.get(exp.id)
    assert retrieved is not None
    assert retrieved.namespace == "primeline"
    assert retrieved.content == "Primeline-scoped knowledge"


@pytest.mark.asyncio
async def test_namespace_isolation_across_saves(engine):
    """Multiple saves with different namespaces remain independently tagged."""
    exp_a = await engine.save(
        content="Alpha tenant insight", type="solution", namespace="alpha"
    )
    exp_b = await engine.save(
        content="Beta tenant insight", type="solution", namespace="beta"
    )
    exp_default = await engine.save(content="Shared insight", type="solution")

    retrieved_a = await engine.get(exp_a.id)
    retrieved_b = await engine.get(exp_b.id)
    retrieved_default = await engine.get(exp_default.id)

    assert retrieved_a.namespace == "alpha"
    assert retrieved_b.namespace == "beta"
    assert retrieved_default.namespace == "knowledge"


# ──────────────────────────────────────────────────────────────────────
# Batch access tracking
# ──────────────────────────────────────────────────────────────────────


class TestTouchAccessed:
    """Batch access tracking used by intelligence layer read path."""

    @pytest.mark.asyncio
    async def test_touch_accessed_single_id(self, engine):
        """Single-ID batch call increments access_count and last_accessed."""
        exp = await engine.save(
            content="touch accessed single",
            type="gotcha",
            confidence="high",
        )
        assert exp.access_count == 0

        await engine.touch_accessed([exp.id])

        updated = await engine.get(exp.id)
        assert updated.access_count == 1
        assert updated.last_accessed is not None

    @pytest.mark.asyncio
    async def test_touch_accessed_multiple_ids(self, engine):
        """Multi-ID batch call increments all targeted experiences."""
        ids = []
        for i in range(4):
            exp = await engine.save(
                content=f"batch-touch-{i}",
                type="pattern",
                confidence="high",
            )
            ids.append(exp.id)

        await engine.touch_accessed(ids)

        for exp_id in ids:
            updated = await engine.get(exp_id)
            assert updated.access_count == 1

    @pytest.mark.asyncio
    async def test_touch_accessed_empty_list_noop(self, engine):
        """Empty list must not issue SQL and must not crash."""
        # This must be safe to call even when no experiences were matched.
        result = await engine.touch_accessed([])
        assert result == 0  # rows affected count

    @pytest.mark.asyncio
    async def test_touch_accessed_nonexistent_id_tolerated(self, engine):
        """Passing an unknown ID alongside real ones affects only the real."""
        exp = await engine.save(
            content="real experience",
            type="gotcha",
            confidence="high",
        )
        assert exp.access_count == 0

        # Mix real ID with fake ones.
        await engine.touch_accessed([exp.id, "nonexistent-aaa", "nonexistent-bbb"])

        updated = await engine.get(exp.id)
        assert updated.access_count == 1

    @pytest.mark.asyncio
    async def test_touch_accessed_repeated_calls_compound(self, engine):
        """Calling touch_accessed N times increments count by N."""
        exp = await engine.save(
            content="repeated touch",
            type="gotcha",
            confidence="high",
        )

        for _ in range(3):
            await engine.touch_accessed([exp.id])

        updated = await engine.get(exp.id)
        assert updated.access_count == 3

    @pytest.mark.asyncio
    async def test_touch_accessed_5_triggers_promotion_flag(self, engine):
        """The SQL trigger exp_auto_promote fires at access_count >= 5."""
        exp = await engine.save(
            content="touch triggers promotion",
            type="gotcha",
            confidence="high",
        )

        # 5 touches in a batch (same experience, repeated).
        for _ in range(5):
            await engine.touch_accessed([exp.id])

        promotable = await engine.store.get_promotable_experiences()
        assert any(p["id"] == exp.id for p in promotable)


# ──────────────────────────────────────────────────────────────────────
# Promotion sweeper
# ──────────────────────────────────────────────────────────────────────


class TestPromotePending:
    """Sweeper that consumes the needs_promotion flag set by the
    exp_auto_promote SQL trigger and turns flagged experiences into
    permanent graph nodes. Without this sweeper, the flag is set but
    never acted on - experiences would stay flagged forever.
    """

    @pytest.mark.asyncio
    async def test_promote_pending_no_flagged_returns_zero(self, engine):
        """No flagged experiences -> stats all zero, no errors."""
        result = await engine.promote_pending()
        assert result["flagged"] == 0
        assert result["promoted"] == 0
        assert result["failed"] == 0
        assert result["nodes_created"] == []
        assert result["errors"] == []

    @pytest.mark.asyncio
    async def test_promote_pending_promotes_single_flagged(self, engine):
        """A flagged experience becomes a permanent node and is no
        longer in the promotable set after the sweep."""
        exp = await engine.save(
            content="solution worth promoting",
            type="solution",
            confidence="high",
        )

        # Flag it via 5 touch_accessed calls (trigger fires at >=5).
        for _ in range(5):
            await engine.touch_accessed([exp.id])

        # Pre-sweep: should be in promotable set.
        pre = await engine.store.get_promotable_experiences()
        assert any(p["id"] == exp.id for p in pre)

        # Sweep.
        result = await engine.promote_pending()
        assert result["flagged"] >= 1
        assert result["promoted"] >= 1
        assert result["failed"] == 0
        assert len(result["nodes_created"]) == result["promoted"]

        # Post-sweep: experience has promoted_to_node_id set, so it's
        # no longer in the promotable set.
        refreshed = await engine.get(exp.id)
        assert refreshed.promoted_to_node_id is not None
        assert refreshed.promoted_to_node_id in result["nodes_created"]

    @pytest.mark.asyncio
    async def test_promote_pending_creates_promoted_experience_node(self, engine):
        """Sweep produces a node with type='promoted_experience' that
        carries the source experience metadata."""
        exp = await engine.save(
            content="pattern X always wins under load",
            type="pattern",
            confidence="high",
            tags=["load-test", "pattern-x"],
        )
        for _ in range(5):
            await engine.touch_accessed([exp.id])

        result = await engine.promote_pending()
        assert result["promoted"] == 1
        node_id = result["nodes_created"][0]
        node = await engine.store.get_node(node_id)
        assert node is not None
        assert node["type"] == "promoted_experience"
        # Properties carry the source experience ID for traceability.
        assert "source_experience_id" in str(node)

    @pytest.mark.asyncio
    async def test_promote_pending_idempotent(self, engine):
        """Re-running on a clean DB is a no-op."""
        await engine.promote_pending()
        result = await engine.promote_pending()
        assert result["flagged"] == 0
        assert result["promoted"] == 0

    @pytest.mark.asyncio
    async def test_promote_pending_does_not_double_promote(self, engine):
        """Already-promoted experiences are not in the promotable set
        because get_promotable_experiences filters
        promoted_to_node_id IS NULL."""
        exp = await engine.save(content="x", type="gotcha", confidence="high")
        for _ in range(5):
            await engine.touch_accessed([exp.id])

        first = await engine.promote_pending()
        assert first["promoted"] == 1
        first_node_id = first["nodes_created"][0]

        # Run sweep again — same experience must NOT promote a second time.
        second = await engine.promote_pending()
        assert second["flagged"] == 0
        assert second["promoted"] == 0

        # Verify node count for this exp didn't double.
        refreshed = await engine.get(exp.id)
        assert refreshed.promoted_to_node_id == first_node_id

    @pytest.mark.asyncio
    async def test_promote_pending_respects_limit(self, engine):
        """Sweep with limit=N promotes at most N flagged experiences."""
        flagged_ids = []
        for i in range(5):
            exp = await engine.save(
                content=f"flagged item {i}",
                type="gotcha",
                confidence="high",
            )
            for _ in range(5):
                await engine.touch_accessed([exp.id])
            flagged_ids.append(exp.id)

        result = await engine.promote_pending(limit=3)
        assert result["flagged"] >= 5  # all 5 are findable
        assert result["promoted"] == 3  # but only 3 processed this call

        # 2 should remain flagged for the next sweep.
        remaining = await engine.store.get_promotable_experiences()
        assert len(remaining) == 2

        # A second sweep finishes the rest.
        result2 = await engine.promote_pending()
        assert result2["promoted"] == 2

        # Now everything is promoted.
        final = await engine.store.get_promotable_experiences()
        assert len(final) == 0

    @pytest.mark.asyncio
    async def test_promote_pending_does_not_increment_access_count(self, engine):
        """The sweep is a one-shot promote, NOT another access. The
        access_count must NOT change as a side effect of sweeping
        (touch_accessed already counted those reads)."""
        exp = await engine.save(content="x", type="pattern", confidence="high")
        for _ in range(5):
            await engine.touch_accessed([exp.id])

        before = (await engine.get(exp.id)).access_count
        await engine.promote_pending()
        after = (await engine.get(exp.id)).access_count

        assert after == before  # promotion is NOT another read
