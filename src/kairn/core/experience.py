"""Experience/Decay Memory engine for Kairn.

Manages experiences with exponential decay based on type-specific half-lives
and confidence multipliers. Automatically promotes frequently accessed experiences
to the knowledge graph.
"""

import logging
import math
from datetime import UTC, datetime
from typing import Any

from kairn.events.bus import EventBus
from kairn.events.types import EventType
from kairn.models.experience import VALID_CONFIDENCES, VALID_TYPES, Experience
from kairn.models.node import Node
from kairn.storage.base import StorageBackend

logger = logging.getLogger(__name__)

# Default half-lives by type (in days)
HALF_LIVES: dict[str, float] = {
    "solution": 200,
    "pattern": 300,
    "decision": 100,
    "workaround": 50,
    "gotcha": 200,
}

# Confidence multipliers for decay_rate
CONFIDENCE_MULTIPLIERS: dict[str, float] = {
    "high": 1.0,
    "medium": 2.0,  # 2x faster decay
    "low": 4.0,  # 4x faster decay
}


def decay_rate_from_half_life(half_life_days: float) -> float:
    """Convert half-life to decay rate: rate = ln(2) / half_life.

    Args:
        half_life_days: Half-life in days

    Returns:
        Decay rate constant
    """
    return math.log(2) / half_life_days


class ExperienceEngine:
    """Engine for managing experiences with decay and promotion."""

    def __init__(self, store: StorageBackend, event_bus: EventBus) -> None:
        """Initialize ExperienceEngine.

        Args:
            store: Storage backend
            event_bus: Event bus for publishing events
        """
        self.store = store
        self.event_bus = event_bus

    async def save(
        self,
        *,
        content: str,
        type: str,
        context: str | None = None,
        confidence: str = "high",
        tags: list[str] | None = None,
        namespace: str = "knowledge",
    ) -> Experience:
        """Save a new experience.

        Args:
            content: Experience content
            type: Experience type (must be in VALID_TYPES)
            context: Optional context information
            confidence: Confidence level (must be in VALID_CONFIDENCES)
            tags: Optional tags
            namespace: Namespace for multi-tenant isolation (default "knowledge")

        Returns:
            Created Experience

        Raises:
            ValueError: If type or confidence is invalid, or content is empty
        """
        # Validate content
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        # Validate type
        if type not in VALID_TYPES:
            raise ValueError(f"Invalid experience type: {type}. Must be one of {VALID_TYPES}")

        # Validate confidence
        if confidence not in VALID_CONFIDENCES:
            raise ValueError(
                f"Invalid confidence level: {confidence}. Must be one of {VALID_CONFIDENCES}"
            )

        # Calculate decay_rate
        half_life = HALF_LIVES[type]
        multiplier = CONFIDENCE_MULTIPLIERS[confidence]
        decay_rate = math.log(2) * multiplier / half_life

        # Create experience
        exp = Experience(
            namespace=namespace,
            type=type,
            content=content,
            context=context,
            confidence=confidence,
            score=1.0,
            decay_rate=decay_rate,
            tags=tags or [],
        )

        # Insert into store
        await self.store.insert_experience(exp.to_storage())

        # Emit event
        await self.event_bus.emit(
            EventType.EXPERIENCE_CREATED,
            {
                "exp_id": exp.id,
                "type": exp.type,
                "confidence": exp.confidence,
            },
        )

        logger.info(f"Created experience {exp.id} (type={type}, confidence={confidence})")

        return exp

    async def get(self, exp_id: str) -> Experience | None:
        """Get experience by ID.

        Args:
            exp_id: Experience ID

        Returns:
            Experience if found, None otherwise
        """
        data = await self.store.get_experience(exp_id)
        if data is None:
            return None
        return Experience(**data)

    async def search(
        self,
        *,
        text: str | None = None,
        exp_type: str | None = None,
        min_relevance: float = 0.0,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Experience]:
        """Search for experiences.

        Args:
            text: Text to search for (FTS5)
            exp_type: Filter by experience type
            min_relevance: Minimum relevance threshold
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of matching experiences, sorted by relevance descending
        """
        # Query from store
        results = await self.store.query_experiences(
            text=text,
            exp_type=exp_type,
            limit=100000,  # Get all first, filter by relevance
            offset=0,
        )

        # Convert to Experience objects and calculate current relevance
        now = datetime.now(UTC)
        experiences = []
        for data in results:
            exp = Experience(**data)
            current_relevance = exp.relevance(at=now)

            # Apply min_relevance filter
            if current_relevance >= min_relevance:
                experiences.append(exp)

        # Sort by relevance descending
        experiences.sort(key=lambda e: e.relevance(at=now), reverse=True)

        # Apply pagination
        return experiences[offset : offset + limit]

    async def touch_accessed(self, exp_ids: list[str]) -> int:
        """Batch-increment access_count for a list of experience IDs.

        Used by the intelligence layer read path so that returning N
        experiences from recall/context/crossref registers N access events
        in a single SQL round-trip. Fires the `exp_auto_promote` SQL trigger
        once per row when access_count crosses the threshold (SQLite
        triggers are always FOR EACH ROW, verified against the schema at
        `src/kairn/schema/triggers.sql`).

        Empty list is a no-op and returns 0 (the store layer is the
        load-bearing short-circuit — this wrapper does not pre-filter).

        Unknown IDs in the list are silently ignored (SQL WHERE IN filter).

        This does NOT perform the application-level promotion step (creating
        a node for a flagged experience). Promotion candidates are picked up
        later via `get_promotable()` and explicit `access()` calls, or by a
        background sweeper. The goal here is to keep the hot read path lean.

        Returns the number of rows affected.
        """
        return await self.store.touch_accessed_experiences(exp_ids)

    async def access(self, exp_id: str) -> Experience | None:
        """Access an experience (increments access count and checks for promotion).

        Args:
            exp_id: Experience ID

        Returns:
            Updated Experience if found, None otherwise
        """
        # Increment access count
        await self.store.increment_access_count(exp_id)

        # Get updated experience
        exp = await self.get(exp_id)
        if exp is None:
            return None

        # Emit access event
        await self.event_bus.emit(
            EventType.EXPERIENCE_ACCESSED,
            {
                "exp_id": exp.id,
                "access_count": exp.access_count,
            },
        )

        logger.info(f"Accessed experience {exp.id} (count={exp.access_count})")

        # Check for promotion
        if exp.promoted_to_node_id is None:
            # Check if needs promotion (flagged by trigger)
            promotable = await self.get_promotable()
            if any(p.id == exp.id for p in promotable):
                node = await self._promote(exp)
                if node:
                    # Refresh experience to get updated promoted_to_node_id
                    exp = await self.get(exp_id)

        return exp

    async def prune(self, *, threshold: float = 0.01) -> list[str]:
        """Prune experiences below relevance threshold.

        Args:
            threshold: Minimum relevance threshold

        Returns:
            List of pruned experience IDs
        """
        # Find all experiences below threshold
        now = datetime.now(UTC)
        all_experiences = await self.store.query_experiences(limit=100000, offset=0)

        pruned_ids = []
        for data in all_experiences:
            exp = Experience(**data)
            current_relevance = exp.relevance(at=now)
            if current_relevance < threshold:
                # Archive before deleting (log)
                logger.info(
                    f"Pruning experience {exp.id} "
                    f"(relevance={exp.relevance(at=now):.4f}, threshold={threshold})"
                )

                # Delete from store
                await self.store.delete_experience(exp.id)

                # Emit event
                await self.event_bus.emit(
                    EventType.EXPERIENCE_PRUNED,
                    {"exp_id": exp.id},
                )

                pruned_ids.append(exp.id)

        logger.info(f"Pruned {len(pruned_ids)} experiences")
        return pruned_ids

    async def promote_pending(self, *, limit: int = 100) -> dict[str, Any]:
        """Promote experiences flagged by the auto-promote SQL trigger.

        The `exp_auto_promote` trigger sets `properties.needs_promotion = 1`
        when `access_count` crosses 5. The flag stays set forever until
        something promotes the experience to a permanent graph node and
        clears the dead-letter state by setting `promoted_to_node_id`.

        This sweeper finds those flagged experiences and calls `_promote()`
        directly on each. Unlike `access()`, this does NOT increment
        access_count again (the trigger fired because the count already
        crossed the threshold via `touch_accessed()` from the read path).

        Race-safety: `_promote()` uses `store.promote_experience_atomic`
        which wraps INSERT + CAS UPDATE in one transaction. Concurrent
        sweepers cannot create duplicate nodes for the same source
        experience — the loser of the race gets `won=False` and the
        partially-inserted node is rolled back.

        Returns a stats dict:
            flagged_total: total experiences in the snapshot (pre-limit)
            attempted: number actually processed this call (= min(flagged_total, limit))
            promoted: attempted that became permanent nodes
            raced: attempted where another writer won the CAS (no orphan)
            failed: attempted where _promote raised an exception
            nodes_created: list of new node ids (length == promoted)
            errors: list of "exp_id: error" strings (capped at 10)

        Invariant: attempted == promoted + raced + failed.
        Idempotent: re-running on a clean DB is a no-op.

        Args:
            limit: maximum number of experiences to process per call.
                   Pushed down into SQL so a large backlog doesn't load
                   the full set into memory.
        """
        # Push the limit into SQL — bounded I/O, bounded memory.
        promotable = await self.get_promotable(limit=limit)
        flagged_total = len(promotable)
        attempted = 0
        promoted = 0
        raced = 0
        failed = 0
        nodes_created: list[str] = []
        errors: list[str] = []

        for exp in promotable:
            attempted += 1
            try:
                node = await self._promote(exp)
                if node is not None:
                    promoted += 1
                    nodes_created.append(node.id)
                else:
                    # _promote returned None: another writer won the CAS.
                    # The store has already rolled back the orphan node.
                    raced += 1
            except Exception as e:  # noqa: BLE001
                failed += 1
                errors.append(f"{exp.id}: {type(e).__name__}: {e}"[:200])
                logger.warning(
                    "promote_pending: failed to promote %s: %s", exp.id, e
                )

        if promoted:
            logger.info(
                "promote_pending: promoted %d/%d (raced=%d failed=%d)",
                promoted, attempted, raced, failed,
            )

        return {
            "flagged_total": flagged_total,
            "attempted": attempted,
            "promoted": promoted,
            "raced": raced,
            "failed": failed,
            "nodes_created": nodes_created,
            "errors": errors[:10],
        }

    async def _promote(self, experience: Experience) -> Node | None:
        """Promote experience to knowledge graph node.

        Atomic + race-safe via `store.promote_experience_atomic`. Returns
        None if another writer (e.g. a parallel sweeper, or a 2-MCP-server
        deployment) already claimed this experience between the
        get_promotable read and our write attempt. The atomic helper
        rolls back the partially-inserted node so the database stays
        orphan-free.

        Args:
            experience: Experience to promote

        Returns:
            Created Node on success, None if a race was lost or
            promotion failed mid-write.
        """
        # Build the candidate node (in-memory only at this point)
        node = Node(
            type="promoted_experience",
            name=f"{experience.type.capitalize()}: {experience.content[:50]}",
            namespace="knowledge",
            description=experience.content,
            properties={
                "source_experience_id": experience.id,
                "experience_type": experience.type,
                "confidence": experience.confidence,
                "access_count": experience.access_count,
                "tags": experience.tags,
            },
        )

        # Single transactional step: INSERT + CAS UPDATE.
        won = await self.store.promote_experience_atomic(
            experience.id, node.to_storage()
        )
        if not won:
            return None

        # Emit event
        await self.event_bus.emit(
            EventType.EXPERIENCE_PROMOTED,
            {
                "exp_id": experience.id,
                "node_id": node.id,
            },
        )

        logger.info(
            f"Promoted experience {experience.id} to node {node.id} "
            f"(access_count={experience.access_count})"
        )

        return node

    async def get_promotable(
        self, *, limit: int | None = None
    ) -> list[Experience]:
        """Get experiences flagged for promotion.

        Args:
            limit: optional SQL-side LIMIT pushed into the store query
                so a large promotion backlog does not load the full
                set into memory.

        Returns:
            List of experiences that need promotion (capped at `limit`
            if provided).
        """
        results = await self.store.get_promotable_experiences(limit=limit)
        return [Experience(**data) for data in results]
