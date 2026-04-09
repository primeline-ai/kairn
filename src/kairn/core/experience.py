"""Experience/Decay Memory engine for Kairn.

Manages experiences with exponential decay based on type-specific half-lives
and confidence multipliers. Automatically promotes frequently accessed experiences
to the knowledge graph.
"""

import logging
import math
from datetime import UTC, datetime

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

    async def _promote(self, experience: Experience) -> Node | None:
        """Promote experience to knowledge graph node.

        Args:
            experience: Experience to promote

        Returns:
            Created Node, or None if promotion failed
        """
        # Create node from experience
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

        # Insert node
        await self.store.insert_node(node.to_storage())

        # Update experience with node_id
        await self.store.update_experience(
            experience.id,
            {"promoted_to_node_id": node.id},
        )

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

    async def get_promotable(self) -> list[Experience]:
        """Get experiences flagged for promotion.

        Returns:
            List of experiences that need promotion
        """
        results = await self.store.get_promotable_experiences()
        return [Experience(**data) for data in results]
