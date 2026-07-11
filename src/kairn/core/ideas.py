"""Idea Lifecycle Engine.

Manages the creation, updating, and lifecycle of ideas with status transitions,
scoring, and graph linking.
"""

import logging
from datetime import UTC, datetime

from kairn.events.bus import EventBus
from kairn.events.types import EventType
from kairn.models.edge import Edge
from kairn.models.idea import VALID_STATUSES, Idea
from kairn.models.node import Node
from kairn.storage.base import StorageBackend

logger = logging.getLogger(__name__)

# Valid status transitions
VALID_TRANSITIONS: dict[str, set[str]] = {
    "draft": {"evaluating"},
    "evaluating": {"approved", "archived"},
    "approved": {"implementing", "archived"},
    "implementing": {"done", "archived"},
    "done": {"archived"},
    "archived": {"draft"},
}

# Happy path for advance()
ADVANCE_PATH: dict[str, str] = {
    "draft": "evaluating",
    "evaluating": "approved",
    "approved": "implementing",
    "implementing": "done",
}


class IdeaEngine:
    """Engine for managing idea lifecycle and operations."""

    def __init__(self, store: StorageBackend, event_bus: EventBus) -> None:
        """Initialize the IdeaEngine.

        Args:
            store: Storage backend for persistence
            event_bus: Event bus for emitting lifecycle events
        """
        self._store = store
        self._event_bus = event_bus

    async def create(
        self,
        *,
        title: str,
        category: str | None = None,
        score: float | None = None,
        properties: dict | None = None,
        visibility: str = "private",
    ) -> Idea:
        """Create a new idea with status="draft".

        Args:
            title: Idea title (required, non-empty)
            category: Optional category classification
            score: Optional numerical score
            properties: Optional metadata dictionary
            visibility: Visibility level (default: "private")

        Returns:
            Created Idea instance

        Raises:
            ValueError: If title is empty
        """
        if not title or not title.strip():
            raise ValueError("title cannot be empty")

        idea = Idea(
            title=title,
            status="draft",
            category=category,
            score=score,
            properties=properties or {},
            visibility=visibility,
        )

        await self._store.insert_idea(idea.to_storage())

        logger.info(f"Created idea: {idea.id} - {idea.title}")

        # Emit creation event
        await self._event_bus.emit(
            EventType.IDEA_CREATED,
            {"idea_id": idea.id, "title": idea.title},
        )

        return idea

    async def get(self, idea_id: str) -> Idea | None:
        """Get an idea by ID.

        Args:
            idea_id: Unique idea identifier

        Returns:
            Idea instance or None if not found
        """
        data = await self._store.get_idea(idea_id)
        if data is None:
            return None

        return Idea(**data)

    async def update(self, idea_id: str, **updates) -> Idea | None:
        """Update an idea with the given changes.

        If status is being updated, validates the transition is allowed.

        Args:
            idea_id: Unique idea identifier
            **updates: Fields to update (title, status, category, score, properties, visibility)

        Returns:
            Updated Idea instance or None if not found

        Raises:
            ValueError: If status transition is invalid
        """
        # Get current idea
        current = await self.get(idea_id)
        if current is None:
            return None

        # Validate status transition if status is being changed
        if "status" in updates:
            new_status = updates["status"]
            if new_status not in VALID_STATUSES:
                raise ValueError(f"Invalid status: {new_status}")

            if new_status != current.status:
                self._validate_status_transition(current.status, new_status)

        # Track what changed
        changed_fields = []
        for key, value in updates.items():
            if getattr(current, key, None) != value:
                changed_fields.append(key)

        # Update the idea. For a status change, pass the snapshot status as a
        # compare-and-set guard: the transition was validated against that
        # snapshot, so it must only land if the row still holds it - otherwise
        # two racers could each pass validation and silently overwrite each
        # other (weakness-audit rank 65).
        updates["updated_at"] = datetime.now(UTC)
        expected = current.status if "status" in updates else None
        result = await self._store.update_idea(
            idea_id, updates, expected_status=expected
        )
        if result is None and expected is not None:
            raise ValueError(
                f"Concurrent status change on idea {idea_id}: expected "
                f"'{expected}' but the row moved on - re-read and retry"
            )

        # Get updated idea
        updated = await self.get(idea_id)

        if updated and changed_fields:
            logger.info(f"Updated idea {idea_id}: {', '.join(changed_fields)}")

            # Emit update event
            await self._event_bus.emit(
                EventType.IDEA_UPDATED,
                {"idea_id": idea_id, "changes": changed_fields},
            )

        return updated

    async def list_ideas(
        self,
        *,
        status: str | None = None,
        category: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Idea]:
        """List ideas with optional filtering.

        Args:
            status: Filter by status
            category: Filter by category
            limit: Maximum number of results (default: 10)
            offset: Number of results to skip (default: 0)

        Returns:
            List of Idea instances matching the criteria
        """
        data_list = await self._store.list_ideas(
            status=status,
            category=category,
            limit=limit,
            offset=offset,
        )

        return [Idea(**data) for data in data_list]

    async def link_to_node(
        self,
        idea_id: str,
        node_id: str,
        edge_type: str = "idea_relates_to",
    ) -> Edge | None:
        """Link an idea to a knowledge graph node.

        Creates a temporary idea node in the graph if it doesn't exist,
        then creates an edge from the idea to the target node.

        Args:
            idea_id: Idea to link from
            node_id: Target node to link to
            edge_type: Type of relationship (default: "idea_relates_to")

        Returns:
            Created Edge instance or None if idea/node not found
        """
        # Verify idea exists
        idea = await self.get(idea_id)
        if idea is None:
            logger.warning(f"Cannot link: idea {idea_id} not found")
            return None

        # Verify target node exists
        target_data = await self._store.get_node(node_id)
        if target_data is None:
            logger.warning(f"Cannot link: node {node_id} not found")
            return None

        # Create/ensure idea node exists in graph
        idea_node_data = await self._store.get_node(idea_id)
        if idea_node_data is None:
            # Create a node for the idea
            idea_node = Node(
                id=idea_id,
                namespace="idea",
                type="idea",
                name=idea.title,
                visibility=idea.visibility,
            )
            await self._store.insert_node(idea_node.to_storage())
            logger.info(f"Created idea node in graph: {idea_id}")

        # Create edge
        edge = Edge(
            source_id=idea_id,
            target_id=node_id,
            type=edge_type,
        )

        await self._store.insert_edge(edge.to_storage())

        logger.info(f"Linked idea {idea_id} to node {node_id} ({edge_type})")

        return edge

    async def advance(self, idea_id: str) -> Idea | None:
        """Advance an idea to the next status in the happy path lifecycle.

        Happy path: draft → evaluating → approved → implementing → done

        Args:
            idea_id: Idea to advance

        Returns:
            Updated Idea instance or None if already at end ("done", "archived") or not found
        """
        current = await self.get(idea_id)
        if current is None:
            return None

        # Check if there's a next status
        next_status = ADVANCE_PATH.get(current.status)
        if next_status is None:
            # Already at "done" or "archived" - no next step
            logger.debug(f"Idea {idea_id} at terminal status: {current.status}")
            return None

        # Advance to next status
        return await self.update(idea_id, status=next_status)

    def _validate_status_transition(self, from_status: str, to_status: str) -> None:
        """Validate a status transition is allowed.

        Args:
            from_status: Current status
            to_status: Desired status

        Raises:
            ValueError: If transition is not valid
        """
        allowed_targets = VALID_TRANSITIONS.get(from_status, set())
        if to_status not in allowed_targets:
            raise ValueError(
                f"Invalid status transition from '{from_status}' to '{to_status}'. "
                f"Allowed transitions: {sorted(allowed_targets)}"
            )
