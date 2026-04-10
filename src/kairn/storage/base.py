"""Abstract storage interface for Kairn."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StorageBackend(ABC):
    """Abstract interface for Kairn storage backends."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize database schema and connections."""

    @abstractmethod
    async def close(self) -> None:
        """Close all connections."""

    # --- Node operations ---

    @abstractmethod
    async def insert_node(self, node: dict[str, Any]) -> dict[str, Any]:
        """Insert a node. Returns the inserted node."""

    @abstractmethod
    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Get a node by ID. Returns None if not found or soft-deleted."""

    @abstractmethod
    async def update_node(self, node_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update a node. Returns updated node or None."""

    @abstractmethod
    async def soft_delete_node(self, node_id: str) -> bool:
        """Soft-delete a node. Returns True if found and deleted."""

    @abstractmethod
    async def restore_node(self, node_id: str) -> bool:
        """Restore a soft-deleted node. Returns True if found and restored."""

    @abstractmethod
    async def query_nodes(
        self,
        *,
        namespace: str | None = None,
        node_type: str | None = None,
        tags: list[str] | None = None,
        text: str | None = None,
        visibility: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query nodes with filters. FTS5 used when text is provided."""

    @abstractmethod
    async def count_nodes(self, *, namespace: str | None = None) -> int:
        """Count non-deleted nodes."""

    # --- Edge operations ---

    @abstractmethod
    async def insert_edge(self, edge: dict[str, Any]) -> dict[str, Any]:
        """Insert an edge. Returns the inserted edge."""

    @abstractmethod
    async def get_edges(
        self,
        *,
        source_id: str | None = None,
        target_id: str | None = None,
        edge_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get edges by source, target, or type."""

    @abstractmethod
    async def delete_edge(self, source_id: str, target_id: str, edge_type: str) -> bool:
        """Delete an edge. Returns True if found and deleted."""

    @abstractmethod
    async def count_edges(self) -> int:
        """Count edges."""

    # --- Experience operations ---

    @abstractmethod
    async def insert_experience(self, experience: dict[str, Any]) -> dict[str, Any]:
        """Insert an experience. Returns the inserted experience."""

    @abstractmethod
    async def get_experience(self, exp_id: str) -> dict[str, Any] | None:
        """Get experience by ID."""

    @abstractmethod
    async def update_experience(
        self, exp_id: str, updates: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Update an experience. Returns updated or None."""

    @abstractmethod
    async def query_experiences(
        self,
        *,
        exp_type: str | None = None,
        text: str | None = None,
        min_score: float | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query experiences with filters. FTS5 used when text is provided."""

    @abstractmethod
    async def increment_access_count(self, exp_id: str) -> dict[str, Any] | None:
        """Increment access count and update last_accessed. Returns updated experience."""

    @abstractmethod
    async def touch_accessed_experiences(self, exp_ids: list[str]) -> int:
        """Batch-increment access_count for multiple experiences in a single UPDATE.

        Used by the intelligence layer read path (recall/context/crossref) so
        that searching N experiences does not require N round-trips. Returns
        the number of rows affected. Empty list input must be a no-op that
        returns 0.
        """

    @abstractmethod
    async def delete_experience(self, exp_id: str) -> bool:
        """Hard-delete an experience."""

    @abstractmethod
    async def get_promotable_experiences(
        self, *, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Get experiences flagged for promotion (needs_promotion=1).

        Args:
            limit: optional SQL-side LIMIT so a large promotion backlog
                does not load the full set into memory.
        """

    @abstractmethod
    async def promote_experience_atomic(
        self,
        experience_id: str,
        node: dict[str, Any],
    ) -> bool:
        """Atomically insert a promoted-experience node and CAS-link it.

        Both the INSERT (into nodes) and the UPDATE (linking the source
        experience via promoted_to_node_id) MUST happen in a single
        transaction. The UPDATE MUST be conditional on
        `promoted_to_node_id IS NULL` so concurrent sweepers cannot
        create duplicate nodes for the same source experience.

        Returns True if this caller won the race and now owns the
        promotion, False if another writer claimed it first (in which
        case the implementation MUST roll back its own node insert
        so the database stays orphan-free).
        """

    # --- Project operations ---

    @abstractmethod
    async def insert_project(self, project: dict[str, Any]) -> dict[str, Any]:
        """Insert a project."""

    @abstractmethod
    async def get_project(self, project_id: str) -> dict[str, Any] | None:
        """Get project by ID."""

    @abstractmethod
    async def update_project(
        self, project_id: str, updates: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Update a project."""

    @abstractmethod
    async def list_projects(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        """List projects."""

    @abstractmethod
    async def set_active_project(self, project_id: str) -> bool:
        """Set a project as active (deactivates others)."""

    # --- Progress operations ---

    @abstractmethod
    async def insert_progress(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Insert a progress/failure entry."""

    @abstractmethod
    async def get_progress(
        self, project_id: str, *, entry_type: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get progress entries for a project."""

    # --- Idea operations ---

    @abstractmethod
    async def insert_idea(self, idea: dict[str, Any]) -> dict[str, Any]:
        """Insert an idea."""

    @abstractmethod
    async def get_idea(self, idea_id: str) -> dict[str, Any] | None:
        """Get idea by ID."""

    @abstractmethod
    async def update_idea(self, idea_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update an idea."""

    @abstractmethod
    async def list_ideas(
        self,
        *,
        status: str | None = None,
        category: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List ideas with filters."""

    # --- Route operations ---

    @abstractmethod
    async def upsert_route(self, keyword: str, node_ids: list[str], confidence: float) -> None:
        """Insert or update a context route."""

    @abstractmethod
    async def get_routes(self, keywords: list[str]) -> list[dict[str, Any]]:
        """Get routes matching keywords."""

    # --- Activity log ---

    @abstractmethod
    async def log_activity(self, entry: dict[str, Any]) -> None:
        """Log an activity entry."""

    @abstractmethod
    async def log_activities(self, entries: list[dict[str, Any]]) -> None:
        """Batch-log multiple activity entries in a single transaction.

        Used by the intelligence layer to log node access after queries
        without incurring per-entry commit overhead. Empty list is a no-op.
        """

    @abstractmethod
    async def get_activity_log(
        self, *, entity_type: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Get recent activity log entries."""

    # --- Stats ---

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """Get workspace statistics."""
