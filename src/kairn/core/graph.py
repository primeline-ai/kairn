"""Graph engine — CRUD, FTS5 search, namespace filtering, pagination."""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import UTC, datetime
from typing import Any

from kairn.events.bus import EventBus
from kairn.events.types import EventType
from kairn.models.edge import Edge
from kairn.models.node import Node
from kairn.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class GraphEngine:
    """Knowledge graph operations over the storage backend."""

    def __init__(self, store: StorageBackend, event_bus: EventBus) -> None:
        self.store = store
        self.bus = event_bus

    async def add_node(
        self,
        *,
        name: str,
        type: str,
        namespace: str = "knowledge",
        description: str | None = None,
        properties: dict | None = None,
        tags: list[str] | None = None,
        visibility: str = "workspace",
        source_type: str = "manual",
    ) -> Node:
        """Create a node and auto-link via FTS5."""
        node = Node(
            type=type,
            name=name,
            namespace=namespace,
            description=description,
            properties=properties,
            tags=tags,
            visibility=visibility,
            source_type=source_type,
        )
        await self.store.insert_node(node.to_storage())
        await self._auto_link(node)
        await self.bus.emit(EventType.NODE_CREATED, {"node_id": node.id, "name": node.name})
        return node

    async def get_node(self, node_id: str) -> Node | None:
        data = await self.store.get_node(node_id)
        if not data:
            return None
        return Node(**data)

    async def update_node(self, node_id: str, **updates: Any) -> Node | None:
        updates["updated_at"] = datetime.now(UTC).isoformat()
        data = await self.store.update_node(node_id, updates)
        if not data:
            return None
        await self.bus.emit(EventType.NODE_UPDATED, {"node_id": node_id})
        return Node(**data)

    async def remove_node(self, node_id: str) -> bool:
        result = await self.store.soft_delete_node(node_id)
        if result:
            await self.bus.emit(EventType.NODE_DELETED, {"node_id": node_id})
        return result

    async def restore_node(self, node_id: str) -> bool:
        result = await self.store.restore_node(node_id)
        if result:
            await self.bus.emit(EventType.NODE_RESTORED, {"node_id": node_id})
        return result

    async def query(
        self,
        *,
        text: str | None = None,
        namespace: str | None = None,
        node_type: str | None = None,
        tags: list[str] | None = None,
        visibility: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Node]:
        rows = await self.store.query_nodes(
            namespace=namespace,
            node_type=node_type,
            tags=tags,
            text=text,
            visibility=visibility,
            limit=limit,
            offset=offset,
        )
        return [Node(**row) for row in rows]

    async def connect(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        *,
        weight: float = 1.0,
        properties: dict | None = None,
        created_by: str | None = None,
    ) -> Edge:
        """Create an edge between two nodes."""
        source = await self.store.get_node(source_id)
        target = await self.store.get_node(target_id)
        if not source:
            raise ValueError(f"Source node not found: {source_id}")
        if not target:
            raise ValueError(f"Target node not found: {target_id}")

        edge = Edge(
            source_id=source_id,
            target_id=target_id,
            type=edge_type,
            weight=weight,
            properties=properties,
            created_by=created_by,
        )
        await self.store.insert_edge(edge.to_storage())
        await self.bus.emit(
            EventType.EDGE_CREATED,
            {"source_id": source_id, "target_id": target_id, "type": edge_type},
        )
        return edge

    async def disconnect(self, source_id: str, target_id: str, edge_type: str) -> bool:
        result = await self.store.delete_edge(source_id, target_id, edge_type)
        if result:
            await self.bus.emit(
                EventType.EDGE_DELETED,
                {"source_id": source_id, "target_id": target_id, "type": edge_type},
            )
        return result

    async def get_edges(
        self,
        *,
        source_id: str | None = None,
        target_id: str | None = None,
        edge_type: str | None = None,
    ) -> list[Edge]:
        rows = await self.store.get_edges(
            source_id=source_id, target_id=target_id, edge_type=edge_type
        )
        return [Edge(**row) for row in rows]

    async def get_related(
        self, node_id: str, *, depth: int = 1, edge_type: str | None = None
    ) -> list[dict[str, Any]]:
        """BFS traversal from a node, returning connected nodes up to depth."""
        visited: set[str] = set()
        results: list[dict[str, Any]] = []
        queue: list[tuple[str, int]] = [(node_id, 0)]

        while queue:
            current_id, current_depth = queue.pop(0)
            if current_id in visited or current_depth > depth:
                continue
            visited.add(current_id)

            if current_id != node_id:
                node = await self.get_node(current_id)
                if node:
                    results.append({"node": node.to_response(), "depth": current_depth})

            if current_depth < depth:
                edges = await self.store.get_edges(source_id=current_id, edge_type=edge_type)
                edges += await self.store.get_edges(target_id=current_id, edge_type=edge_type)
                for edge in edges:
                    src = edge["source_id"]
                    neighbor = edge["target_id"] if src == current_id else src
                    if neighbor not in visited:
                        queue.append((neighbor, current_depth + 1))

        return results

    async def stats(self) -> dict[str, Any]:
        return await self.store.get_stats()

    async def _auto_link(self, node: Node) -> None:
        """Find related nodes via FTS5 and create edges."""
        search_text = f"{node.name} {node.description or ''}"
        words = re.findall(r"[a-zA-Z0-9_]+", search_text.lower())
        fts_reserved = {"and", "or", "not", "near"}
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "in",
            "for",
            "on",
            "with",
            "of",
            "to",
        }
        keywords = [
            w for w in words
            if w not in stop_words and w not in fts_reserved and len(w) > 2
        ]
        if not keywords:
            return
        # Quote each keyword to prevent FTS5 special-character interpretation
        # (aligned with _to_fts_query in intelligence.py).
        fts_query = " OR ".join(f'"{w}"' for w in keywords)

        try:
            related = await self.store.query_nodes(text=fts_query, limit=3)
        except (OSError, RuntimeError, sqlite3.Error) as exc:
            logger.warning("FTS5 auto-link search failed for %s: %s", node.id, exc)
            return

        for row in related:
            if row["id"] != node.id:
                try:
                    edge = Edge(
                        source_id=node.id,
                        target_id=row["id"],
                        type="auto_related",
                        weight=0.5,
                        created_by="auto_link",
                    )
                    await self.store.insert_edge(edge.to_storage())
                except (OSError, RuntimeError, sqlite3.IntegrityError):
                    logger.debug("Auto-link edge skipped: %s → %s", node.id, row["id"])
