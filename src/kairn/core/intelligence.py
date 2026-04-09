"""Intelligence layer — learn, recall, crossref, context, related.

Bridges graph, experience, and router engines into unified knowledge operations.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any

from kairn.core.experience import ExperienceEngine
from kairn.core.graph import GraphEngine
from kairn.core.ideas import IdeaEngine
from kairn.core.memory import ProjectMemory
from kairn.core.router import ContextRouter
from kairn.events.bus import EventBus
from kairn.events.types import EventType
from kairn.models.experience import VALID_CONFIDENCES, VALID_TYPES
from kairn.storage.base import StorageBackend

logger = logging.getLogger(__name__)

_STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "shall",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "about",
    "and",
    "or",
    "but",
    "not",
    "no",
    "so",
    "yet",
    "i",
    "me",
    "we",
    "us",
    "you",
    "he",
    "she",
    "it",
    "they",
    "them",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "this",
    "that",
    "these",
    "those",
    "need",
    "want",
    "try",
}


def _to_fts_query(text: str) -> str | None:
    """Convert natural language to FTS5 OR query with proper escaping."""
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    fts_reserved = {"and", "or", "not", "near"}
    keywords = [w for w in words if w not in _STOP_WORDS and w not in fts_reserved and len(w) > 2]
    if not keywords:
        return None
    return " OR ".join(f'"{w}"' for w in keywords)


class IntelligenceLayer:
    """Unified intelligence operations over graph, experience, and routing."""

    def __init__(
        self,
        *,
        store: StorageBackend,
        event_bus: EventBus,
        graph: GraphEngine,
        router: ContextRouter,
        memory: ProjectMemory,
        experience: ExperienceEngine,
        ideas: IdeaEngine,
    ) -> None:
        self.store = store
        self.event_bus = event_bus
        self.graph = graph
        self.router = router
        self.memory = memory
        self.experience = experience
        self.ideas = ideas

    async def learn(
        self,
        *,
        content: str,
        type: str,
        context: str | None = None,
        confidence: str = "high",
        tags: list[str] | None = None,
        namespace: str = "knowledge",
    ) -> dict[str, Any]:
        """Store knowledge from conversation.

        High confidence creates a permanent node + experience.
        Medium/low confidence creates a decaying experience only.

        The `namespace` parameter isolates knowledge across tenants/projects.
        It is applied to both the high-confidence graph node and the
        backing experience record.
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")
        if type not in VALID_TYPES:
            raise ValueError(f"Invalid type: {type}. Must be one of {VALID_TYPES}")
        if confidence not in VALID_CONFIDENCES:
            raise ValueError(
                f"Invalid confidence: {confidence}. Must be one of {VALID_CONFIDENCES}"
            )

        content = content.strip()
        node_id: str | None = None
        experience_id: str | None = None

        if confidence == "high":
            # Create permanent node
            node = await self.graph.add_node(
                name=f"{type.capitalize()}: {content[:60]}",
                type=f"learned_{type}",
                namespace=namespace,
                description=content,
                tags=tags,
                source_type="learn",
            )
            node_id = node.id

            # Update routes for discoverability
            await self.router.update_routes_for_node(node.id, node.name, node.description)

        # Always create experience (for decay tracking)
        exp = await self.experience.save(
            content=content,
            type=type,
            context=context,
            confidence=confidence,
            tags=tags,
            namespace=namespace,
        )
        experience_id = exp.id

        stored_as = "node" if confidence == "high" else "experience"

        await self.event_bus.emit(
            EventType.KNOWLEDGE_LEARNED,
            {
                "stored_as": stored_as,
                "node_id": node_id,
                "experience_id": experience_id,
                "type": type,
                "confidence": confidence,
            },
        )

        logger.info(
            "Learned %s (confidence=%s, stored_as=%s)",
            type,
            confidence,
            stored_as,
        )

        return {
            "_v": "1.0",
            "stored_as": stored_as,
            "node_id": node_id,
            "experience_id": experience_id,
            "type": type,
            "confidence": confidence,
            "namespace": namespace,
        }

    async def recall(
        self,
        *,
        topic: str | None = None,
        limit: int = 10,
        min_relevance: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Surface relevant past knowledge for context.

        Searches across both nodes (permanent) and experiences (decaying).
        Returns combined, ranked results.
        """
        results: list[dict[str, Any]] = []
        fts_query = _to_fts_query(topic) if topic else None

        # Search nodes via FTS5
        if fts_query:
            nodes = await self.graph.query(text=fts_query, limit=limit)
        else:
            nodes = await self.graph.query(limit=limit)

        for node in nodes:
            results.append(
                {
                    "source": "node",
                    "id": node.id,
                    "name": node.name,
                    "type": node.type,
                    "description": node.description,
                    "relevance": 1.0,  # Nodes are permanent, full relevance
                }
            )

        # Search experiences (decay-aware)
        experiences = await self.experience.search(
            text=fts_query,
            min_relevance=min_relevance,
            limit=limit,
        )

        now = datetime.now(UTC)
        for exp in experiences:
            results.append(
                {
                    "source": "experience",
                    "id": exp.id,
                    "type": exp.type,
                    "content": exp.content,
                    "confidence": exp.confidence,
                    "relevance": round(exp.relevance(at=now), 4),
                }
            )

        # Sort by relevance descending
        results.sort(key=lambda r: r["relevance"], reverse=True)

        # Apply limit to combined results
        results = results[:limit]

        await self.event_bus.emit(
            EventType.KNOWLEDGE_RECALLED,
            {"topic": topic, "result_count": len(results)},
        )

        return results

    async def crossref(
        self,
        *,
        problem: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find similar solutions from current workspace.

        In V1 (single workspace), searches nodes and experiences.
        In team mode, will search across accessible workspaces.
        """
        if not problem or not problem.strip():
            raise ValueError("Problem description cannot be empty")

        problem = problem.strip()
        fts_query = _to_fts_query(problem)
        results: list[dict[str, Any]] = []

        # Search nodes for solutions/patterns
        if fts_query:
            nodes = await self.graph.query(text=fts_query, limit=limit)
        else:
            nodes = []
        for node in nodes:
            results.append(
                {
                    "source": "node",
                    "workspace": "default",
                    "id": node.id,
                    "name": node.name,
                    "type": node.type,
                    "description": node.description,
                    "relevance": 1.0,
                }
            )

        # Search experiences for solutions
        experiences = await self.experience.search(
            text=fts_query,
            min_relevance=0.1,
            limit=limit,
        )

        now = datetime.now(UTC)
        for exp in experiences:
            results.append(
                {
                    "source": "experience",
                    "workspace": "default",
                    "id": exp.id,
                    "type": exp.type,
                    "content": exp.content,
                    "confidence": exp.confidence,
                    "relevance": round(exp.relevance(at=now), 4),
                }
            )

        results.sort(key=lambda r: r["relevance"], reverse=True)
        results = results[:limit]

        await self.event_bus.emit(
            EventType.CROSSREF_FOUND,
            {"problem": problem, "result_count": len(results)},
        )

        return results

    async def context(
        self,
        *,
        keywords: str,
        detail: str = "summary",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Get relevant context subgraph with progressive disclosure.

        Combines router-based node discovery with experience search.
        """
        if not keywords or not keywords.strip():
            return {
                "_v": "1.0",
                "query": keywords,
                "detail": detail,
                "count": 0,
                "nodes": [],
                "experiences": [],
            }

        keywords = keywords.strip()
        fts_query = _to_fts_query(keywords)

        # Route-based node discovery
        route_results = await self.router.route(keywords, limit=limit)

        nodes = []
        for r in route_results:
            node_data = r["node"]
            node_out: dict[str, Any] = {
                "id": node_data["id"],
                "name": node_data["name"],
                "type": node_data["type"],
                "confidence": r["confidence"],
            }
            if detail != "summary":
                node_out["description"] = node_data.get("description")
                node_out["tags"] = node_data.get("tags")
                node_out["properties"] = node_data.get("properties")
            nodes.append(node_out)

        # Also search by FTS5 if router found nothing
        if not nodes and fts_query:
            fts_nodes = await self.graph.query(text=fts_query, limit=limit)
            for n in fts_nodes:
                node_out = {
                    "id": n.id,
                    "name": n.name,
                    "type": n.type,
                    "confidence": 0.5,
                }
                if detail != "summary":
                    node_out["description"] = n.description
                    node_out["tags"] = n.tags
                    node_out["properties"] = n.properties
                nodes.append(node_out)

        # Experience search
        experiences = await self.experience.search(
            text=fts_query,
            min_relevance=0.1,
            limit=limit,
        )

        now = datetime.now(UTC)
        exp_items = []
        for e in experiences:
            exp_out: dict[str, Any] = {
                "id": e.id,
                "type": e.type,
                "content": e.content[:200] if detail == "summary" else e.content,
                "relevance": round(e.relevance(at=now), 4),
            }
            if detail != "summary":
                exp_out["confidence"] = e.confidence
                exp_out["tags"] = e.tags
                exp_out["context"] = e.context
            exp_items.append(exp_out)

        total_count = len(nodes) + len(exp_items)

        return {
            "_v": "1.0",
            "query": keywords,
            "detail": detail,
            "count": total_count,
            "nodes": nodes,
            "experiences": exp_items,
        }

    async def related(
        self,
        *,
        node_id: str,
        depth: int = 1,
        edge_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find nodes connected to a starting point via BFS."""
        return await self.graph.get_related(node_id, depth=depth, edge_type=edge_type)
