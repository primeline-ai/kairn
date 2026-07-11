"""Context Router — keyword-based node routing with progressive disclosure."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from kairn.events.bus import EventBus
from kairn.events.types import EventType
from kairn.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class ContextRouter:
    """Routes keywords to relevant graph nodes with confidence scoring."""

    def __init__(self, store: StorageBackend, event_bus: EventBus) -> None:
        self.store = store
        self.bus = event_bus

    async def route(
        self, text: str, *, limit: int = 10, min_confidence: float = 0.3
    ) -> list[dict[str, Any]]:
        """Extract keywords from text, find matching routes, return nodes."""
        keywords = self._extract_keywords(text)
        if not keywords:
            return []

        routes = await self.store.get_routes(keywords)
        if not routes:
            return []

        node_scores: dict[str, float] = {}
        for route in routes:
            if route["confidence"] < min_confidence:
                continue
            node_ids = route["node_ids"]
            if isinstance(node_ids, str):
                try:
                    node_ids = json.loads(node_ids)
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Corrupted node_ids in route: %s", route.get("keyword"))
                    continue
            for nid in node_ids:
                node_scores[nid] = max(node_scores.get(nid, 0), route["confidence"])

        sorted_ids = sorted(node_scores, key=lambda nid: node_scores[nid], reverse=True)

        # Collect until `limit` LIVE nodes are found instead of slicing first:
        # soft-deleted ids stay in route arrays deliberately (restore_node
        # keeps them routable), but they must not starve result slots
        # (weakness-audit rank 62).
        results = []
        for nid in sorted_ids:
            node = await self.store.get_node(nid)
            if node:
                results.append(
                    {
                        "node": node,
                        "confidence": node_scores[nid],
                    }
                )
                if len(results) >= limit:
                    break

        return results

    async def update_routes_for_node(
        self,
        node_id: str,
        name: str,
        description: str | None,
    ) -> None:
        """Extract keywords from node and create/update routes."""
        text = f"{name} {description or ''}"
        keywords = self._extract_keywords(text)

        for keyword in keywords:
            # Single-statement atomic merge in the store (weakness-audit rank
            # 14): the prior get_routes -> append -> upsert_route sequence
            # interleaved at its await points under concurrent writers and
            # silently dropped node_ids (last-write-wins).
            await self.store.merge_route_node_id(keyword, node_id)

        await self.bus.emit(EventType.ROUTE_UPDATED, {"node_id": node_id, "keywords": keywords})

    async def context(
        self, text: str, *, detail: str = "summary", limit: int = 10
    ) -> dict[str, Any]:
        """Get relevant context subgraph with progressive disclosure."""
        results = await self.route(text, limit=limit)

        nodes = []
        for r in results:
            node_data = r["node"]
            node_out = {
                "id": node_data["id"],
                "name": node_data["name"],
                "type": node_data["type"],
                "namespace": node_data.get("namespace"),
                "confidence": r["confidence"],
            }
            if detail != "summary":
                node_out["description"] = node_data.get("description")
                node_out["tags"] = node_data.get("tags")
                node_out["properties"] = node_data.get("properties")
            nodes.append(node_out)

        return {
            "_v": "1.0",
            "query": text,
            "detail": detail,
            "count": len(nodes),
            "nodes": nodes,
        }

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from text."""
        stop_words = {
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
            "between",
            "through",
            "after",
            "before",
            "above",
            "below",
            "and",
            "or",
            "but",
            "not",
            "no",
            "nor",
            "so",
            "yet",
            "both",
            "either",
            "neither",
            "each",
            "every",
            "all",
            "any",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "than",
            "too",
            "very",
            "just",
            "also",
            "how",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "my",
            "your",
            "his",
            "her",
            "its",
            "our",
            "their",
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
            "if",
            "then",
            "else",
            "when",
            "where",
            "why",
        }

        words = re.findall(r"[a-zA-Z0-9_-]+", text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        seen: set[str] = set()
        unique: list[str] = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)

        return unique[:20]
