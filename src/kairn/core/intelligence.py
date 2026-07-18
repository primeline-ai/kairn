"""Intelligence layer — learn, recall, crossref, context, related.

Bridges graph, experience, and router engines into unified knowledge operations.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from kairn.core.experience import ExperienceEngine
from kairn.core.fts import _to_fts_query  # used here + re-exported for back-compat
from kairn.core.graph import GraphEngine
from kairn.core.ideas import IdeaEngine
from kairn.core.memory import ProjectMemory
from kairn.core.router import ContextRouter
from kairn.events.bus import EventBus
from kairn.events.types import EventType
from kairn.models.experience import VALID_CONFIDENCES, VALID_TYPES
from kairn.storage.base import StorageBackend

logger = logging.getLogger(__name__)

# FTS5 query shaping (_to_fts_query, _STOP_WORDS) now lives in core.fts and is
# re-exported via the top-of-module import so the experience path can share it
# without an import cycle. See core/fts.py.


# Max candidates returned by kn_learn FTS5 scan. Set conservatively;
# Phase 0 bench confirmed p95 = 1.258ms for the 5-row LIMIT shape at
# the current 4923-node scale (`_autonomous/benchmarks/kairn-fts5-latency.json`).
_CANDIDATES_LIMIT = 5
_CANDIDATE_SNIPPET_CHARS = 160

# bm25 score at which node relevance = 0.5. Larger => the same bm25 match maps
# to a lower relevance, so weak keyword overlaps fall under a strict
# min_relevance floor while strong multi-term matches clear it.
_BM25_RELEVANCE_MIDPOINT = 5.0


def _bm25_to_relevance(rank: float | None) -> float:
    """Map an FTS5 bm25 `rank` to a bounded (0, 1] relevance.

    SQLite FTS5 exposes bm25 as a negative score where a more-negative value
    means a stronger match. A saturating transform (score / (score + K))
    preserves the raw bm25 ordering while yielding an absolute-ish relevance
    the min_relevance gate can act on. `rank is None` (a non-text browse query
    with no MATCH) has no match strength to report, so it stays 1.0.
    """
    if rank is None:
        return 1.0
    score = max(0.0, -float(rank))
    return round(score / (score + _BM25_RELEVANCE_MIDPOINT), 4)


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
        embedder: Callable[[list[str]], list[list[float]]] | None = None,
        embedder_model: str | None = None,
        semantic_recall: bool = False,
        semantic_floor: float = 0.5,
        semantic_top_n: int = 30,
    ) -> None:
        self.store = store
        self.event_bus = event_bus
        self.graph = graph
        self.router = router
        self.memory = memory
        self.experience = experience
        self.ideas = ideas
        # Optional semantic_recall (opt-in flag, default OFF). When on, recall's
        # node path reranks the FTS5 top-N by local-embedding cosine and abstains
        # below semantic_floor. All-None/False => the keyword path runs unchanged.
        self.embedder = embedder
        self.embedder_model = embedder_model
        self.semantic_recall = semantic_recall
        self.semantic_floor = semantic_floor
        self.semantic_top_n = semantic_top_n

    async def _log_node_access(
        self, activity_type: str, node_ids: list[str]
    ) -> None:
        """Best-effort batch-log of node accesses to activity_log.

        Fires after recall/context/crossref return nodes so downstream
        analytics can track which nodes are actually queried. Failures
        are logged and swallowed to keep the read path fail-open.
        """
        if not node_ids:
            return
        now = datetime.now(UTC).isoformat()
        entries = [
            {
                "id": str(uuid.uuid4())[:8],
                "user_id": None,
                "activity_type": activity_type,
                "entity_type": "node",
                "entity_id": nid,
                "description": None,
                "created_at": now,
            }
            for nid in node_ids
        ]
        try:
            await self.store.log_activities(entries)
        except Exception:
            logger.debug("Failed to log node access for %s", activity_type, exc_info=True)

    async def learn(
        self,
        *,
        content: str,
        type: str,
        context: str | None = None,
        confidence: str = "high",
        tags: list[str] | None = None,
        namespace: str = "knowledge",
        with_candidates: bool = True,
    ) -> dict[str, Any]:
        """Store knowledge from conversation.

        High confidence creates a permanent node + experience.
        Medium/low confidence creates a decaying experience only.

        The `namespace` parameter isolates knowledge across tenants/projects.
        It is applied to both the high-confidence graph node and the
        backing experience record.

        When `with_candidates=True` (default), runs a follow-up FTS5 scan
        over existing nodes using the saved content as the seed query.
        Returns up to `_CANDIDATES_LIMIT` semantically-related node
        snippets in the response envelope under the `candidates` key so
        the caller can decide whether to invoke `kn_judge` to assert a
        relationship verb (conflicts_with / supersedes / compatible /
        scoped / related). The just-created node (if any) is excluded
        from candidates. Set `with_candidates=False` for high-volume
        bulk-save scripts that do not need the judgment hook.
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

        response: dict[str, Any] = {
            "_v": "1.0",
            "stored_as": stored_as,
            "node_id": node_id,
            "experience_id": experience_id,
            "type": type,
            "confidence": confidence,
            "namespace": namespace,
        }

        if with_candidates:
            response["candidates"] = await self._scan_candidates(
                content=content,
                exclude_node_id=node_id,
                namespace=namespace,
            )

        return response

    async def _scan_candidates(
        self,
        *,
        content: str,
        exclude_node_id: str | None,
        namespace: str,
    ) -> list[dict[str, Any]]:
        """FTS5-scan for semantically-related existing nodes.

        Mirrors the Phase 0 benchmark query shape (validated p95 1.258ms
        at 4923-node scale). Used by `learn()` to surface judgment
        candidates without forcing the caller to issue a separate
        `kn_context` / `kn_recall` query.

        Returns up to `_CANDIDATES_LIMIT` candidate dicts, each shaped
        `{id, name, type, snippet, sim_rank}`. `snippet` is the node
        description truncated to `_CANDIDATE_SNIPPET_CHARS`; `sim_rank`
        is the integer position 0..N-1 in FTS5 rank order (lower is
        more relevant). Empty list when no FTS5 hits or the content
        produced no usable keywords.
        """
        fts_query = _to_fts_query(content)
        if not fts_query:
            return []

        # Over-fetch by one so we can drop the just-created node and
        # still return up to _CANDIDATES_LIMIT.
        # Fail-open: the save already persisted (lines above). A scan
        # error must not abort the caller, otherwise a retry would
        # create a duplicate node. Mirrors GraphEngine._auto_link
        # guard at graph.py.
        try:
            nodes = await self.graph.query(
                text=fts_query,
                namespace=namespace,
                limit=_CANDIDATES_LIMIT + 1,
            )
        except (OSError, RuntimeError, sqlite3.Error):
            logger.warning(
                "FTS5 candidate scan failed for content (len=%d); returning []",
                len(content),
                exc_info=True,
            )
            return []

        candidates: list[dict[str, Any]] = []
        for node in nodes:
            if exclude_node_id and node.id == exclude_node_id:
                continue
            description = node.description or ""
            snippet = description[:_CANDIDATE_SNIPPET_CHARS]
            if len(description) > _CANDIDATE_SNIPPET_CHARS:
                snippet += "..."
            candidates.append(
                {
                    "id": node.id,
                    "name": node.name,
                    "type": node.type,
                    "snippet": snippet,
                    "sim_rank": len(candidates),
                }
            )
            if len(candidates) >= _CANDIDATES_LIMIT:
                break
        return candidates

    def _node_result(self, *, node_id, name, type_, namespace, description, relevance):
        # namespace travels in every item shape so downstream namespace-based
        # access filters can enforce their allowlists on this surface.
        return {
            "source": "node",
            "id": node_id,
            "name": name,
            "type": type_,
            "namespace": namespace,
            "description": description,
            "relevance": relevance,
        }

    async def _keyword_node_recall(
        self, *, fts_query: str | None, limit: int, min_relevance: float
    ) -> list[dict[str, Any]]:
        """Keyword node path: FTS5 bm25 relevance, min_relevance gate. This is
        the default recall for nodes (semantic_recall OFF)."""
        if fts_query:
            ranked = await self.graph.query_ranked(text=fts_query, limit=limit)
        else:
            ranked = await self.graph.query_ranked(limit=limit)
        out: list[dict[str, Any]] = []
        for node, rank in ranked:
            relevance = _bm25_to_relevance(rank)
            if relevance < min_relevance:
                continue
            out.append(
                self._node_result(
                    node_id=node.id,
                    name=node.name,
                    type_=node.type,
                    namespace=node.namespace,
                    description=node.description,
                    relevance=relevance,
                )
            )
        return out

    async def _semantic_node_recall(
        self, topic: str, fts_query: str, limit: int
    ) -> list[dict[str, Any]]:
        """Semantic node path (semantic_recall ON): rerank the FTS5 top-N by
        local-embedding cosine and abstain below semantic_floor.

        Uses each candidate's stored vector when its model matches the live
        embedder; embeds any missing/stale candidate on the fly (correctness
        over speed - works before a backfill). Fail-open: any embedding error
        falls back to the keyword node path so recall never crashes and never
        silently returns nothing because of an embedder outage."""
        from kairn.core.embeddings import (
            cosine,
            node_embedding_text,
            normalize,
            unpack_vector,
        )

        candidates = await self.store.query_nodes_with_embeddings(
            text=fts_query, limit=self.semantic_top_n
        )
        if not candidates:
            return []
        loop = asyncio.get_running_loop()
        try:
            query_vectors = await loop.run_in_executor(None, self.embedder, [topic])
        except Exception:
            logger.warning(
                "semantic recall query-embed failed; falling back to keyword", exc_info=True
            )
            return await self._keyword_node_recall(
                fts_query=fts_query, limit=limit, min_relevance=0.0
            )
        if not query_vectors or not query_vectors[0]:
            return []
        qvec = normalize(query_vectors[0])

        scored: list[tuple[float, dict[str, Any]]] = []
        missing: list[dict[str, Any]] = []
        for row in candidates:
            blob = row.get("embedding")
            if blob and row.get("embedding_model") == self.embedder_model:
                scored.append((cosine(qvec, unpack_vector(blob)), row))
            else:
                missing.append(row)
        if missing:
            texts = [
                node_embedding_text(row.get("name"), row.get("description"))
                for row in missing
            ]
            try:
                fresh = await loop.run_in_executor(None, self.embedder, texts)
            except Exception:
                logger.warning("semantic recall candidate-embed failed", exc_info=True)
                fresh = []
            for row, vec in zip(missing, fresh, strict=False):
                if vec:
                    scored.append((cosine(qvec, normalize(vec)), row))

        scored.sort(key=lambda item: item[0], reverse=True)
        out: list[dict[str, Any]] = []
        for score, row in scored:
            if score < self.semantic_floor:
                continue
            out.append(
                self._node_result(
                    node_id=row["id"],
                    name=row["name"],
                    type_=row["type"],
                    namespace=row["namespace"],
                    description=row.get("description"),
                    relevance=round(score, 4),
                )
            )
            if len(out) >= limit:
                break
        return out

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

        # Node path. Default = keyword (honest bm25 relevance + min_relevance
        # gate). With the opt-in semantic_recall flag on AND a query present,
        # rerank the FTS5 top-N by local-embedding cosine and abstain below the
        # cosine floor. Flag OFF runs the keyword path unchanged.
        if self.semantic_recall and self.embedder is not None and fts_query and topic:
            node_results = await self._semantic_node_recall(topic, fts_query, limit)
        else:
            node_results = await self._keyword_node_recall(
                fts_query=fts_query, limit=limit, min_relevance=min_relevance
            )
        results.extend(node_results)
        kept_node_ids = [r["id"] for r in node_results]

        # Log node access for activity tracking (only nodes we surfaced).
        if kept_node_ids:
            await self._log_node_access("node_recall", kept_node_ids)

        # Search experiences (decay-aware)
        experiences = await self.experience.search(
            text=fts_query,
            min_relevance=min_relevance,
            limit=limit,
        )

        # Batch-increment access_count for all returned experiences so
        # the exp_auto_promote trigger can fire after repeated hits.
        # Mirror the increment on the in-memory objects so callers reading
        # exp.access_count from the result set are not off by one.
        if experiences:
            await self.experience.touch_accessed([e.id for e in experiences])
            for exp in experiences:
                exp.access_count += 1

        now = datetime.now(UTC)
        for exp in experiences:
            results.append(
                {
                    "source": "experience",
                    "id": exp.id,
                    "type": exp.type,
                    "namespace": exp.namespace,
                    "content": exp.content,
                    "confidence": exp.confidence,
                    "relevance": round(exp.relevance(at=now), 4),
                }
            )

        # Nodes (curated, permanent) lead, then experiences (decaying). Each
        # group is already ranked internally - nodes by bm25 match strength
        # (query_ranked order), experiences by the experience engine's search
        # order. We deliberately do NOT merge-sort the union by a single
        # "relevance" float: node bm25 match-strength and experience time-decay
        # are different scales, and sorting them together buries curated nodes
        # under fresh (high-decay-relevance) experiences.
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
                    "namespace": node.namespace,
                    "description": node.description,
                    "relevance": 1.0,
                }
            )

        # Log node access for activity tracking
        if nodes:
            await self._log_node_access("node_crossref", [n.id for n in nodes])

        # Search experiences for solutions
        experiences = await self.experience.search(
            text=fts_query,
            min_relevance=0.1,
            limit=limit,
        )

        # Batch-increment access_count for all returned experiences.
        if experiences:
            await self.experience.touch_accessed([e.id for e in experiences])
            for exp in experiences:
                exp.access_count += 1

        now = datetime.now(UTC)
        for exp in experiences:
            results.append(
                {
                    "source": "experience",
                    "workspace": "default",
                    "id": exp.id,
                    "type": exp.type,
                    "namespace": exp.namespace,
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
                "namespace": node_data.get("namespace"),
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
                    "namespace": n.namespace,
                    "confidence": 0.5,
                }
                if detail != "summary":
                    node_out["description"] = n.description
                    node_out["tags"] = n.tags
                    node_out["properties"] = n.properties
                nodes.append(node_out)

        # Log node access for activity tracking
        if nodes:
            await self._log_node_access(
                "node_context", [n["id"] for n in nodes]
            )

        # Experience search
        experiences = await self.experience.search(
            text=fts_query,
            min_relevance=0.1,
            limit=limit,
        )

        # Batch-increment access_count for all returned experiences.
        if experiences:
            await self.experience.touch_accessed([e.id for e in experiences])
            for exp in experiences:
                exp.access_count += 1

        now = datetime.now(UTC)
        exp_items = []
        for e in experiences:
            exp_out: dict[str, Any] = {
                "id": e.id,
                "type": e.type,
                "namespace": e.namespace,
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
