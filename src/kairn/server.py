"""FastMCP server — Gate 3: 18 tools, 3 resources, 2 prompts."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from kairn.core.experience import ExperienceEngine
from kairn.core.graph import GraphEngine
from kairn.core.ideas import IdeaEngine
from kairn.core.intelligence import IntelligenceLayer
from kairn.core.memory import ProjectMemory
from kairn.core.router import ContextRouter
from kairn.diagnostic import run_checks
from kairn.events.bus import EventBus
from kairn.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, default=str)


def create_server(db_path: str) -> FastMCP:
    """Create FastMCP server: 22 tools (5 graph + kn_judge + kn_doctor + 3 project + 4 exp + 2 ideas + 6 intel including kn_learn with candidates)."""
    state: dict[str, Any] = {}
    # _lock serializes lazy init/teardown ONLY - tool bodies run unserialized
    # by design (weakness-audit rank 22). Correctness under concurrency is the
    # storage layer's job (WAL + busy_timeout + single-statement atomic ops
    # such as merge_route_node_id), not a server-wide mutex, which would
    # serialize every read for no isolation gain.
    _lock = asyncio.Lock()

    @asynccontextmanager
    async def _lifespan(server: FastMCP) -> AsyncIterator[dict[str, Any]]:
        """Close the lazily-opened store when the server run ends.

        The aiosqlite connection owns a non-daemon worker thread; without
        this close, any process that ran a tool hangs at interpreter
        shutdown waiting on that thread. State is cleared so a later run
        on the same server object re-initializes cleanly; deliberately,
        that also resets the ``init_failed`` marker, so a failed init is
        retried on the next run instead of pinning the server object to
        the failure. The yielded dict is an unused placeholder required
        by the lifespan protocol - ``state`` is the real per-run state.

        Constraint: fastmcp runs this lifespan once per process for the
        stdio and HTTP transports, but per client connection for the
        in-memory transport, where teardown fires when the first
        connected session exits. Concurrent in-memory sessions on one
        server object are therefore not supported.
        """
        try:
            yield {}
        finally:
            async with _lock:
                store = state.get("store")
                state.clear()
                if store is not None:
                    await store.close()

    mcp = FastMCP("kairn", version="0.2.1", lifespan=_lifespan)

    async def _init() -> dict[str, Any]:
        async with _lock:
            if "init_failed" in state:
                raise RuntimeError(f"Kairn init previously failed for {db_path}")
            if "graph" not in state:
                from pathlib import Path

                from kairn.config import Config
                from kairn.core.embeddings import embedder_from_config

                config = Config.load(Path(db_path).parent)
                embedder, embedder_model = embedder_from_config(config)
                try:
                    store = SQLiteStore(
                        Path(db_path), embedder=embedder, embedder_model=embedder_model
                    )
                    await store.initialize()
                except Exception as e:
                    state["init_failed"] = True
                    logger.error("Failed to initialize database: %s", e)
                    raise RuntimeError(f"Kairn init failed: {db_path}") from e
                bus = EventBus()
                state["store"] = store
                state["bus"] = bus
                graph = GraphEngine(store, bus)
                router = ContextRouter(store, bus)
                memory = ProjectMemory(store, bus)
                experience = ExperienceEngine(store, bus)
                ideas = IdeaEngine(store, bus)
                state["graph"] = graph
                state["router"] = router
                state["memory"] = memory
                state["experience"] = experience
                state["ideas"] = ideas
                state["intel"] = IntelligenceLayer(
                    store=store,
                    event_bus=bus,
                    graph=graph,
                    router=router,
                    memory=memory,
                    experience=experience,
                    ideas=ideas,
                    embedder=embedder,
                    embedder_model=embedder_model,
                    semantic_recall=config.semantic_recall,
                    semantic_floor=config.semantic_recall_floor,
                    semantic_top_n=config.semantic_recall_top_n,
                )
        return state

    @mcp.tool()
    async def kn_add(
        name: Annotated[str, Field(description="Node name")],
        type: Annotated[
            str,
            Field(
                description="Node type (concept, pattern, etc.)",
            ),
        ],
        namespace: Annotated[
            str,
            Field(
                description="Namespace",
            ),
        ] = "knowledge",
        description: Annotated[
            str | None,
            Field(
                description="Node description",
            ),
        ] = None,
        tags: Annotated[
            list[str] | None,
            Field(
                description="Tags for categorization",
            ),
        ] = None,
    ) -> str:
        """Add node to knowledge graph. Auto-links via FTS5."""
        if not name or not name.strip():
            return _json({"_v": "1.0", "error": "name is required"})
        if not type or not type.strip():
            return _json({"_v": "1.0", "error": "type is required"})

        s = await _init()
        node = await s["graph"].add_node(
            name=name.strip(),
            type=type.strip(),
            namespace=namespace,
            description=description,
            tags=tags,
        )
        await s["router"].update_routes_for_node(
            node.id,
            node.name,
            node.description,
        )
        result = node.to_response(detail="full")
        result["_v"] = "1.0"
        return _json(result)

    @mcp.tool()
    async def kn_connect(
        source_id: Annotated[str, Field(description="Source node ID")],
        target_id: Annotated[str, Field(description="Target node ID")],
        edge_type: Annotated[str, Field(description="Relationship type")],
        weight: Annotated[
            float,
            Field(
                description="Edge weight 0.0-1.0",
                ge=0.0,
                le=1.0,
            ),
        ] = 1.0,
        created_by: Annotated[
            str | None,
            Field(description="Who/what created this edge (for provenance)"),
        ] = "kn_connect",
    ) -> str:
        """Create typed, weighted edge between nodes."""
        if not source_id or not source_id.strip():
            return _json({"_v": "1.0", "error": "source_id is required"})
        if not target_id or not target_id.strip():
            return _json({"_v": "1.0", "error": "target_id is required"})
        if not edge_type or not edge_type.strip():
            return _json({"_v": "1.0", "error": "edge_type is required"})

        s = await _init()
        try:
            edge = await s["graph"].connect(
                source_id,
                target_id,
                edge_type,
                weight=weight,
                created_by=created_by,
            )
            result = edge.to_storage()
            result["_v"] = "1.0"
            return _json(result)
        except ValueError as e:
            return _json({"_v": "1.0", "error": str(e)})

    @mcp.tool()
    async def kn_judge(
        source_id: Annotated[str, Field(description="Source node ID")],
        target_id: Annotated[str, Field(description="Target node ID")],
        relation: Annotated[
            str,
            Field(
                description="One of the 5 canonical verbs: "
                "conflicts_with, supersedes, compatible, scoped, related",
            ),
        ],
        reason: Annotated[
            str | None,
            Field(
                description="Optional rationale stored in edge.properties.reason",
            ),
        ] = None,
        confidence: Annotated[
            float,
            Field(
                description="Confidence 0.0-1.0; persisted as edge.weight",
                ge=0.0,
                le=1.0,
            ),
        ] = 1.0,
    ) -> str:
        """Record a typed relationship judgment between two nodes.

        Strict-mode wrapper around kn_connect: only the 5 canonical
        relation verbs (conflicts_with, supersedes, compatible, scoped,
        related) are accepted. Use after kn_learn returns a non-empty
        candidates[] list to assert how the new node relates to an
        existing one. For legacy or system-generated edges, use
        kn_connect (lax mode).
        """
        if not source_id or not source_id.strip():
            return _json({"_v": "1.0", "error": "source_id is required"})
        if not target_id or not target_id.strip():
            return _json({"_v": "1.0", "error": "target_id is required"})
        if not relation or not relation.strip():
            return _json({"_v": "1.0", "error": "relation is required"})

        # Normalize to lowercase to match CLI behavior (click.Choice
        # uses case_sensitive=False). Without this, kn_judge(relation="Compatible")
        # would error while `kairn judge --relation Compatible` would succeed.
        relation = relation.strip().lower()

        s = await _init()
        properties = {"reason": reason} if reason else None
        try:
            edge = await s["graph"].connect(
                source_id,
                target_id,
                relation,
                weight=confidence,
                properties=properties,
                created_by="kn_judge",
                strict_relation=True,
            )
            result = edge.to_storage()
            result["_v"] = "1.0"
            return _json(result)
        except ValueError as e:
            return _json({"_v": "1.0", "error": str(e)})
        except sqlite3.IntegrityError:
            # Composite PK violation: (source_id, target_id, relation)
            # already exists. Surface as error envelope rather than tool
            # exception. Use a fixed-shape message (no raw SQLite text)
            # to avoid leaking column names; the caller has the triple
            # it submitted and can disconnect-then-rejudge if needed.
            return _json(
                {
                    "_v": "1.0",
                    "error": (
                        f"duplicate judgment edge: "
                        f"({source_id}, {target_id}, {relation}) already exists"
                    ),
                }
            )

    @mcp.tool()
    async def kn_query(
        text: Annotated[
            str | None,
            Field(
                description="Full-text search query",
            ),
        ] = None,
        namespace: Annotated[
            str | None,
            Field(
                description="Filter by namespace",
            ),
        ] = None,
        node_type: Annotated[
            str | None,
            Field(
                description="Filter by type",
            ),
        ] = None,
        tags: Annotated[
            list[str] | None,
            Field(
                description="Filter by tags",
            ),
        ] = None,
        detail: Annotated[
            str,
            Field(
                description="summary or full",
            ),
        ] = "summary",
        limit: Annotated[
            int,
            Field(
                description="Max results",
                ge=1,
                le=50,
            ),
        ] = 10,
        offset: Annotated[
            int,
            Field(
                description="Pagination offset",
                ge=0,
            ),
        ] = 0,
    ) -> str:
        """Search nodes by text, type, tags, or namespace."""
        s = await _init()
        nodes = await s["graph"].query(
            text=text,
            namespace=namespace,
            node_type=node_type,
            tags=tags,
            limit=limit,
            offset=offset,
        )
        # namespace travels on every node/experience shape so downstream
        # namespace-based access filters can enforce uniformly - to_response's
        # summary projection omits it, so inject it (matches kn_related /
        # intelligence recall/context/crossref).
        items = []
        for n in nodes:
            item = n.to_response(detail=detail)
            item.setdefault("namespace", n.namespace)
            items.append(item)
        return _json(
            {
                "_v": "1.0",
                "count": len(items),
                "nodes": items,
            }
        )

    @mcp.tool()
    async def kn_remove(
        node_id: Annotated[
            str | None,
            Field(
                description="Node ID to remove",
            ),
        ] = None,
        source_id: Annotated[
            str | None,
            Field(
                description="Edge source ID",
            ),
        ] = None,
        target_id: Annotated[
            str | None,
            Field(
                description="Edge target ID",
            ),
        ] = None,
        edge_type: Annotated[
            str | None,
            Field(
                description="Edge type",
            ),
        ] = None,
    ) -> str:
        """Soft-delete node or edge. Supports undo."""
        s = await _init()

        if node_id and node_id.strip():
            ok = await s["graph"].remove_node(node_id)
            if ok:
                return _json(
                    {
                        "_v": "1.0",
                        "removed": "node",
                        "id": node_id,
                    }
                )
            return _json(
                {
                    "_v": "1.0",
                    "error": f"Node not found: {node_id}",
                }
            )

        if source_id and target_id and edge_type:
            ok = await s["graph"].disconnect(
                source_id,
                target_id,
                edge_type,
            )
            if ok:
                return _json(
                    {
                        "_v": "1.0",
                        "removed": "edge",
                        "source_id": source_id,
                        "target_id": target_id,
                    }
                )
            return _json({"_v": "1.0", "error": "Edge not found"})

        return _json(
            {
                "_v": "1.0",
                "error": "Provide node_id or (source_id + target_id + edge_type)",
            }
        )

    @mcp.tool()
    async def kn_status() -> str:
        """Graph stats, health, and system overview."""
        s = await _init()
        stats = await s["graph"].stats()
        stats["_v"] = "1.0"
        return _json(stats)

    # ── Project Memory tools (3) ───────────────────────────────

    @mcp.tool()
    async def kn_project(
        name: Annotated[str, Field(description="Project name")],
        project_id: Annotated[
            str | None,
            Field(description="Project ID (omit to create new)"),
        ] = None,
        phase: Annotated[
            str | None,
            Field(description="Phase: planning|active|paused|done"),
        ] = None,
        goals: Annotated[
            list[str] | None,
            Field(description="Project goals"),
        ] = None,
        stakeholders: Annotated[
            list[str] | None,
            Field(description="Stakeholders"),
        ] = None,
        success_metrics: Annotated[
            list[str] | None,
            Field(description="Success metrics / KPIs"),
        ] = None,
    ) -> str:
        """Create or update project (upsert)."""
        if not name or not name.strip():
            return _json({"_v": "1.0", "error": "name is required"})

        s = await _init()
        mem = s["memory"]

        if project_id:
            # Update existing
            updates: dict[str, Any] = {"name": name.strip()}
            if phase is not None:
                updates["phase"] = phase
            if goals is not None:
                updates["goals"] = goals
            if stakeholders is not None:
                updates["stakeholders"] = stakeholders
            if success_metrics is not None:
                updates["success_metrics"] = success_metrics
            try:
                project = await mem.update_project(project_id, **updates)
            except ValueError as e:
                return _json({"_v": "1.0", "error": str(e)})
            if not project:
                return _json({"_v": "1.0", "error": f"Project not found: {project_id}"})
        else:
            # Create new — phase always starts at "planning"
            if phase is not None:
                return _json(
                    {
                        "_v": "1.0",
                        "error": "phase cannot be set on create (starts at planning)",
                    }
                )
            try:
                project = await mem.create_project(
                    name=name.strip(),
                    goals=goals,
                    stakeholders=stakeholders,
                    success_metrics=success_metrics,
                )
            except ValueError as e:
                return _json({"_v": "1.0", "error": str(e)})

        result = {
            "_v": "1.0",
            "id": project.id,
            "name": project.name,
            "phase": project.phase,
            "active": project.active,
            "goals": project.goals,
        }
        return _json(result)

    @mcp.tool()
    async def kn_projects(
        active_only: Annotated[
            bool,
            Field(description="Only show active projects"),
        ] = False,
        set_active: Annotated[
            str | None,
            Field(description="Project ID to set as active"),
        ] = None,
    ) -> str:
        """List projects and switch active."""
        s = await _init()
        mem = s["memory"]

        if set_active:
            ok = await mem.set_active_project(set_active)
            if not ok:
                return _json({"_v": "1.0", "error": f"Project not found: {set_active}"})

        projects = await mem.list_projects(active_only=active_only)
        items = [
            {
                "id": p.id,
                "name": p.name,
                "phase": p.phase,
                "active": p.active,
            }
            for p in projects
        ]
        return _json({"_v": "1.0", "count": len(items), "projects": items})

    @mcp.tool()
    async def kn_log(
        project_id: Annotated[str, Field(description="Project ID")],
        action: Annotated[str, Field(description="What was done or what failed")],
        type: Annotated[
            str,
            Field(description="progress or failure"),
        ] = "progress",
        result: Annotated[
            str | None,
            Field(description="Outcome or error message"),
        ] = None,
        next_step: Annotated[
            str | None,
            Field(description="Recommended next step"),
        ] = None,
    ) -> str:
        """Log progress or failure entry."""
        if not project_id or not project_id.strip():
            return _json({"_v": "1.0", "error": "project_id is required"})
        if not action or not action.strip():
            return _json({"_v": "1.0", "error": "action is required"})

        s = await _init()
        mem = s["memory"]

        if type == "failure":
            entry = await mem.log_failure(
                project_id=project_id,
                action=action,
                result=result,
                next_step=next_step,
            )
        else:
            entry = await mem.log_progress(
                project_id=project_id,
                action=action,
                result=result,
                next_step=next_step,
            )

        return _json(
            {
                "_v": "1.0",
                "id": entry.id,
                "project_id": entry.project_id,
                "type": entry.type,
                "action": entry.action,
            }
        )

    # ── Experience Memory tools (5) ──────────────────────────

    @mcp.tool()
    async def kn_save(
        content: Annotated[str, Field(description="What was learned/discovered")],
        type: Annotated[
            str,
            Field(description="solution|pattern|decision|workaround|gotcha|preference"),
        ],
        context: Annotated[
            str | None,
            Field(description="Situation when this was learned"),
        ] = None,
        confidence: Annotated[
            str,
            Field(description="high|medium|low — affects decay rate"),
        ] = "high",
        tags: Annotated[
            list[str] | None,
            Field(description="Tags for categorization"),
        ] = None,
    ) -> str:
        """Save experience with configurable decay."""
        if not content or not content.strip():
            return _json({"_v": "1.0", "error": "content is required"})

        s = await _init()
        try:
            exp = await s["experience"].save(
                content=content.strip(),
                type=type,
                context=context,
                confidence=confidence,
                tags=tags,
            )
        except ValueError as e:
            return _json({"_v": "1.0", "error": str(e)})

        return _json(
            {
                "_v": "1.0",
                "id": exp.id,
                "type": exp.type,
                "confidence": exp.confidence,
                "decay_rate": round(exp.decay_rate, 6),
                "score": exp.score,
            }
        )

    @mcp.tool()
    async def kn_preference(
        content: Annotated[
            str,
            Field(
                description="The user preference as ONE explicit, standalone "
                "sentence (e.g. 'User prefers Adobe Premiere Pro tutorials "
                "focused on advanced settings'). You, the calling model, are "
                "the extractor: state the preference explicitly rather than "
                "pasting the raw conversation."
            ),
        ],
        context: Annotated[
            str | None,
            Field(description="Situation in which the preference was stated"),
        ] = None,
        tags: Annotated[
            list[str] | None,
            Field(description="Tags for categorization (subject of the preference)"),
        ] = None,
        namespace: Annotated[
            str,
            Field(
                description="Namespace for multi-tenant isolation "
                "(default 'knowledge')",
            ),
        ] = "knowledge",
    ) -> str:
        """Capture a stated user preference at utterance time.

        The dedicated write path for preferences: stores a `preference`-type
        experience with the longest half-life of any type (durable by
        default). Call it the moment a preference is expressed - the host
        model hearing the preference is the extraction step; Kairn's server
        stays deterministic and never runs an LLM.
        """
        if not content or not content.strip():
            return _json({"_v": "1.0", "error": "content is required"})

        s = await _init()
        try:
            exp = await s["experience"].save(
                content=content.strip(),
                type="preference",
                context=context,
                confidence="high",
                tags=tags,
                namespace=namespace,
            )
        except ValueError as e:
            return _json({"_v": "1.0", "error": str(e)})

        return _json(
            {
                "_v": "1.0",
                "id": exp.id,
                "type": exp.type,
                "confidence": exp.confidence,
                "decay_rate": round(exp.decay_rate, 6),
                "score": exp.score,
            }
        )

    @mcp.tool()
    async def kn_memories(
        text: Annotated[
            str | None,
            Field(description="Full-text search query"),
        ] = None,
        type: Annotated[
            str | None,
            Field(description="Filter by experience type"),
        ] = None,
        min_relevance: Annotated[
            float,
            Field(
                description="Minimum relevance 0.0-1.0",
                ge=0.0,
                le=1.0,
            ),
        ] = 0.0,
        limit: Annotated[
            int,
            Field(description="Max results", ge=1, le=50),
        ] = 10,
        offset: Annotated[
            int,
            Field(description="Pagination offset", ge=0),
        ] = 0,
    ) -> str:
        """Decay-aware experience search."""
        s = await _init()
        experiences = await s["experience"].search(
            text=text,
            exp_type=type,
            min_relevance=min_relevance,
            limit=limit,
            offset=offset,
        )
        items = [
            {
                "id": e.id,
                "type": e.type,
                # namespace on every experience shape too (uniform invariant
                # for downstream namespace-based access filters).
                "namespace": e.namespace,
                "content": e.content,
                "confidence": e.confidence,
                "relevance": round(e.relevance(), 4),
                "tags": e.tags,
            }
            for e in experiences
        ]
        return _json({"_v": "1.0", "count": len(items), "experiences": items})

    @mcp.tool()
    async def kn_prune(
        threshold: Annotated[
            float,
            Field(
                description="Remove experiences below this relevance",
                ge=0.0,
                le=1.0,
            ),
        ] = 0.01,
    ) -> str:
        """Remove expired experiences (archive first)."""
        s = await _init()
        pruned = await s["experience"].prune(threshold=threshold)
        return _json(
            {
                "_v": "1.0",
                "pruned_count": len(pruned),
                "pruned_ids": pruned,
            }
        )

    @mcp.tool()
    async def kn_promote_pending(
        limit: Annotated[
            int,
            Field(
                description=(
                    "Maximum experiences to promote in this call. "
                    "Bounds a single sweep so it cannot block on a "
                    "large promotion backlog."
                ),
                ge=1,
                le=10000,
            ),
        ] = 100,
    ) -> str:
        """Promote experiences flagged by the auto-promote SQL trigger.

        The exp_auto_promote trigger sets needs_promotion=1 when an
        experience's access_count crosses 5 (via the read-path
        touch_accessed wiring from Phase 8). This tool consumes those
        flags and turns each flagged experience into a permanent graph
        node via the existing _promote() path.

        Idempotent: repeatedly safe to call; no-op when nothing is
        flagged. Returns stats including the list of newly created
        node ids.
        """
        s = await _init()
        stats = await s["experience"].promote_pending(limit=limit)
        return _json({"_v": "1.0", **stats})

    # ── Idea tools (2) ───────────────────────────────────────

    @mcp.tool()
    async def kn_idea(
        title: Annotated[str, Field(description="Idea title")],
        idea_id: Annotated[
            str | None,
            Field(description="Idea ID (omit to create new)"),
        ] = None,
        category: Annotated[
            str | None,
            Field(description="Category classification"),
        ] = None,
        score: Annotated[
            float | None,
            Field(description="Numerical score"),
        ] = None,
        status: Annotated[
            str | None,
            Field(description="Status: draft|evaluating|approved|implementing|done|archived"),
        ] = None,
        link_to: Annotated[
            str | None,
            Field(description="Node ID to link this idea to"),
        ] = None,
    ) -> str:
        """Create or update idea with graph links."""
        if not title or not title.strip():
            return _json({"_v": "1.0", "error": "title is required"})

        s = await _init()
        ideas_engine = s["ideas"]

        if idea_id:
            # Update existing
            updates: dict[str, Any] = {"title": title.strip()}
            if category is not None:
                updates["category"] = category
            if score is not None:
                updates["score"] = score
            if status is not None:
                updates["status"] = status
            try:
                idea = await ideas_engine.update(idea_id, **updates)
            except ValueError as e:
                return _json({"_v": "1.0", "error": str(e)})
            if not idea:
                return _json({"_v": "1.0", "error": f"Idea not found: {idea_id}"})
        else:
            # Create new
            try:
                idea = await ideas_engine.create(
                    title=title.strip(),
                    category=category,
                    score=score,
                )
            except ValueError as e:
                return _json({"_v": "1.0", "error": str(e)})

        result: dict[str, Any] = {
            "_v": "1.0",
            "id": idea.id,
            "title": idea.title,
            "status": idea.status,
            "category": idea.category,
            "score": idea.score,
        }

        # Optional graph link
        if link_to:
            edge = await ideas_engine.link_to_node(idea.id, link_to)
            result["linked_to"] = link_to if edge else None
            if not edge:
                result["link_error"] = f"Node not found: {link_to}"

        return _json(result)

    @mcp.tool()
    async def kn_ideas(
        status: Annotated[
            str | None,
            Field(description="Filter by status"),
        ] = None,
        category: Annotated[
            str | None,
            Field(description="Filter by category"),
        ] = None,
        limit: Annotated[
            int,
            Field(description="Max results", ge=1, le=50),
        ] = 10,
        offset: Annotated[
            int,
            Field(description="Pagination offset", ge=0),
        ] = 0,
    ) -> str:
        """List and filter ideas by status/score."""
        s = await _init()
        ideas_list = await s["ideas"].list_ideas(
            status=status,
            category=category,
            limit=limit,
            offset=offset,
        )
        items = [
            {
                "id": i.id,
                "title": i.title,
                "status": i.status,
                "category": i.category,
                "score": i.score,
            }
            for i in ideas_list
        ]
        return _json({"_v": "1.0", "count": len(items), "ideas": items})

    # ── Intelligence tools (5) ────────────────────────────────

    @mcp.tool()
    async def kn_learn(
        content: Annotated[str, Field(description="What was learned/decided/discovered")],
        type: Annotated[
            str,
            Field(description="decision|pattern|solution|workaround|gotcha|preference"),
        ],
        context: Annotated[
            str | None,
            Field(description="Situation when this was learned"),
        ] = None,
        confidence: Annotated[
            str,
            Field(description="high=permanent, medium=likely, low=exploratory"),
        ] = "high",
        tags: Annotated[
            list[str] | None,
            Field(description="Tags for categorization"),
        ] = None,
        namespace: Annotated[
            str,
            Field(
                description="Namespace for multi-tenant isolation "
                "(default 'knowledge')",
            ),
        ] = "knowledge",
        with_candidates: Annotated[
            bool,
            Field(
                description="Return up to 5 FTS5-matched related nodes "
                "in candidates[] so the caller can decide whether to "
                "invoke kn_judge for a relationship verb. Set False "
                "for bulk-save scripts that do not need the hook.",
            ),
        ] = True,
    ) -> str:
        """Store knowledge from conversation. Creates node (high) or experience (medium/low)."""
        if not content or not content.strip():
            return _json({"_v": "1.0", "error": "content is required"})

        s = await _init()
        try:
            result = await s["intel"].learn(
                content=content.strip(),
                type=type,
                context=context,
                confidence=confidence,
                tags=tags,
                namespace=namespace,
                with_candidates=with_candidates,
            )
        except ValueError as e:
            return _json({"_v": "1.0", "error": str(e)})

        return _json(result)

    @mcp.tool()
    async def kn_recall(
        topic: Annotated[
            str | None,
            Field(description="Topic to recall knowledge about"),
        ] = None,
        limit: Annotated[
            int,
            Field(description="Max results", ge=1, le=50),
        ] = 10,
        min_relevance: Annotated[
            float,
            Field(
                description="Minimum relevance 0.0-1.0",
                ge=0.0,
                le=1.0,
            ),
        ] = 0.0,
    ) -> str:
        """Surface relevant past knowledge for context."""
        s = await _init()
        results = await s["intel"].recall(
            topic=topic,
            limit=limit,
            min_relevance=min_relevance,
        )
        return _json(
            {
                "_v": "1.0",
                "count": len(results),
                "results": results,
            }
        )

    @mcp.tool()
    async def kn_crossref(
        problem: Annotated[str, Field(description="Problem description to find solutions for")],
        limit: Annotated[
            int,
            Field(description="Max results", ge=1, le=50),
        ] = 10,
    ) -> str:
        """Find similar solutions from other workspaces."""
        if not problem or not problem.strip():
            return _json({"_v": "1.0", "error": "problem is required"})

        s = await _init()
        try:
            results = await s["intel"].crossref(
                problem=problem.strip(),
                limit=limit,
            )
        except ValueError as e:
            return _json({"_v": "1.0", "error": str(e)})

        return _json(
            {
                "_v": "1.0",
                "count": len(results),
                "results": results,
            }
        )

    @mcp.tool()
    async def kn_context(
        keywords: Annotated[str, Field(description="Keywords to find relevant context for")],
        detail: Annotated[
            str,
            Field(description="summary or full"),
        ] = "summary",
        limit: Annotated[
            int,
            Field(description="Max results", ge=1, le=50),
        ] = 10,
    ) -> str:
        """Keywords to relevant subgraph with progressive disclosure."""
        s = await _init()
        result = await s["intel"].context(
            keywords=keywords,
            detail=detail,
            limit=limit,
        )
        return _json(result)

    @mcp.tool()
    async def kn_related(
        node_id: Annotated[str, Field(description="Starting node ID")],
        depth: Annotated[
            int,
            Field(description="Traversal depth", ge=1, le=5),
        ] = 1,
        edge_type: Annotated[
            str | None,
            Field(description="Filter by edge type"),
        ] = None,
    ) -> str:
        """Find nodes connected to a starting point."""
        if not node_id or not node_id.strip():
            return _json({"_v": "1.0", "error": "node_id is required"})

        s = await _init()
        results = await s["intel"].related(
            node_id=node_id.strip(),
            depth=depth,
            edge_type=edge_type,
        )
        return _json(
            {
                "_v": "1.0",
                "count": len(results),
                "results": results,
            }
        )

    @mcp.tool()
    async def kn_doctor(
        check: Annotated[
            str | None,
            Field(
                description="Run a single check by ID; omit to run all. "
                "Known: check_lock_mode, check_fts_index_health, "
                "check_promoted_experience_consistency, "
                "check_namespace_distribution, check_orphan_edges",
            ),
        ] = None,
    ) -> str:
        """Read-only Kairn health diagnostics.

        Returns a structured envelope with per-check verdicts and a
        roll-up summary. Identical JSON shape to `kairn doctor --json`
        (envelope parity). Safe to call any time; checks never mutate
        workspace state.
        """
        s = await _init()
        try:
            report = await run_checks(s["store"], only=check)
            return _json(report)
        except ValueError as e:
            return _json({"_v": "1.0", "error": str(e)})

    # ── Resources (3) ──────────────────────────────────────────

    @mcp.resource("kn://status")
    async def kn_resource_status() -> str:
        """Graph and system overview."""
        s = await _init()
        stats = await s["graph"].stats()
        projects = await s["memory"].list_projects(active_only=True)
        active = projects[0] if projects else None
        return _json(
            {
                "_v": "1.0",
                "graph": stats,
                "active_project": {
                    "id": active.id,
                    "name": active.name,
                    "phase": active.phase,
                }
                if active
                else None,
            }
        )

    @mcp.resource("kn://projects")
    async def kn_resource_projects() -> str:
        """All projects with progress summaries."""
        s = await _init()
        mem = s["memory"]
        projects = await mem.list_projects()
        items = []
        for p in projects:
            progress = await mem.get_progress(p.id, limit=3)
            items.append(
                {
                    "id": p.id,
                    "name": p.name,
                    "phase": p.phase,
                    "active": p.active,
                    "goals": p.goals,
                    "recent_progress": [{"action": e.action, "type": e.type} for e in progress],
                }
            )
        return _json({"_v": "1.0", "count": len(items), "projects": items})

    @mcp.resource("kn://memories")
    async def kn_resource_memories() -> str:
        """Recent high-relevance experiences."""
        s = await _init()
        experiences = await s["experience"].search(
            min_relevance=0.1,
            limit=20,
        )
        items = [
            {
                "id": e.id,
                "type": e.type,
                "content": e.content[:200],
                "confidence": e.confidence,
                "relevance": round(e.relevance(), 4),
            }
            for e in experiences
        ]
        return _json({"_v": "1.0", "count": len(items), "experiences": items})

    # ── Prompts (2) ──────────────────────────────────────────

    @mcp.prompt()
    async def kn_bootup() -> str:
        """Session start — load active project, recent progress, and top memories."""
        s = await _init()
        mem = s["memory"]

        # Active project
        projects = await mem.list_projects(active_only=True)
        active = projects[0] if projects else None

        # Recent progress
        progress_lines = []
        if active:
            entries = await mem.get_progress(active.id, limit=5)
            for e in entries:
                prefix = "FAIL" if e.type == "failure" else "OK"
                progress_lines.append(f"  [{prefix}] {e.action}")

        # Top experiences
        experiences = await s["experience"].search(min_relevance=0.3, limit=5)
        memory_lines = [f"  [{e.type}] {e.content[:80]}" for e in experiences]

        # Build context
        parts = ["# Kairn Session Context\n"]

        if active:
            parts.append(f"## Active Project: {active.name}")
            parts.append(f"Phase: {active.phase}")
            if active.goals:
                parts.append("Goals: " + ", ".join(active.goals))
            if progress_lines:
                parts.append("\nRecent progress:")
                parts.extend(progress_lines)
        else:
            parts.append("No active project. Use kn_project to create one.")

        if memory_lines:
            parts.append("\n## Key Memories")
            parts.extend(memory_lines)

        # Ideas in progress
        ideas = await s["ideas"].list_ideas(status="implementing", limit=3)
        if ideas:
            parts.append("\n## Ideas in Progress")
            for idea in ideas:
                parts.append(f"  - {idea.title} ({idea.status})")

        return "\n".join(parts)

    @mcp.prompt()
    async def kn_review() -> str:
        """Session review — summarize what happened and suggest next steps."""
        s = await _init()
        mem = s["memory"]

        projects = await mem.list_projects(active_only=True)
        active = projects[0] if projects else None

        parts = ["# Session Review\n"]

        if active:
            parts.append(f"## Project: {active.name} ({active.phase})")

            # All progress this session (recent entries)
            progress = await mem.get_progress(active.id, limit=10)
            successes = [e for e in progress if e.type == "progress"]
            failures = [e for e in progress if e.type == "failure"]

            if successes:
                parts.append(f"\n### Completed ({len(successes)})")
                for e in successes:
                    parts.append(f"  - {e.action}")
                    if e.result:
                        parts.append(f"    Result: {e.result}")

            if failures:
                parts.append(f"\n### Issues ({len(failures)})")
                for e in failures:
                    parts.append(f"  - {e.action}")
                    if e.result:
                        parts.append(f"    Error: {e.result}")
                    if e.next_step:
                        parts.append(f"    Next: {e.next_step}")

            # Suggest next step from most recent entry
            if progress and progress[0].next_step:
                parts.append("\n## Suggested Next Step")
                parts.append(f"{progress[0].next_step}")
        else:
            parts.append("No active project to review.")

        # Experience stats
        all_exp = await s["experience"].search(limit=50)
        if all_exp:
            by_type: dict[str, int] = {}
            for e in all_exp:
                by_type[e.type] = by_type.get(e.type, 0) + 1
            parts.append("\n## Memory Stats")
            for t, count in sorted(by_type.items()):
                parts.append(f"  {t}: {count}")

        return "\n".join(parts)

    return mcp
