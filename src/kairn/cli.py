"""CLI — init, serve, status, workspace, demo, benchmark, token-audit."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from kairn.config import Config
from kairn.storage.metadata_store import MetadataStore
from kairn.storage.sqlite_store import SQLiteStore


@click.group()
@click.version_option(package_name="kairn-ai")
def main() -> None:
    """Kairn — your AI's persistent memory."""


@main.command()
@click.argument("path", type=click.Path(), default="~/.kairn")
def init(path: str) -> None:
    """Initialize a new kairn workspace."""
    workspace = Path(path).expanduser().resolve()

    async def _init() -> None:
        config = Config(workspace_path=workspace)
        store = SQLiteStore(workspace / "kairn.db")
        await store.initialize()
        await store.close()
        config.save()

    asyncio.run(_init())
    click.echo(f"Initialized workspace at {workspace}")
    click.echo(f"Database: {workspace / 'kairn.db'}")
    click.echo("Add to Claude Desktop config:")
    click.echo(f'  "kairn": {{"command": "kairn", "args": ["serve", "{workspace}"]}}')


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--transport", type=click.Choice(["stdio"]), default="stdio")
def serve(path: str, transport: str) -> None:
    """Start the MCP server."""
    workspace = Path(path).expanduser().resolve()
    db_path = workspace / "kairn.db"

    if not db_path.exists():
        click.echo(f"Error: No database at {db_path}. Run 'kairn init' first.", err=True)
        sys.exit(1)

    from kairn.server import create_server

    server = create_server(str(db_path))
    server.run(transport=transport)  # type: ignore[arg-type]


@main.command()
@click.argument("path", type=click.Path(exists=True))
def status(path: str) -> None:
    """Show workspace status."""
    workspace = Path(path).expanduser().resolve()
    db_path = workspace / "kairn.db"

    if not db_path.exists():
        click.echo(f"Error: No database at {db_path}", err=True)
        sys.exit(1)

    async def _status() -> dict:
        store = SQLiteStore(db_path)
        try:
            await store.initialize()
            return await store.get_stats()
        finally:
            await store.close()

    stats = asyncio.run(_status())
    click.echo(json.dumps(stats, indent=2))


@main.group()
def workspace() -> None:
    """Manage workspaces."""


@workspace.command()
@click.argument("name")
@click.option("--org", default="default", help="Organization ID")
@click.option("--description", default=None, help="Workspace description")
@click.option("--type", "workspace_type", default="project", help="Workspace type")
def create(name: str, org: str, description: str | None, workspace_type: str) -> None:
    """Create a new workspace."""
    config = Config.load()

    async def _create() -> None:
        store = MetadataStore(config.metadata_db_path)
        await store.initialize()

        try:
            workspace_id = str(uuid.uuid4())
            user_id = "cli-user"
            org_id = org

            existing_user = await store.get_user(user_id)
            if not existing_user:
                await store.create_user(user_id, "cli@local", "CLI User")

            existing_org = await store.get_org(org_id)
            if not existing_org:
                await store.create_org(org_id, org_id.capitalize(), user_id)

            await store.create_workspace(
                workspace_id=workspace_id,
                org_id=org_id,
                name=name,
                created_by=user_id,
                description=description,
                workspace_type=workspace_type,
            )

            await store.add_member(workspace_id, user_id, role="owner")

            workspace_db = config.workspace_db_path(workspace_id)
            workspace_db.parent.mkdir(parents=True, exist_ok=True)
            ws_store = SQLiteStore(workspace_db)
            await ws_store.initialize()
            await ws_store.close()

            console = Console()
            console.print(
                Panel(
                    f"[green]✓[/green] Workspace created: {name}\n"
                    f"ID: {workspace_id}\n"
                    f"Organization: {org_id}\n"
                    f"Database: {workspace_db}",
                    title="Workspace Created",
                )
            )

        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        finally:
            await store.close()

    asyncio.run(_create())


@workspace.command()
@click.argument("workspace_id")
@click.option("--token", required=True, help="JWT token for authentication")
def join(workspace_id: str, token: str) -> None:
    """Join an existing workspace."""
    config = Config.load()
    secret = os.environ.get("KAIRN_JWT_SECRET", "test-secret-key-do-not-use")

    async def _join() -> None:
        from kairn.auth.jwt import TokenExpiredError, TokenInvalidError, verify_token

        store = MetadataStore(config.metadata_db_path)
        await store.initialize()

        try:
            try:
                payload = verify_token(token, secret)
                user_id = payload.get("sub")
                if not user_id:
                    raise TokenInvalidError("Token missing user ID")
            except (TokenExpiredError, TokenInvalidError) as e:
                click.echo(f"Authentication failed: {e}", err=True)
                sys.exit(1)

            workspace_data = await store.get_workspace(workspace_id)
            if not workspace_data:
                click.echo(f"Error: Workspace {workspace_id} not found", err=True)
                sys.exit(1)

            existing_role = await store.get_member_role(workspace_id, user_id)
            if existing_role:
                click.echo(f"You are already a member of this workspace (role: {existing_role})")
                return

            await store.add_member(workspace_id, user_id, role="contributor")

            console = Console()
            console.print(
                Panel(
                    f"[green]✓[/green] Joined workspace: {workspace_data['name']}\n"
                    f"ID: {workspace_id}\n"
                    f"Role: contributor",
                    title="Workspace Joined",
                )
            )

        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        finally:
            await store.close()

    asyncio.run(_join())


@workspace.command()
@click.argument("workspace_id")
def leave(workspace_id: str) -> None:
    """Leave a workspace."""
    config = Config.load()

    async def _leave() -> None:
        store = MetadataStore(config.metadata_db_path)
        await store.initialize()

        try:
            user_id = "cli-user"

            workspace_data = await store.get_workspace(workspace_id)
            if not workspace_data:
                click.echo(f"Error: Workspace {workspace_id} not found", err=True)
                sys.exit(1)

            existing_role = await store.get_member_role(workspace_id, user_id)
            if not existing_role:
                click.echo("You are not a member of this workspace")
                return

            removed = await store.remove_member(workspace_id, user_id)
            if removed:
                console = Console()
                console.print(
                    Panel(
                        f"[green]✓[/green] Left workspace: {workspace_data['name']}\n"
                        f"ID: {workspace_id}",
                        title="Workspace Left",
                    )
                )
            else:
                click.echo("Failed to leave workspace", err=True)
                sys.exit(1)

        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        finally:
            await store.close()

    asyncio.run(_leave())


@main.command()
@click.argument("path", type=click.Path(exists=True))
def demo(path: str) -> None:
    """Interactive demo tutorial."""
    workspace = Path(path).expanduser().resolve()
    db_path = workspace / "kairn.db"

    if not db_path.exists():
        click.echo(f"Error: No database at {db_path}. Run 'kairn init' first.", err=True)
        sys.exit(1)

    console = Console()

    async def _demo() -> None:
        from kairn.core.experience import ExperienceEngine
        from kairn.core.graph import GraphEngine
        from kairn.core.ideas import IdeaEngine
        from kairn.core.intelligence import IntelligenceLayer
        from kairn.core.memory import ProjectMemory
        from kairn.core.router import ContextRouter
        from kairn.events.bus import EventBus

        store = SQLiteStore(db_path)
        await store.initialize()
        bus = EventBus()
        graph = GraphEngine(store, bus)
        router = ContextRouter(store, bus)
        memory_eng = ProjectMemory(store, bus)
        experience = ExperienceEngine(store, bus)
        ideas_eng = IdeaEngine(store, bus)
        intel = IntelligenceLayer(
            store=store,
            event_bus=bus,
            graph=graph,
            router=router,
            memory=memory_eng,
            experience=experience,
            ideas=ideas_eng,
        )

        console.print(
            Panel(
                "[bold cyan]Kairn Demo Tutorial[/bold cyan]\n\n"
                "This demo walks through core features:\n"
                "1. Create a node    4. Learn knowledge\n"
                "2. Query nodes      5. Recall knowledge\n"
                "3. Save experience  6. Get context",
                title="Welcome to Kairn",
            )
        )

        console.print("\n[bold]Step 1: Creating a node[/bold]")
        node = await graph.add_node(
            name="JWT Authentication",
            type="auth",
            namespace="knowledge",
            description="Use JWT with refresh tokens for stateless auth",
            tags=["authentication", "security"],
        )
        console.print(f"  [green]✓[/green] Created node: {node.name} ({node.id[:8]}...)")

        console.print("\n[bold]Step 2: Querying nodes[/bold]")
        results = await graph.query(text="authentication", limit=5)
        console.print(f"  [green]✓[/green] Found {len(results)} node(s)")

        console.print("\n[bold]Step 3: Saving an experience[/bold]")
        exp = await experience.save(
            content="Token bucket rate limiting solved API abuse",
            type="solution",
            context="API Gateway sprint",
            confidence="high",
            tags=["rate-limiting"],
        )
        console.print(f"  [green]✓[/green] Saved experience: {exp.id[:8]}...")

        console.print("\n[bold]Step 4: Learning knowledge[/bold]")
        learn_result = await intel.learn(
            content="Redis for session storage improves auth performance",
            type="pattern",
            confidence="high",
            tags=["redis", "auth"],
        )
        console.print(f"  [green]✓[/green] Stored as: {learn_result['stored_as']}")

        console.print("\n[bold]Step 5: Recalling knowledge[/bold]")
        recall_results = await intel.recall(topic="authentication", limit=3)
        console.print(f"  [green]✓[/green] Recalled {len(recall_results)} item(s)")

        console.print("\n[bold]Step 6: Getting context[/bold]")
        ctx = await intel.context(keywords="authentication security", limit=3)
        console.print(f"  [green]✓[/green] Context has {ctx['count']} item(s)")

        await store.close()

        console.print(
            Panel(
                "[bold green]Demo Complete![/bold green]\n\n"
                "You've seen Kairn's core capabilities:\n"
                "  Graph storage and search\n"
                "  Experience tracking with decay\n"
                "  Intelligence: learn, recall, context\n\n"
                "Ready to integrate with your AI workflow!",
                title="Success",
            )
        )

    asyncio.run(_demo())


async def _build_intel_stack(db_path: Path):
    """Build full IntelligenceLayer stack for CLI commands.

    Returns (store, intel) tuple. Caller is responsible for awaiting store.close().
    Reuses the exact wiring pattern from the demo command so MCP and CLI execute
    through the same engines.
    """
    from kairn.core.experience import ExperienceEngine
    from kairn.core.graph import GraphEngine
    from kairn.core.ideas import IdeaEngine
    from kairn.core.intelligence import IntelligenceLayer
    from kairn.core.memory import ProjectMemory
    from kairn.core.router import ContextRouter
    from kairn.events.bus import EventBus

    store = SQLiteStore(db_path)
    await store.initialize()
    bus = EventBus()
    graph = GraphEngine(store, bus)
    router = ContextRouter(store, bus)
    memory_eng = ProjectMemory(store, bus)
    experience = ExperienceEngine(store, bus)
    ideas_eng = IdeaEngine(store, bus)
    intel = IntelligenceLayer(
        store=store,
        event_bus=bus,
        graph=graph,
        router=router,
        memory=memory_eng,
        experience=experience,
        ideas=ideas_eng,
    )
    return store, intel


def _resolve_db(path: str) -> Path:
    """Resolve workspace path to a kairn.db Path, exit on missing."""
    workspace = Path(path).expanduser().resolve()
    db_path = workspace / "kairn.db"
    if not db_path.exists():
        click.echo(
            f"Error: No database at {db_path}. Run 'kairn init' first.",
            err=True,
        )
        sys.exit(1)
    return db_path


def _parse_tags(tags: str | None) -> list[str] | None:
    """Parse comma-separated tag string into list, or None."""
    if not tags:
        return None
    return [t.strip() for t in tags.split(",") if t.strip()]


def _run_json(coro_factory) -> None:
    """Run an async factory and emit its result as JSON, or an error envelope.

    Catches ValueError raised by engines (invalid type/confidence/etc.), emits
    {"_v": "1.0", "error": "..."} to stderr, and exits non-zero. Successful
    results are json.dumps'd to stdout.

    The factory is a zero-arg callable returning a coroutine so each invocation
    starts a fresh event loop cleanly.
    """
    try:
        result = asyncio.run(coro_factory())
    except ValueError as e:
        click.echo(json.dumps({"_v": "1.0", "error": str(e)}), err=True)
        sys.exit(1)
    click.echo(json.dumps(result, default=str))


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--content", required=True, help="Knowledge content to learn")
@click.option(
    "--type",
    "type_",
    required=True,
    help="Experience type: decision|pattern|solution|workaround|gotcha",
)
@click.option("--context", default=None, help="Optional context for the learning")
@click.option(
    "--confidence",
    default="high",
    type=click.Choice(["high", "medium", "low"]),
    help="Confidence level (affects decay rate for experiences)",
)
@click.option("--tags", default=None, help="Comma-separated tags")
@click.option(
    "--namespace",
    default="knowledge",
    help="Namespace for multi-tenant isolation (default 'knowledge')",
)
def learn(
    path: str,
    content: str,
    type_: str,
    context: str | None,
    confidence: str,
    tags: str | None,
    namespace: str,
) -> None:
    """Store knowledge.

    High confidence creates a permanent node + experience.
    Medium/low confidence creates a decaying experience only.
    """
    db_path = _resolve_db(path)
    tag_list = _parse_tags(tags)

    async def _run() -> dict:
        store, intel = await _build_intel_stack(db_path)
        try:
            return await intel.learn(
                content=content,
                type=type_,
                context=context,
                confidence=confidence,
                tags=tag_list,
                namespace=namespace,
            )
        finally:
            await store.close()

    _run_json(_run)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--topic", default=None, help="Topic to recall knowledge about")
@click.option("--limit", default=10, type=click.IntRange(1, 50), help="Max results")
@click.option(
    "--min-relevance",
    default=0.0,
    type=click.FloatRange(0.0, 1.0),
    help="Minimum relevance filter",
)
def recall(path: str, topic: str | None, limit: int, min_relevance: float) -> None:
    """Surface relevant past knowledge (cross-searches nodes + experiences)."""
    db_path = _resolve_db(path)

    async def _run() -> dict:
        store, intel = await _build_intel_stack(db_path)
        try:
            results = await intel.recall(
                topic=topic,
                limit=limit,
                min_relevance=min_relevance,
            )
            return {"_v": "1.0", "count": len(results), "results": results}
        finally:
            await store.close()

    _run_json(_run)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--keywords", required=True, help="Keywords to find relevant context")
@click.option(
    "--detail",
    default="summary",
    type=click.Choice(["summary", "full"]),
    help="Detail level",
)
@click.option("--limit", default=10, type=click.IntRange(1, 50), help="Max results per section")
def context(path: str, keywords: str, detail: str, limit: int) -> None:
    """Get relevant context subgraph (nodes + experiences) with progressive disclosure.

    Note: intelligence.context() returns its own versioned envelope (_v, query,
    detail, count, nodes, experiences), so this command is a pass-through. The
    envelope ownership is intentional - engine owns shape, CLI only serializes.
    """
    db_path = _resolve_db(path)

    async def _run() -> dict:
        store, intel = await _build_intel_stack(db_path)
        try:
            return await intel.context(
                keywords=keywords,
                detail=detail,
                limit=limit,
            )
        finally:
            await store.close()

    _run_json(_run)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--text", default=None, help="Full-text search query")
@click.option("--type", "type_", default=None, help="Filter by experience type")
@click.option(
    "--min-relevance",
    default=0.0,
    type=click.FloatRange(0.0, 1.0),
    help="Minimum relevance filter",
)
@click.option("--limit", default=10, type=click.IntRange(1, 50), help="Max results")
@click.option("--offset", default=0, type=click.IntRange(0), help="Pagination offset")
def memories(
    path: str,
    text: str | None,
    type_: str | None,
    min_relevance: float,
    limit: int,
    offset: int,
) -> None:
    """Decay-aware experience search.

    Mirrors the kn_memories MCP tool's experience.search() call. Delegates to
    intel.experience.search() directly (same pattern as server.py kn_memories).
    """
    db_path = _resolve_db(path)

    async def _run() -> dict:
        store, intel = await _build_intel_stack(db_path)
        try:
            experiences = await intel.experience.search(
                text=text,
                exp_type=type_,
                min_relevance=min_relevance,
                limit=limit,
                offset=offset,
            )
            items = [
                {
                    "id": e.id,
                    "type": e.type,
                    "content": e.content,
                    "confidence": e.confidence,
                    "relevance": round(e.relevance(), 4),
                    "tags": e.tags,
                }
                for e in experiences
            ]
            return {"_v": "1.0", "count": len(items), "experiences": items}
        finally:
            await store.close()

    _run_json(_run)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--problem", required=True, help="Problem description to find solutions for")
@click.option("--limit", default=10, type=click.IntRange(1, 50), help="Max results")
def crossref(path: str, problem: str, limit: int) -> None:
    """Find similar solutions in the workspace (cross-references nodes + experiences)."""
    db_path = _resolve_db(path)

    async def _run() -> dict:
        store, intel = await _build_intel_stack(db_path)
        try:
            results = await intel.crossref(problem=problem, limit=limit)
            return {"_v": "1.0", "count": len(results), "results": results}
        finally:
            await store.close()

    _run_json(_run)


# ──────────────────────────────────────────────────────────
# Phase 4: Graph CRUD + query + project/idea/log parity
# ──────────────────────────────────────────────────────────


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--text", default=None, help="Full-text search query")
@click.option("--namespace", default=None, help="Filter by namespace")
@click.option("--node-type", default=None, help="Filter by node type")
@click.option("--tags", default=None, help="Comma-separated tags")
@click.option(
    "--detail",
    default="summary",
    type=click.Choice(["summary", "full"]),
    help="Detail level for node responses",
)
@click.option("--limit", default=10, type=click.IntRange(1, 1000), help="Max results")
@click.option("--offset", default=0, type=click.IntRange(0), help="Pagination offset")
@click.option(
    "--since",
    default=None,
    help="ISO-8601 timestamp: return experiences created at or after this time",
)
@click.option(
    "--format",
    "format_",
    default="envelope",
    type=click.Choice(["envelope", "json"]),
    help="Output format: envelope (default) or raw JSON list (used by sync scripts)",
)
def query(
    path: str,
    text: str | None,
    namespace: str | None,
    node_type: str | None,
    tags: str | None,
    detail: str,
    limit: int,
    offset: int,
    since: str | None,
    format_: str,
) -> None:
    """Query nodes, or experiences by timestamp.

    Two modes:
    - Default: search nodes by text/tags/type/namespace (mirrors kn_query).
    - `--since <iso>`: switches to experience time-range query, useful for
      incremental exports and replication pipelines. With `--format json`,
      emits a bare JSON list so shell consumers can parse directly.
    """
    db_path = _resolve_db(path)
    tag_list = _parse_tags(tags)

    async def _run() -> dict | list:
        store, intel = await _build_intel_stack(db_path)
        try:
            if since:
                rows = await store.query_experiences_since(
                    since,
                    namespace=namespace,
                    limit=limit,
                )
                if format_ == "json":
                    return rows
                return {"_v": "1.0", "count": len(rows), "experiences": rows}

            nodes = await intel.graph.query(
                text=text,
                namespace=namespace,
                node_type=node_type,
                tags=tag_list,
                limit=limit,
                offset=offset,
            )
            items = [n.to_response(detail=detail) for n in nodes]
            return {"_v": "1.0", "count": len(items), "nodes": items}
        finally:
            await store.close()

    _run_json(_run)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", required=True, help="Node name")
@click.option("--type", "type_", required=True, help="Node type (concept, pattern, etc.)")
@click.option("--namespace", default="knowledge", help="Namespace")
@click.option("--description", default=None, help="Node description")
@click.option("--tags", default=None, help="Comma-separated tags")
def add(
    path: str,
    name: str,
    type_: str,
    namespace: str,
    description: str | None,
    tags: str | None,
) -> None:
    """Add a node to the knowledge graph. Auto-links via FTS5."""
    db_path = _resolve_db(path)
    tag_list = _parse_tags(tags)

    async def _run() -> dict:
        store, intel = await _build_intel_stack(db_path)
        try:
            node = await intel.graph.add_node(
                name=name,
                type=type_,
                namespace=namespace,
                description=description,
                tags=tag_list,
            )
            await intel.router.update_routes_for_node(
                node.id,
                node.name,
                node.description,
            )
            result = node.to_response(detail="full")
            result["_v"] = "1.0"
            return result
        finally:
            await store.close()

    _run_json(_run)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--source-id", required=True, help="Source node ID")
@click.option("--target-id", required=True, help="Target node ID")
@click.option("--edge-type", required=True, help="Relationship type")
@click.option(
    "--weight",
    default=1.0,
    type=click.FloatRange(0.0, 1.0),
    help="Edge weight 0.0-1.0",
)
def connect(
    path: str,
    source_id: str,
    target_id: str,
    edge_type: str,
    weight: float,
) -> None:
    """Create a typed, weighted edge between two nodes."""
    db_path = _resolve_db(path)

    async def _run() -> dict:
        store, intel = await _build_intel_stack(db_path)
        try:
            edge = await intel.graph.connect(
                source_id,
                target_id,
                edge_type,
                weight=weight,
            )
            result = edge.to_storage()
            result["_v"] = "1.0"
            return result
        finally:
            await store.close()

    _run_json(_run)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--node-id", default=None, help="Node ID to soft-delete")
@click.option("--source-id", default=None, help="Edge source ID")
@click.option("--target-id", default=None, help="Edge target ID")
@click.option("--edge-type", default=None, help="Edge type")
def remove(
    path: str,
    node_id: str | None,
    source_id: str | None,
    target_id: str | None,
    edge_type: str | None,
) -> None:
    """Soft-delete a node or an edge.

    Pass `--node-id` for node removal, or all three edge flags for an edge.
    """
    if not node_id and not (source_id and target_id and edge_type):
        raise click.UsageError(
            "Provide --node-id OR all of --source-id, --target-id, --edge-type"
        )

    db_path = _resolve_db(path)

    async def _run() -> dict:
        store, intel = await _build_intel_stack(db_path)
        try:
            if node_id:
                ok = await intel.graph.remove_node(node_id)
                if not ok:
                    raise ValueError(f"Node not found: {node_id}")
                return {"_v": "1.0", "removed": "node", "id": node_id}

            # Edge removal
            ok = await intel.graph.disconnect(source_id, target_id, edge_type)
            if not ok:
                raise ValueError("Edge not found")
            return {
                "_v": "1.0",
                "removed": "edge",
                "source_id": source_id,
                "target_id": target_id,
                "edge_type": edge_type,
            }
        finally:
            await store.close()

    _run_json(_run)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--threshold",
    default=0.01,
    type=click.FloatRange(0.0, 1.0),
    help="Remove experiences below this relevance",
)
def prune(path: str, threshold: float) -> None:
    """Remove expired experiences below the relevance threshold."""
    db_path = _resolve_db(path)

    async def _run() -> dict:
        store, intel = await _build_intel_stack(db_path)
        try:
            pruned = await intel.experience.prune(threshold=threshold)
            return {
                "_v": "1.0",
                "pruned_count": len(pruned),
                "pruned_ids": pruned,
            }
        finally:
            await store.close()

    _run_json(_run)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--name", required=True, help="Project name")
@click.option("--project-id", default=None, help="Project ID (omit to create new)")
@click.option("--phase", default=None, help="planning|active|paused|done")
@click.option("--goals", default=None, help="Comma-separated goals")
@click.option("--stakeholders", default=None, help="Comma-separated stakeholders")
@click.option("--success-metrics", default=None, help="Comma-separated success metrics")
def project(
    path: str,
    name: str,
    project_id: str | None,
    phase: str | None,
    goals: str | None,
    stakeholders: str | None,
    success_metrics: str | None,
) -> None:
    """Create or update a project (upsert)."""
    db_path = _resolve_db(path)
    goals_list = _parse_tags(goals)
    stakeholders_list = _parse_tags(stakeholders)
    metrics_list = _parse_tags(success_metrics)

    async def _run() -> dict:
        store, intel = await _build_intel_stack(db_path)
        try:
            mem = intel.memory
            if project_id:
                updates: dict = {"name": name}
                if phase is not None:
                    updates["phase"] = phase
                if goals_list is not None:
                    updates["goals"] = goals_list
                if stakeholders_list is not None:
                    updates["stakeholders"] = stakeholders_list
                if metrics_list is not None:
                    updates["success_metrics"] = metrics_list
                proj = await mem.update_project(project_id, **updates)
                if not proj:
                    raise ValueError(f"Project not found: {project_id}")
            else:
                if phase is not None:
                    raise ValueError(
                        "phase cannot be set on create (starts at planning)"
                    )
                proj = await mem.create_project(
                    name=name,
                    goals=goals_list,
                    stakeholders=stakeholders_list,
                    success_metrics=metrics_list,
                )
            return {
                "_v": "1.0",
                "id": proj.id,
                "name": proj.name,
                "phase": proj.phase,
                "active": proj.active,
                "goals": proj.goals,
            }
        finally:
            await store.close()

    _run_json(_run)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--active-only", is_flag=True, help="Only show active projects")
@click.option("--set-active", default=None, help="Project ID to set as active")
def projects(path: str, active_only: bool, set_active: str | None) -> None:
    """List projects and optionally switch the active one."""
    db_path = _resolve_db(path)

    async def _run() -> dict:
        store, intel = await _build_intel_stack(db_path)
        try:
            mem = intel.memory
            if set_active:
                ok = await mem.set_active_project(set_active)
                if not ok:
                    raise ValueError(f"Project not found: {set_active}")
            proj_list = await mem.list_projects(active_only=active_only)
            items = [
                {
                    "id": p.id,
                    "name": p.name,
                    "phase": p.phase,
                    "active": p.active,
                }
                for p in proj_list
            ]
            return {"_v": "1.0", "count": len(items), "projects": items}
        finally:
            await store.close()

    _run_json(_run)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--project-id", required=True, help="Project ID")
@click.option("--action", required=True, help="What was done (or what failed)")
@click.option(
    "--type",
    "type_",
    default="progress",
    type=click.Choice(["progress", "failure"]),
    help="Entry type",
)
@click.option("--result", default=None, help="Outcome or error message")
@click.option("--next-step", default=None, help="Recommended next step")
def log(
    path: str,
    project_id: str,
    action: str,
    type_: str,
    result: str | None,
    next_step: str | None,
) -> None:
    """Log a progress or failure entry against a project."""
    db_path = _resolve_db(path)

    async def _run() -> dict:
        store, intel = await _build_intel_stack(db_path)
        try:
            mem = intel.memory
            if type_ == "failure":
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
            return {
                "_v": "1.0",
                "id": entry.id,
                "project_id": entry.project_id,
                "type": entry.type,
                "action": entry.action,
            }
        finally:
            await store.close()

    _run_json(_run)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--title", required=True, help="Idea title")
@click.option("--idea-id", default=None, help="Idea ID (omit to create new)")
@click.option("--category", default=None, help="Category classification")
@click.option("--score", default=None, type=float, help="Numerical score")
@click.option(
    "--status",
    default=None,
    help="draft|evaluating|approved|implementing|done|archived",
)
@click.option("--link-to", default=None, help="Node ID to link this idea to")
def idea(
    path: str,
    title: str,
    idea_id: str | None,
    category: str | None,
    score: float | None,
    status: str | None,
    link_to: str | None,
) -> None:
    """Create or update an idea, optionally linking it to a graph node."""
    db_path = _resolve_db(path)

    async def _run() -> dict:
        store, intel = await _build_intel_stack(db_path)
        try:
            ideas_engine = intel.ideas
            if idea_id:
                updates: dict = {"title": title}
                if category is not None:
                    updates["category"] = category
                if score is not None:
                    updates["score"] = score
                if status is not None:
                    updates["status"] = status
                obj = await ideas_engine.update(idea_id, **updates)
                if not obj:
                    raise ValueError(f"Idea not found: {idea_id}")
            else:
                obj = await ideas_engine.create(
                    title=title,
                    category=category,
                    score=score,
                )

            result: dict = {
                "_v": "1.0",
                "id": obj.id,
                "title": obj.title,
                "status": obj.status,
                "category": obj.category,
                "score": obj.score,
            }
            if link_to:
                edge = await ideas_engine.link_to_node(obj.id, link_to)
                result["linked_to"] = link_to if edge else None
                if not edge:
                    result["link_error"] = f"Node not found: {link_to}"
            return result
        finally:
            await store.close()

    _run_json(_run)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--status", default=None, help="Filter by status")
@click.option("--category", default=None, help="Filter by category")
@click.option("--limit", default=10, type=click.IntRange(1, 50), help="Max results")
@click.option("--offset", default=0, type=click.IntRange(0), help="Pagination offset")
def ideas(
    path: str,
    status: str | None,
    category: str | None,
    limit: int,
    offset: int,
) -> None:
    """List and filter ideas by status or category."""
    db_path = _resolve_db(path)

    async def _run() -> dict:
        store, intel = await _build_intel_stack(db_path)
        try:
            ideas_list = await intel.ideas.list_ideas(
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
            return {"_v": "1.0", "count": len(items), "ideas": items}
        finally:
            await store.close()

    _run_json(_run)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--node-id", required=True, help="Starting node ID")
@click.option("--depth", default=1, type=click.IntRange(1, 5), help="Traversal depth")
@click.option("--edge-type", default=None, help="Filter by edge type")
def related(
    path: str,
    node_id: str,
    depth: int,
    edge_type: str | None,
) -> None:
    """Find nodes connected to a starting point (via the intelligence layer)."""
    db_path = _resolve_db(path)

    async def _run() -> dict:
        store, intel = await _build_intel_stack(db_path)
        try:
            results = await intel.related(
                node_id=node_id,
                depth=depth,
                edge_type=edge_type,
            )
            return {"_v": "1.0", "count": len(results), "results": results}
        finally:
            await store.close()

    _run_json(_run)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--nodes", default=1000, help="Number of nodes to create")
def benchmark(path: str, nodes: int) -> None:
    """Run performance benchmarks."""
    workspace = Path(path).expanduser().resolve()
    db_path = workspace / "kairn.db"

    if not db_path.exists():
        click.echo(f"Error: No database at {db_path}. Run 'kairn init' first.", err=True)
        sys.exit(1)

    console = Console()

    async def _benchmark() -> None:
        from kairn.core.graph import GraphEngine
        from kairn.events.bus import EventBus

        store = SQLiteStore(db_path)
        await store.initialize()
        bus = EventBus()
        graph = GraphEngine(store, bus)

        console.print(
            Panel(
                f"[bold cyan]Performance Benchmark[/bold cyan]\n\n"
                f"Creating {nodes} nodes and measuring performance...",
                title="Kairn Benchmark",
            )
        )

        table = Table(title="Benchmark Results")
        table.add_column("Operation", style="cyan")
        table.add_column("Time (s)", style="magenta")
        table.add_column("Ops/sec", style="green")

        # Insert benchmark
        node_ids: list[str] = []
        start = time.time()
        for i in range(nodes):
            node = await graph.add_node(
                name=f"bench-node-{i}",
                type="test",
                namespace="knowledge",
                description=f"Benchmark node {i} with content for search testing",
            )
            node_ids.append(node.id)
        insert_time = time.time() - start
        table.add_row("Insert", f"{insert_time:.2f}", f"{nodes / insert_time:.0f}")

        # FTS5 query benchmark
        iterations = 100
        start = time.time()
        for _ in range(iterations):
            await graph.query(text="benchmark search testing", limit=10)
        query_time = (time.time() - start) / iterations
        table.add_row("FTS5 Query", f"{query_time:.4f}", f"{1 / query_time:.0f}")

        # Graph traversal benchmark
        if len(node_ids) >= 2:
            await graph.connect(node_ids[0], node_ids[1], "test_link")
            start = time.time()
            for _ in range(iterations):
                await graph.get_related(node_ids[0], depth=1)
            traverse_time = (time.time() - start) / iterations
            table.add_row("Graph Traversal", f"{traverse_time:.4f}", f"{1 / traverse_time:.0f}")

        console.print(table)

        # Cleanup
        console.print("\n[yellow]Cleaning up test data...[/yellow]")
        for node_id in node_ids:
            await store.soft_delete_node(node_id)
        await store.close()
        console.print("[green]✓[/green] Cleanup complete")

    asyncio.run(_benchmark())


@main.command()
@click.argument("path", type=click.Path(exists=True))
def token_audit(path: str) -> None:
    """Count tokens in tool definitions."""
    workspace = Path(path).expanduser().resolve()
    db_path = workspace / "kairn.db"

    if not db_path.exists():
        click.echo(f"Error: No database at {db_path}. Run 'kairn init' first.", err=True)
        sys.exit(1)

    console = Console()

    async def _token_audit() -> None:
        from fastmcp import Client

        from kairn.server import create_server

        server = create_server(str(db_path))
        async with Client(server) as client:
            tools = await client.list_tools()

        table = Table(title="Token Audit")
        table.add_column("Tool", style="cyan")
        table.add_column("Estimated Tokens", style="magenta", justify="right")

        total_tokens = 0
        for tool in tools:
            tool_text = f"{tool.name} {tool.description or ''}"
            if hasattr(tool, "inputSchema") and tool.inputSchema:
                tool_text += f" {json.dumps(tool.inputSchema)}"

            words = len(tool_text.split())
            estimated_tokens = int(words * 1.3)
            total_tokens += estimated_tokens
            table.add_row(tool.name, str(estimated_tokens))

        table.add_row("[bold]TOTAL[/bold]", f"[bold]{total_tokens}[/bold]")
        console.print(table)

        if total_tokens > 3000:
            console.print(
                f"\n[red]Warning: Total tokens ({total_tokens}) exceeds target (3000)[/red]"
            )
        else:
            console.print(
                f"\n[green]✓ Token count ({total_tokens}) is within target (3000)[/green]"
            )

    asyncio.run(_token_audit())


if __name__ == "__main__":
    main()
