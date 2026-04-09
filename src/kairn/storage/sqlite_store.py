"""SQLite storage backend with FTS5 and WAL mode."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import aiosqlite

from kairn.storage.base import StorageBackend

logger = logging.getLogger(__name__)

# Column whitelists per table — prevents SQL injection in UPDATE operations
_ALLOWED_COLUMNS: dict[str, set[str]] = {
    "nodes": {
        "namespace",
        "type",
        "name",
        "description",
        "properties",
        "tags",
        "visibility",
        "source_type",
        "source_ref",
        "created_by",
        "updated_at",
        "deleted_at",
    },
    "experiences": {
        "namespace",
        "type",
        "content",
        "context",
        "confidence",
        "score",
        "decay_rate",
        "tags",
        "created_by",
        "access_count",
        "promoted_to_node_id",
        "last_accessed",
        "properties",
    },
    "projects": {
        "name",
        "phase",
        "goals",
        "active",
        "created_by",
        "stakeholders",
        "success_metrics",
        "updated_at",
    },
    "ideas": {
        "title",
        "status",
        "category",
        "score",
        "properties",
        "created_by",
        "visibility",
        "updated_at",
    },
}


def _validate_update_keys(table: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Filter update dict to only allowed column names."""
    allowed = _ALLOWED_COLUMNS.get(table, set())
    filtered = {k: v for k, v in updates.items() if k in allowed}
    rejected = set(updates.keys()) - allowed - {"id"}
    if rejected:
        logger.warning("Rejected invalid column names for %s: %s", table, rejected)
    return filtered


class SQLiteStore(StorageBackend):
    """SQLite-based storage with FTS5 full-text search and WAL mode."""

    def __init__(self, db_path: Path, *, wal_mode: bool = True) -> None:
        self.db_path = db_path
        self.wal_mode = wal_mode
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create database, apply schema and triggers."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row

        if self.wal_mode:
            await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")

        # Apply workspace schema
        schema_sql = _load_sql("workspace.sql")
        await self._db.executescript(schema_sql)

        # Apply triggers (FTS5 + auto-promotion)
        triggers_sql = _load_sql("triggers.sql")
        await self._db.executescript(triggers_sql)

        # Idempotent migrations for pre-existing databases
        await self._migrate_schema()

        await self._db.commit()
        logger.info("Initialized SQLite store at %s", self.db_path)

    async def _migrate_schema(self) -> None:
        """Apply idempotent schema migrations for older databases.

        Each migration checks current state before acting so running on a
        fresh DB (where CREATE TABLE already includes the column) is a no-op.

        Concurrency assumption: Kairn runs single-process per workspace
        (each MCP server / CLI invocation owns its own SQLite connection).
        The PRAGMA-check then ALTER-TABLE window is not guarded beyond
        SQLite's own write serialization, so simultaneous init of a fresh
        legacy DB from two processes could theoretically race. Not
        exploitable under current deployment shape — flagged for the
        future if multi-writer init becomes a thing.
        """
        assert self._db is not None

        # Migration: experiences.namespace (added 2026-04-09)
        cursor = await self._db.execute("PRAGMA table_info(experiences)")
        columns = {row[1] for row in await cursor.fetchall()}
        if "namespace" not in columns:
            logger.info("Migrating experiences: adding namespace column")
            # SQLite rejects NOT NULL on ALTER TABLE ADD COLUMN unless the
            # default is non-null — which 'knowledge' is, so this is safe.
            await self._db.execute(
                "ALTER TABLE experiences ADD COLUMN namespace TEXT NOT NULL "
                "DEFAULT 'knowledge'"
            )

        # Always ensure the namespace index exists (idempotent on fresh and
        # migrated DBs). The CREATE INDEX was intentionally removed from
        # workspace.sql so legacy DBs can be migrated without the initial
        # executescript failing on a missing column.
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiences_namespace "
            "ON experiences(namespace)"
        )

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")
        return self._db

    # --- Node operations ---

    async def insert_node(self, node: dict[str, Any]) -> dict[str, Any]:
        await self.db.execute(
            """INSERT INTO nodes (id, namespace, type, name, description,
               properties, tags, created_by, visibility, source_type,
               source_ref, created_at, updated_at)
               VALUES (:id, :namespace, :type, :name, :description,
               :properties, :tags, :created_by, :visibility, :source_type,
               :source_ref, :created_at, :updated_at)""",
            _serialize_json_fields(node, ["properties", "tags"]),
        )
        await self.db.commit()
        return node

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        cursor = await self.db.execute(
            "SELECT * FROM nodes WHERE id = ? AND deleted_at IS NULL", (node_id,)
        )
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None

    async def update_node(self, node_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        existing = await self.get_node(node_id)
        if not existing:
            return None

        updates = _validate_update_keys("nodes", updates)
        updates = _serialize_json_fields(updates, ["properties", "tags"])
        set_clauses = []
        values = []
        for key, value in updates.items():
            set_clauses.append(f"{key} = ?")
            values.append(value)

        if not set_clauses:
            return existing

        values.append(node_id)
        await self.db.execute(
            f"UPDATE nodes SET {', '.join(set_clauses)} WHERE id = ?",
            values,
        )
        await self.db.commit()
        return await self.get_node(node_id)

    async def soft_delete_node(self, node_id: str) -> bool:
        cursor = await self.db.execute(
            "UPDATE nodes SET deleted_at = datetime('now') WHERE id = ? AND deleted_at IS NULL",
            (node_id,),
        )
        await self.db.commit()
        return cursor.rowcount > 0

    async def restore_node(self, node_id: str) -> bool:
        cursor = await self.db.execute(
            "UPDATE nodes SET deleted_at = NULL WHERE id = ? AND deleted_at IS NOT NULL",
            (node_id,),
        )
        await self.db.commit()
        return cursor.rowcount > 0

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
        if text:
            return await self._query_nodes_fts(
                text,
                namespace=namespace,
                node_type=node_type,
                visibility=visibility,
                limit=limit,
                offset=offset,
            )

        conditions = ["deleted_at IS NULL"]
        params: list[Any] = []

        if namespace:
            conditions.append("namespace = ?")
            params.append(namespace)
        if node_type:
            conditions.append("type = ?")
            params.append(node_type)
        if visibility:
            conditions.append("visibility = ?")
            params.append(visibility)
        if tags:
            for tag in tags:
                conditions.append("json_each.value = ?")
                params.append(tag)

        where = " AND ".join(conditions)

        if tags:
            query = f"""
                SELECT DISTINCT nodes.* FROM nodes, json_each(nodes.tags)
                WHERE {where}
                ORDER BY nodes.updated_at DESC NULLS LAST, nodes.created_at DESC
                LIMIT ? OFFSET ?
            """
        else:
            query = f"""
                SELECT * FROM nodes
                WHERE {where}
                ORDER BY updated_at DESC NULLS LAST, created_at DESC
                LIMIT ? OFFSET ?
            """

        params.extend([limit, offset])
        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    async def _query_nodes_fts(
        self,
        text: str,
        *,
        namespace: str | None = None,
        node_type: str | None = None,
        visibility: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        conditions = ["nodes.deleted_at IS NULL"]
        params: list[Any] = [text]

        if namespace:
            conditions.append("nodes.namespace = ?")
            params.append(namespace)
        if node_type:
            conditions.append("nodes.type = ?")
            params.append(node_type)
        if visibility:
            conditions.append("nodes.visibility = ?")
            params.append(visibility)

        where = " AND ".join(conditions)
        query = f"""
            SELECT nodes.*, rank FROM nodes_fts
            JOIN nodes ON nodes.rowid = nodes_fts.rowid
            WHERE nodes_fts MATCH ? AND {where}
            ORDER BY rank
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    async def count_nodes(self, *, namespace: str | None = None) -> int:
        if namespace:
            cursor = await self.db.execute(
                "SELECT COUNT(*) FROM nodes WHERE namespace = ? AND deleted_at IS NULL",
                (namespace,),
            )
        else:
            cursor = await self.db.execute("SELECT COUNT(*) FROM nodes WHERE deleted_at IS NULL")
        row = await cursor.fetchone()
        return row[0] if row else 0

    # --- Edge operations ---

    async def insert_edge(self, edge: dict[str, Any]) -> dict[str, Any]:
        await self.db.execute(
            """INSERT INTO edges (source_id, target_id, type, weight,
               properties, created_by, created_at)
               VALUES (:source_id, :target_id, :type, :weight,
               :properties, :created_by, :created_at)""",
            _serialize_json_fields(edge, ["properties"]),
        )
        await self.db.commit()
        return edge

    async def get_edges(
        self,
        *,
        source_id: str | None = None,
        target_id: str | None = None,
        edge_type: str | None = None,
    ) -> list[dict[str, Any]]:
        conditions = []
        params: list[Any] = []

        if source_id:
            conditions.append("source_id = ?")
            params.append(source_id)
        if target_id:
            conditions.append("target_id = ?")
            params.append(target_id)
        if edge_type:
            conditions.append("type = ?")
            params.append(edge_type)

        where = " AND ".join(conditions) if conditions else "1=1"
        cursor = await self.db.execute(f"SELECT * FROM edges WHERE {where}", params)
        rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    async def delete_edge(self, source_id: str, target_id: str, edge_type: str) -> bool:
        cursor = await self.db.execute(
            "DELETE FROM edges WHERE source_id = ? AND target_id = ? AND type = ?",
            (source_id, target_id, edge_type),
        )
        await self.db.commit()
        return cursor.rowcount > 0

    async def count_edges(self) -> int:
        cursor = await self.db.execute("SELECT COUNT(*) FROM edges")
        row = await cursor.fetchone()
        return row[0] if row else 0

    # --- Experience operations ---

    async def insert_experience(self, experience: dict[str, Any]) -> dict[str, Any]:
        # Ensure namespace is always present (defaults to 'knowledge' for
        # callers that pre-date the namespace migration).
        payload = dict(experience)
        payload.setdefault("namespace", "knowledge")
        await self.db.execute(
            """INSERT INTO experiences (id, namespace, type, content, context, confidence, score,
               decay_rate, tags, properties, created_by, access_count, promoted_to_node_id,
               created_at, last_accessed)
               VALUES (:id, :namespace, :type, :content, :context, :confidence, :score,
               :decay_rate, :tags, :properties, :created_by, :access_count, :promoted_to_node_id,
               :created_at, :last_accessed)""",
            _serialize_json_fields(payload, ["tags", "properties"]),
        )
        await self.db.commit()
        return payload

    async def get_experience(self, exp_id: str) -> dict[str, Any] | None:
        cursor = await self.db.execute("SELECT * FROM experiences WHERE id = ?", (exp_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None

    async def update_experience(
        self, exp_id: str, updates: dict[str, Any]
    ) -> dict[str, Any] | None:
        existing = await self.get_experience(exp_id)
        if not existing:
            return None

        updates = _validate_update_keys("experiences", updates)
        updates = _serialize_json_fields(updates, ["tags", "properties"])
        set_clauses = []
        values = []
        for key, value in updates.items():
            set_clauses.append(f"{key} = ?")
            values.append(value)

        if not set_clauses:
            return existing

        values.append(exp_id)
        await self.db.execute(
            f"UPDATE experiences SET {', '.join(set_clauses)} WHERE id = ?",
            values,
        )
        await self.db.commit()
        return await self.get_experience(exp_id)

    async def query_experiences(
        self,
        *,
        exp_type: str | None = None,
        text: str | None = None,
        min_score: float | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if text:
            return await self._query_experiences_fts(
                text, exp_type=exp_type, min_score=min_score, limit=limit, offset=offset
            )

        conditions: list[str] = []
        params: list[Any] = []

        if exp_type:
            conditions.append("type = ?")
            params.append(exp_type)
        if min_score is not None:
            conditions.append("score >= ?")
            params.append(min_score)

        where = " AND ".join(conditions) if conditions else "1=1"
        query = f"""
            SELECT * FROM experiences WHERE {where}
            ORDER BY score DESC, created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    async def _query_experiences_fts(
        self,
        text: str,
        *,
        exp_type: str | None = None,
        min_score: float | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = [text]

        if exp_type:
            conditions.append("experiences.type = ?")
            params.append(exp_type)
        if min_score is not None:
            conditions.append("experiences.score >= ?")
            params.append(min_score)

        extra_where = (" AND " + " AND ".join(conditions)) if conditions else ""
        query = f"""
            SELECT experiences.*, rank FROM experiences_fts
            JOIN experiences ON experiences.rowid = experiences_fts.rowid
            WHERE experiences_fts MATCH ?{extra_where}
            ORDER BY rank
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    async def increment_access_count(self, exp_id: str) -> dict[str, Any] | None:
        cursor = await self.db.execute(
            """UPDATE experiences
               SET access_count = access_count + 1, last_accessed = datetime('now')
               WHERE id = ?""",
            (exp_id,),
        )
        await self.db.commit()
        if cursor.rowcount == 0:
            return None
        return await self.get_experience(exp_id)

    async def touch_accessed_experiences(self, exp_ids: list[str]) -> int:
        """Batch UPDATE of access_count + last_accessed for a list of IDs.

        Single SQL statement so it fires the `exp_auto_promote` trigger once
        per row in one pass. Empty list is a no-op.
        """
        if not exp_ids:
            return 0
        placeholders = ",".join("?" * len(exp_ids))
        # Placeholders are generated from len(exp_ids) only, never from user
        # input — this is safe parameterized SQL. SQLite's host-parameter
        # limit is 32766 (SQLITE_MAX_VARIABLE_NUMBER) on modern builds;
        # if typical call sites ever exceed ~10k IDs in a single batch,
        # chunk the list before calling.
        cursor = await self.db.execute(
            f"""UPDATE experiences
                SET access_count = access_count + 1,
                    last_accessed = datetime('now')
                WHERE id IN ({placeholders})""",
            list(exp_ids),
        )
        await self.db.commit()
        return cursor.rowcount or 0

    async def delete_experience(self, exp_id: str) -> bool:
        cursor = await self.db.execute("DELETE FROM experiences WHERE id = ?", (exp_id,))
        await self.db.commit()
        return cursor.rowcount > 0

    async def query_experiences_since(
        self,
        since: str,
        *,
        namespace: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Return experiences created at or after `since` (ISO-8601 UTC string).

        Used by the `kairn query --since` CLI subcommand that powers
        bidirectional sync between Kairn workspaces. Returns newest first.
        """
        conditions = ["created_at >= ?"]
        params: list[Any] = [since]
        if namespace:
            conditions.append("namespace = ?")
            params.append(namespace)
        where = " AND ".join(conditions)
        params.append(limit)
        cursor = await self.db.execute(
            f"SELECT * FROM experiences WHERE {where} "
            f"ORDER BY created_at DESC LIMIT ?",
            params,
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    async def get_promotable_experiences(self) -> list[dict[str, Any]]:
        cursor = await self.db.execute(
            """SELECT * FROM experiences
               WHERE json_extract(properties, '$.needs_promotion') = 1
               AND promoted_to_node_id IS NULL"""
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    # --- Project operations ---

    async def insert_project(self, project: dict[str, Any]) -> dict[str, Any]:
        await self.db.execute(
            """INSERT INTO projects (id, name, phase, goals, active, created_by,
               stakeholders, success_metrics, created_at, updated_at)
               VALUES (:id, :name, :phase, :goals, :active, :created_by,
               :stakeholders, :success_metrics, :created_at, :updated_at)""",
            _serialize_json_fields(project, ["goals", "stakeholders", "success_metrics"]),
        )
        await self.db.commit()
        return project

    async def get_project(self, project_id: str) -> dict[str, Any] | None:
        cursor = await self.db.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None

    async def update_project(
        self, project_id: str, updates: dict[str, Any]
    ) -> dict[str, Any] | None:
        existing = await self.get_project(project_id)
        if not existing:
            return None

        updates = _validate_update_keys("projects", updates)
        updates = _serialize_json_fields(updates, ["goals", "stakeholders", "success_metrics"])
        set_clauses = []
        values = []
        for key, value in updates.items():
            if key != "id":
                set_clauses.append(f"{key} = ?")
                values.append(value)

        if not set_clauses:
            return existing

        values.append(project_id)
        await self.db.execute(
            f"UPDATE projects SET {', '.join(set_clauses)} WHERE id = ?",
            values,
        )
        await self.db.commit()
        return await self.get_project(project_id)

    async def list_projects(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        if active_only:
            cursor = await self.db.execute(
                "SELECT * FROM projects WHERE active = 1 ORDER BY updated_at DESC"
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM projects ORDER BY active DESC, updated_at DESC"
            )
        rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    async def set_active_project(self, project_id: str) -> bool:
        project = await self.get_project(project_id)
        if not project:
            return False
        await self.db.execute("UPDATE projects SET active = 0")
        await self.db.execute("UPDATE projects SET active = 1 WHERE id = ?", (project_id,))
        await self.db.commit()
        return True

    # --- Progress operations ---

    async def insert_progress(self, entry: dict[str, Any]) -> dict[str, Any]:
        await self.db.execute(
            """INSERT INTO progress (id, project_id, type, action, result, next_step,
               created_by, created_at)
               VALUES (:id, :project_id, :type, :action, :result, :next_step,
               :created_by, :created_at)""",
            entry,
        )
        await self.db.commit()
        return entry

    async def get_progress(
        self, project_id: str, *, entry_type: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        if entry_type:
            cursor = await self.db.execute(
                "SELECT * FROM progress WHERE project_id = ? AND type = ?"
                " ORDER BY created_at DESC LIMIT ?",
                (project_id, entry_type, limit),
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM progress WHERE project_id = ? ORDER BY created_at DESC LIMIT ?",
                (project_id, limit),
            )
        rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    # --- Idea operations ---

    async def insert_idea(self, idea: dict[str, Any]) -> dict[str, Any]:
        await self.db.execute(
            """INSERT INTO ideas (id, title, status, category, score, properties,
               created_by, visibility, created_at, updated_at)
               VALUES (:id, :title, :status, :category, :score, :properties,
               :created_by, :visibility, :created_at, :updated_at)""",
            _serialize_json_fields(idea, ["properties"]),
        )
        await self.db.commit()
        return idea

    async def get_idea(self, idea_id: str) -> dict[str, Any] | None:
        cursor = await self.db.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None

    async def update_idea(self, idea_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        existing = await self.get_idea(idea_id)
        if not existing:
            return None

        updates = _validate_update_keys("ideas", updates)
        updates = _serialize_json_fields(updates, ["properties"])
        set_clauses = []
        values = []
        for key, value in updates.items():
            set_clauses.append(f"{key} = ?")
            values.append(value)

        if not set_clauses:
            return existing

        values.append(idea_id)
        await self.db.execute(
            f"UPDATE ideas SET {', '.join(set_clauses)} WHERE id = ?",
            values,
        )
        await self.db.commit()
        return await self.get_idea(idea_id)

    async def list_ideas(
        self,
        *,
        status: str | None = None,
        category: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[Any] = []

        if status:
            conditions.append("status = ?")
            params.append(status)
        if category:
            conditions.append("category = ?")
            params.append(category)

        where = " AND ".join(conditions) if conditions else "1=1"
        query = f"""
            SELECT * FROM ideas WHERE {where}
            ORDER BY score DESC NULLS LAST, created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        cursor = await self.db.execute(query, params)
        rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    # --- Route operations ---

    async def upsert_route(self, keyword: str, node_ids: list[str], confidence: float) -> None:
        await self.db.execute(
            """INSERT INTO routes (keyword, node_ids, confidence)
               VALUES (?, ?, ?)
               ON CONFLICT(keyword) DO UPDATE SET node_ids = ?, confidence = ?""",
            (keyword, json.dumps(node_ids), confidence, json.dumps(node_ids), confidence),
        )
        await self.db.commit()

    async def get_routes(self, keywords: list[str]) -> list[dict[str, Any]]:
        placeholders = ",".join("?" * len(keywords))
        cursor = await self.db.execute(
            f"SELECT * FROM routes WHERE keyword IN ({placeholders})",
            keywords,
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    # --- Activity log ---

    async def log_activity(self, entry: dict[str, Any]) -> None:
        await self.db.execute(
            """INSERT INTO activity_log (id, user_id, activity_type, entity_type,
               entity_id, description, created_at)
               VALUES (:id, :user_id, :activity_type, :entity_type,
               :entity_id, :description, :created_at)""",
            entry,
        )
        await self.db.commit()

    async def get_activity_log(
        self, *, entity_type: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        if entity_type:
            cursor = await self.db.execute(
                "SELECT * FROM activity_log WHERE entity_type = ? ORDER BY created_at DESC LIMIT ?",
                (entity_type, limit),
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM activity_log ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    # --- Stats ---

    async def get_stats(self) -> dict[str, Any]:
        node_count = await self.count_nodes()
        edge_count = await self.count_edges()

        cursor = await self.db.execute("SELECT COUNT(*) FROM experiences")
        row = await cursor.fetchone()
        exp_count = row[0] if row else 0

        cursor = await self.db.execute("SELECT COUNT(*) FROM projects")
        row = await cursor.fetchone()
        project_count = row[0] if row else 0

        cursor = await self.db.execute("SELECT COUNT(*) FROM ideas")
        row = await cursor.fetchone()
        idea_count = row[0] if row else 0

        # Namespace breakdown
        cursor = await self.db.execute(
            "SELECT namespace, COUNT(*) as count FROM nodes"
            " WHERE deleted_at IS NULL GROUP BY namespace"
        )
        rows = await cursor.fetchall()
        namespaces = {row["namespace"]: row["count"] for row in rows}

        return {
            "nodes": node_count,
            "edges": edge_count,
            "experiences": exp_count,
            "projects": project_count,
            "ideas": idea_count,
            "namespaces": namespaces,
            "db_path": str(self.db_path),
        }


# --- Helpers ---


def _load_sql(filename: str) -> str:
    """Load SQL file from the schema package."""
    schema_dir = Path(__file__).parent.parent / "schema"
    return (schema_dir / filename).read_text()


def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    """Convert an aiosqlite Row to a dict, deserializing JSON fields."""
    d = dict(row)
    for key in ("properties", "tags", "goals", "stakeholders", "success_metrics", "node_ids"):
        if key in d and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                pass
    return d


def _serialize_json_fields(data: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """Serialize dict/list fields to JSON strings for SQLite storage."""
    result = dict(data)
    for field in fields:
        if field in result and not isinstance(result[field], str) and result[field] is not None:
            result[field] = json.dumps(result[field])
    return result
