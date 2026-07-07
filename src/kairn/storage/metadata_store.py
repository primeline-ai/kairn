"""Central metadata store for users, organizations, and workspaces."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


class MetadataStore:
    """SQLite-based metadata store for team features."""

    def __init__(self, db_path: Path, *, wal_mode: bool = True) -> None:
        self.db_path = db_path
        self.wal_mode = wal_mode
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create database and apply schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        try:
            self._db.row_factory = aiosqlite.Row

            if self.wal_mode:
                await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA foreign_keys=ON")

            schema_sql = _load_sql("metadata.sql")
            await self._db.executescript(schema_sql)
            await self._db.commit()
        except BaseException:
            # The open connection owns a non-daemon worker thread; callers
            # only close stores that initialized successfully, so a failure
            # past connect() must release the connection here or the thread
            # leaks and blocks interpreter shutdown.
            db, self._db = self._db, None
            await db.close()
            raise
        logger.info("Initialized metadata store at %s", self.db_path)

    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        """Get database connection."""
        if self._db is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")
        return self._db

    async def create_user(
        self,
        user_id: str,
        email: str,
        name: str,
        *,
        auth_provider: str = "local",
    ) -> dict[str, Any]:
        """Create a new user."""
        now = datetime.now(UTC).isoformat()
        await self.db.execute(
            """INSERT INTO users (user_id, email, name, created_at, auth_provider, is_active)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, email, name, now, auth_provider, True),
        )
        await self.db.commit()
        return {
            "user_id": user_id,
            "email": email,
            "name": name,
            "created_at": now,
            "auth_provider": auth_provider,
            "is_active": True,
            "last_active": None,
        }

    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        """Get a user by ID."""
        cursor = await self.db.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None

    async def update_user(self, user_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update a user."""
        existing = await self.get_user(user_id)
        if not existing:
            return None

        allowed_fields = {"name", "last_active", "is_active"}
        filtered = {k: v for k, v in updates.items() if k in allowed_fields}

        if not filtered:
            return existing

        set_clauses = []
        values = []
        for key, value in filtered.items():
            set_clauses.append(f"{key} = ?")
            values.append(value)

        values.append(user_id)
        await self.db.execute(
            f"UPDATE users SET {', '.join(set_clauses)} WHERE user_id = ?",
            values,
        )
        await self.db.commit()
        return await self.get_user(user_id)

    async def list_users(self) -> list[dict[str, Any]]:
        """List all users."""
        cursor = await self.db.execute("SELECT * FROM users ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    async def create_org(
        self,
        org_id: str,
        name: str,
        created_by: str,
        *,
        plan_tier: str = "free",
        max_workspaces: int = 1,
        max_members: int = 1,
    ) -> dict[str, Any]:
        """Create a new organization."""
        now = datetime.now(UTC).isoformat()
        await self.db.execute(
            """INSERT INTO organizations (org_id, name, created_at, created_by,
               plan_tier, max_workspaces, max_members)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (org_id, name, now, created_by, plan_tier, max_workspaces, max_members),
        )
        await self.db.commit()
        return {
            "org_id": org_id,
            "name": name,
            "created_at": now,
            "created_by": created_by,
            "plan_tier": plan_tier,
            "max_workspaces": max_workspaces,
            "max_members": max_members,
        }

    async def get_org(self, org_id: str) -> dict[str, Any] | None:
        """Get an organization by ID."""
        cursor = await self.db.execute("SELECT * FROM organizations WHERE org_id = ?", (org_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None

    async def list_orgs(self) -> list[dict[str, Any]]:
        """List all organizations."""
        cursor = await self.db.execute("SELECT * FROM organizations ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    async def create_workspace(
        self,
        workspace_id: str,
        org_id: str,
        name: str,
        created_by: str,
        *,
        description: str | None = None,
        visibility: str = "org",
        workspace_type: str = "project",
        repo_url: str | None = None,
        tech_stack: str | None = None,
    ) -> dict[str, Any]:
        """Create a new workspace."""
        now = datetime.now(UTC).isoformat()
        await self.db.execute(
            """INSERT INTO workspaces (workspace_id, org_id, name, description,
               created_at, created_by, visibility, workspace_type, repo_url, tech_stack)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                workspace_id,
                org_id,
                name,
                description,
                now,
                created_by,
                visibility,
                workspace_type,
                repo_url,
                tech_stack,
            ),
        )
        await self.db.commit()
        return {
            "workspace_id": workspace_id,
            "org_id": org_id,
            "name": name,
            "description": description,
            "created_at": now,
            "created_by": created_by,
            "visibility": visibility,
            "workspace_type": workspace_type,
            "repo_url": repo_url,
            "tech_stack": tech_stack,
        }

    async def get_workspace(self, workspace_id: str) -> dict[str, Any] | None:
        """Get a workspace by ID."""
        cursor = await self.db.execute(
            "SELECT * FROM workspaces WHERE workspace_id = ?", (workspace_id,)
        )
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None

    async def list_workspaces(self) -> list[dict[str, Any]]:
        """List all workspaces."""
        cursor = await self.db.execute("SELECT * FROM workspaces ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    async def list_workspaces_for_user(self, user_id: str) -> list[dict[str, Any]]:
        """List all workspaces a user is a member of."""
        cursor = await self.db.execute(
            """SELECT w.* FROM workspaces w
               JOIN workspace_members wm ON w.workspace_id = wm.workspace_id
               WHERE wm.user_id = ?
               ORDER BY w.created_at DESC""",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    async def add_member(
        self, workspace_id: str, user_id: str, role: str = "contributor"
    ) -> dict[str, Any]:
        """Add a member to a workspace."""
        now = datetime.now(UTC).isoformat()
        await self.db.execute(
            """INSERT INTO workspace_members (workspace_id, user_id, role, joined_at)
               VALUES (?, ?, ?, ?)""",
            (workspace_id, user_id, role, now),
        )
        await self.db.commit()
        return {
            "workspace_id": workspace_id,
            "user_id": user_id,
            "role": role,
            "joined_at": now,
        }

    async def remove_member(self, workspace_id: str, user_id: str) -> bool:
        """Remove a member from a workspace."""
        cursor = await self.db.execute(
            "DELETE FROM workspace_members WHERE workspace_id = ? AND user_id = ?",
            (workspace_id, user_id),
        )
        await self.db.commit()
        return cursor.rowcount > 0

    async def get_members(self, workspace_id: str) -> list[dict[str, Any]]:
        """Get all members of a workspace."""
        cursor = await self.db.execute(
            "SELECT * FROM workspace_members WHERE workspace_id = ?",
            (workspace_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]

    async def get_member_role(self, workspace_id: str, user_id: str) -> str | None:
        """Get a user's role in a workspace."""
        cursor = await self.db.execute(
            "SELECT role FROM workspace_members WHERE workspace_id = ? AND user_id = ?",
            (workspace_id, user_id),
        )
        row = await cursor.fetchone()
        return row["role"] if row else None


def _load_sql(filename: str) -> str:
    """Load SQL file from the schema package."""
    schema_dir = Path(__file__).parent.parent / "schema"
    return (schema_dir / filename).read_text()


def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    """Convert an aiosqlite Row to a dict."""
    return dict(row)
