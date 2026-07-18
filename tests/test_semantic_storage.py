"""Storage-layer tests for the semantic_recall embedding column (migration 006).

The embedder is a deterministic fake - no Ollama. Covers: fresh-DB columns,
legacy-DB migration (R5 safety: additive, existing rows preserved), embed-at-
write on insert + re-embed on update, read-back with the model stamp, and the
flag-OFF invariant (no embedder => embedding stays NULL).
"""

from __future__ import annotations

import aiosqlite
import pytest
import pytest_asyncio

from kairn.core.embeddings import unpack_vector
from kairn.storage.sqlite_store import SQLiteStore


def _fake_embedder(dim: int = 8):
    """Deterministic embedder: maps each text to a stable vector from its hash."""

    def embed(texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = abs(hash(t))
            out.append([((h >> (i * 3)) % 97) / 97.0 for i in range(dim)])
        return out

    return embed


@pytest_asyncio.fixture
async def store_with_embedder(tmp_path):
    store = SQLiteStore(
        tmp_path / "ws.db",
        embedder=_fake_embedder(),
        embedder_model="fake-8",
    )
    await store.initialize()
    yield store
    await store.close()


@pytest_asyncio.fixture
async def plain_store(tmp_path):
    store = SQLiteStore(tmp_path / "plain.db")
    await store.initialize()
    yield store
    await store.close()


def _node(nid: str, name: str, desc: str) -> dict:
    return {
        "id": nid,
        "namespace": "knowledge",
        "type": "pattern",
        "name": name,
        "description": desc,
        "properties": None,
        "tags": None,
        "created_by": None,
        "visibility": "workspace",
        "source_type": None,
        "source_ref": None,
        "created_at": "2026-07-18T00:00:00+00:00",
        "updated_at": None,
    }


class TestMigration006:
    @pytest.mark.asyncio
    async def test_fresh_db_has_embedding_columns(self, plain_store):
        cursor = await plain_store.db.execute("PRAGMA table_info(nodes)")
        cols = {row[1] for row in await cursor.fetchall()}
        assert "embedding" in cols
        assert "embedding_model" in cols

    @pytest.mark.asyncio
    async def test_legacy_db_migrates_and_preserves_rows(self, tmp_path):
        # Simulate a pre-006 DB: a nodes table WITHOUT the embedding columns,
        # with a real row already in it.
        db_path = tmp_path / "legacy.db"
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "CREATE TABLE nodes (id TEXT PRIMARY KEY, namespace TEXT, type TEXT, "
                "name TEXT, description TEXT, properties JSON, tags JSON, created_by TEXT, "
                "visibility TEXT, source_type TEXT, source_ref TEXT, created_at TEXT, "
                "updated_at TEXT, deleted_at TEXT)"
            )
            await db.execute(
                "INSERT INTO nodes (id, type, name, created_at) "
                "VALUES ('n1', 'pattern', 'Old node', 't')"
            )
            await db.commit()

        store = SQLiteStore(db_path)
        await store.initialize()  # runs _migrate_schema
        try:
            cursor = await store.db.execute("PRAGMA table_info(nodes)")
            cols = {row[1] for row in await cursor.fetchall()}
            assert "embedding" in cols and "embedding_model" in cols
            # existing row survives, embedding NULL
            row = await store.get_node("n1")
            assert row is not None and row["name"] == "Old node"
            assert row.get("embedding") is None
        finally:
            await store.close()


class TestEmbedAtWrite:
    @pytest.mark.asyncio
    async def test_insert_embeds_and_stamps_model(self, store_with_embedder):
        await store_with_embedder.insert_node(_node("a1", "Kafka partitions", "consumer groups"))
        row = await store_with_embedder.get_node("a1")
        assert row["embedding"] is not None
        assert row["embedding_model"] == "fake-8"
        assert len(unpack_vector(row["embedding"])) == 8

    @pytest.mark.asyncio
    async def test_update_reembeds_on_description_change(self, store_with_embedder):
        await store_with_embedder.insert_node(_node("a2", "Title", "original text"))
        before = (await store_with_embedder.get_node("a2"))["embedding"]
        await store_with_embedder.update_node("a2", {"description": "totally different text now"})
        after = (await store_with_embedder.get_node("a2"))["embedding"]
        assert after is not None and after != before

    @pytest.mark.asyncio
    async def test_query_with_embeddings_returns_blob(self, store_with_embedder):
        await store_with_embedder.insert_node(_node("a3", "Postgres concurrency", "writers"))
        rows = await store_with_embedder.query_nodes_with_embeddings(
            text="postgres concurrency", limit=10
        )
        assert rows
        hit = next(r for r in rows if r["id"] == "a3")
        assert hit["embedding"] is not None
        assert hit["embedding_model"] == "fake-8"


class TestFlagOffInvariant:
    @pytest.mark.asyncio
    async def test_no_embedder_leaves_embedding_null(self, plain_store):
        await plain_store.insert_node(_node("b1", "No embed", "should stay null"))
        row = await plain_store.get_node("b1")
        assert row["embedding"] is None
        assert row["embedding_model"] is None
