#!/usr/bin/env python3
"""One-shot backfill of node embeddings for the optional semantic_recall path.

Embeds every node that lacks a vector for the current embedding model and
stores it on ``nodes.embedding`` (+ ``embedding_model``). Idempotent and
resumable: re-running only touches nodes still missing or embedded by a
different model. Uses a LOCAL Ollama server (default bge-m3), so corpus content
never leaves the machine.

This is opt-in and never runs automatically - enable semantic_recall in the
workspace ``config.yaml`` first, then run this once to populate existing nodes.
New nodes embed at write time via the store.

Usage:
  python3 scripts/backfill_embeddings.py <workspace-dir-or-db-path>
  python3 scripts/backfill_embeddings.py ~/.kairn/default --model bge-m3
  python3 scripts/backfill_embeddings.py ./kairn.db --batch 64 --limit 100
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path


async def _run(args: argparse.Namespace) -> int:
    from kairn.config import Config
    from kairn.core.embeddings import (
        OllamaEmbedder,
        node_embedding_text,
        normalize,
        pack_vector,
    )
    from kairn.storage.sqlite_store import SQLiteStore

    target = Path(args.workspace).expanduser().resolve()
    db_path = target if target.suffix == ".db" else target / "kairn.db"
    if not db_path.exists():
        print(f"no kairn.db at {db_path}")
        return 1

    config = Config.load(db_path.parent)
    model = args.model or config.embedding_model
    host = args.host or config.embedding_host
    embedder = OllamaEmbedder(model=model, host=host)

    # No embed-at-write here (we batch directly); the store just gives us the
    # connection + migration so the embedding columns exist.
    store = SQLiteStore(db_path)
    await store.initialize()
    try:
        cursor = await store.db.execute(
            "SELECT id, name, description FROM nodes "
            "WHERE deleted_at IS NULL AND (embedding IS NULL OR embedding_model IS NOT ?)",
            (model,),
        )
        todo = await cursor.fetchall()
        if args.limit:
            todo = todo[: args.limit]
        total = len(todo)
        if total == 0:
            print(f"nothing to backfill: all nodes already embedded with {model}")
            return 0
        print(f"backfilling {total} node(s) with {model} (batch {args.batch})...")

        done = 0
        for start in range(0, total, args.batch):
            chunk = todo[start : start + args.batch]
            texts = [node_embedding_text(r[1], r[2]) for r in chunk]
            vectors = embedder(texts)
            for row, vec in zip(chunk, vectors, strict=False):
                blob = pack_vector(normalize(vec))
                await store.db.execute(
                    "UPDATE nodes SET embedding = ?, embedding_model = ? WHERE id = ?",
                    (blob, model, row[0]),
                )
            await store.db.commit()
            done += len(chunk)
            print(f"  {done}/{total}")
        print(f"done: {done} node(s) embedded with {model}")
        return 0
    finally:
        await store.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workspace", help="workspace dir containing kairn.db, or a .db path")
    ap.add_argument("--model", default=None, help="override embedding model (default: config)")
    ap.add_argument("--host", default=None, help="override Ollama host (default: config)")
    ap.add_argument("--batch", type=int, default=64, help="embed batch size")
    ap.add_argument("--limit", type=int, default=None, help="cap nodes processed (testing)")
    return asyncio.run(_run(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
