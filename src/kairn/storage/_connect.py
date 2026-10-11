"""Database connection helper."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import aiosqlite


async def open_connection(db_path: Path) -> aiosqlite.Connection:
    """Probe the file first so a failed aiosqlite open cannot outlive the event loop."""

    def _probe(path: Path) -> None:
        sqlite3.connect(str(path)).close()

    await asyncio.to_thread(_probe, db_path)
    return await aiosqlite.connect(str(db_path))
