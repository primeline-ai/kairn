"""Database connection helper."""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite


async def open_connection(db_path: Path) -> aiosqlite.Connection:
    """Wait for aiosqlite's worker thread when opening the database fails.

    aiosqlite's connect does not wait for its worker thread to stop when the
    open fails. The thread could finish after the caller's event loop closed
    and print "Event loop is closed". This waits while the loop is still running.
    We use aiosqlite's internal _thread attribute because there is no public
    way to wait for the stop. In 0.20, the connection itself is the thread.
    """
    conn = aiosqlite.connect(str(db_path))
    try:
        return await conn
    except BaseException:
        worker = getattr(conn, "_thread", conn)
        if hasattr(worker, "join"):
            await asyncio.to_thread(worker.join, 10)
        raise
