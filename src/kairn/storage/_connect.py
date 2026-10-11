"""Database connection helper."""

from __future__ import annotations

import logging
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)


async def open_connection(db_path: Path) -> aiosqlite.Connection:
    """Wait for aiosqlite's worker thread when opening the database fails.

    aiosqlite's connect does not wait for its worker thread to stop when the
    open fails. This waits for it while the caller's event loop is still running.
    The wait blocks so a second cancellation cannot interrupt it.
    _thread is aiosqlite's internal attribute in 0.22 and later; in 0.21,
    the connection itself is the thread.
    """
    conn = aiosqlite.connect(str(db_path))
    try:
        return await conn
    except BaseException:
        worker = getattr(conn, "_thread", conn)
        if hasattr(worker, "join"):
            worker.join(5)
            if worker.is_alive():
                logger.warning(
                    "aiosqlite worker thread is still running after a failed open"
                )
        raise
