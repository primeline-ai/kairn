"""A store whose database file cannot be opened must not leave a worker thread behind.

aiosqlite starts a worker thread for every connection. When opening the file
fails, its `_connect` calls `stop()` without waiting for it (aiosqlite 0.22.1,
and upstream main at the time of writing), so the thread can finish after the
caller's event loop has closed and print "RuntimeError: Event loop is closed"
to stderr. `kairn serve --init` on a read-only folder hit that on CI now and
then, breaking its own one-line error. `kairn.storage._connect.open_connection`
therefore waits for the worker thread to end before it re-raises.
"""

from __future__ import annotations

import os
import sqlite3
import stat
import sys
import threading
from pathlib import Path

import aiosqlite
import pytest

from kairn.storage import _connect
from kairn.storage.metadata_store import MetadataStore
from kairn.storage.sqlite_store import SQLiteStore

STORES = [
    pytest.param(SQLiteStore, id="SQLiteStore"),
    pytest.param(MetadataStore, id="MetadataStore"),
]

needs_posix_permissions = pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs POSIX permissions and a non-root user",
)


def _aiosqlite_worker_idents() -> set[int]:
    """Idents of live aiosqlite worker threads, in both shapes aiosqlite has used.

    Up to 0.21 the Connection object is itself the Thread (named "Thread-N");
    from 0.22 it starts a separate thread whose target is
    `_connection_worker_thread`.
    """
    return {
        t.ident
        for t in threading.enumerate()
        if isinstance(t, aiosqlite.Connection) or "_connection_worker_thread" in t.name
    }


@pytest.mark.asyncio
async def test_the_thread_check_sees_a_live_connection(tmp_path: Path) -> None:
    """Positive control: without it, a filter that matches nothing makes the test below vacuous."""
    baseline = _aiosqlite_worker_idents()
    store = SQLiteStore(tmp_path / "live.db")
    await store.initialize()
    try:
        live = _aiosqlite_worker_idents() - baseline
        assert live, "the thread check cannot see a live connection"
    finally:
        await store.close()


@needs_posix_permissions
@pytest.mark.asyncio
@pytest.mark.parametrize("store_cls", STORES)
async def test_failed_open_leaves_no_worker_thread(tmp_path: Path, store_cls) -> None:
    folder = tmp_path / "readonly"
    folder.mkdir()
    folder.chmod(stat.S_IRUSR | stat.S_IXUSR)  # the db file cannot be created here
    baseline = _aiosqlite_worker_idents()
    store = store_cls(folder / "kairn.db")
    try:
        with pytest.raises(sqlite3.OperationalError):
            await store.initialize()
        # No grace period: the thread must already be gone when initialize raises,
        # because the caller may close its event loop right after.
        leftover = _aiosqlite_worker_idents() - baseline
    finally:
        folder.chmod(stat.S_IRWXU)
        await store.close()
    assert not leftover, "an aiosqlite worker thread outlived the failed open"


@pytest.mark.asyncio
@pytest.mark.parametrize("store_cls", STORES)
async def test_openable_database_opens_once_through_aiosqlite(
    tmp_path: Path, monkeypatch, store_cls
) -> None:
    calls: list[str] = []
    real_connect = _connect.aiosqlite.connect

    def counting(path, *args, **kwargs):
        calls.append(str(path))
        return real_connect(path, *args, **kwargs)

    monkeypatch.setattr(_connect.aiosqlite, "connect", counting)
    store = store_cls(tmp_path / "ws" / "kairn.db")
    try:
        await store.initialize()
    finally:
        await store.close()
    assert len(calls) == 1, calls
    assert (tmp_path / "ws" / "kairn.db").is_file()
