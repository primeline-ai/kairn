"""A store whose database file cannot be opened must not leave a worker thread behind.

aiosqlite starts a worker thread for every connection. When opening the file
fails, its `_connect` calls `stop()` without waiting for it (aiosqlite 0.22.1,
and upstream main at the time of writing), so the thread can finish after the
caller's event loop has closed and print "RuntimeError: Event loop is closed"
to stderr. `kairn serve --init` on a read-only folder hit that on CI now and
then, breaking its own one-line error. The stores therefore try the open with
plain sqlite3 first and only start aiosqlite once the file can be opened.
"""

from __future__ import annotations

import os
import sqlite3
import stat
import sys
from pathlib import Path

import pytest

from kairn.storage import metadata_store, sqlite_store
from kairn.storage.metadata_store import MetadataStore
from kairn.storage.sqlite_store import SQLiteStore

STORES = [
    pytest.param(SQLiteStore, sqlite_store, id="SQLiteStore"),
    pytest.param(MetadataStore, metadata_store, id="MetadataStore"),
]

needs_posix_permissions = pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs POSIX permissions and a non-root user",
)


def _counting_connect(module, monkeypatch) -> list[str]:
    calls: list[str] = []
    real_connect = module.aiosqlite.connect

    def counting(path, *args, **kwargs):
        calls.append(str(path))
        return real_connect(path, *args, **kwargs)

    monkeypatch.setattr(module.aiosqlite, "connect", counting)
    return calls


@needs_posix_permissions
@pytest.mark.asyncio
@pytest.mark.parametrize(("store_cls", "module"), STORES)
async def test_unopenable_database_fails_before_aiosqlite_starts(
    tmp_path: Path, monkeypatch, store_cls, module
) -> None:
    folder = tmp_path / "readonly"
    folder.mkdir()
    folder.chmod(stat.S_IRUSR | stat.S_IXUSR)  # the db file cannot be created here
    calls = _counting_connect(module, monkeypatch)
    store = store_cls(folder / "kairn.db")
    try:
        with pytest.raises(sqlite3.OperationalError):
            await store.initialize()
    finally:
        folder.chmod(stat.S_IRWXU)
        await store.close()
    assert calls == [], "aiosqlite started a worker thread for a file that cannot be opened"


@pytest.mark.asyncio
@pytest.mark.parametrize(("store_cls", "module"), STORES)
async def test_openable_database_still_opens_through_aiosqlite(
    tmp_path: Path, monkeypatch, store_cls, module
) -> None:
    calls = _counting_connect(module, monkeypatch)
    store = store_cls(tmp_path / "ws" / "kairn.db")
    try:
        await store.initialize()
    finally:
        await store.close()
    assert len(calls) == 1, calls
    assert (tmp_path / "ws" / "kairn.db").is_file()
