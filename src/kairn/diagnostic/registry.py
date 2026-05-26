"""Diagnostic check registry + run_checks orchestrator.

Pattern adapted from Gentleman-Programming/engram's `internal/diagnostic/`
(MIT, archived at Evolving's `_archive/repos/2026-05-26-engram-gentleman/`):
each check is a stable string ID -> async callable, run sequentially
because the workload is tiny (Kairn workspace is single-process, single-file).
Parallelism would not help and would complicate envelope determinism.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from kairn.diagnostic.checks import (
    check_fts_index_health,
    check_lock_mode,
    check_namespace_distribution,
    check_orphan_edges,
    check_promoted_experience_consistency,
)
from kairn.storage.base import StorageBackend

CheckFn = Callable[[StorageBackend], Awaitable[dict[str, Any]]]

CHECK_REGISTRY: dict[str, CheckFn] = {
    "check_lock_mode": check_lock_mode,
    "check_fts_index_health": check_fts_index_health,
    "check_promoted_experience_consistency": check_promoted_experience_consistency,
    "check_namespace_distribution": check_namespace_distribution,
    "check_orphan_edges": check_orphan_edges,
}


async def run_checks(
    store: StorageBackend,
    *,
    only: str | None = None,
) -> dict[str, Any]:
    """Run registered checks against a workspace store.

    Args:
        store: Initialized SQLiteStore.
        only: If provided, run only the named check. Otherwise run all.

    Returns:
        Envelope dict:
            {
                "_v": "1.0",
                "summary": {"ok": int, "warn": int, "fail": int, "error": int},
                "checks": [<envelope>, ...],
            }

    Raises:
        ValueError: If `only` is provided but not a known check_id.
    """
    if only is not None and only not in CHECK_REGISTRY:
        raise ValueError(
            f"Unknown check_id: {only!r}. "
            f"Known: {sorted(CHECK_REGISTRY)}"
        )

    selected = (
        [(only, CHECK_REGISTRY[only])]
        if only
        else list(CHECK_REGISTRY.items())
    )

    results: list[dict[str, Any]] = []
    summary = {"ok": 0, "warn": 0, "fail": 0, "error": 0}
    for check_id, fn in selected:
        try:
            envelope = await fn(store)
        except Exception as exc:  # noqa: BLE001
            envelope = {
                "check_id": check_id,
                "severity": "error",
                "status": "error",
                "evidence": f"{type(exc).__name__}: {exc}"[:200],
                "safe_next_step": (
                    "This check itself raised. Inspect the workspace "
                    "manually or open an issue with the evidence above."
                ),
            }
        results.append(envelope)
        status = envelope.get("status", "error")
        if status not in summary:
            status = "error"
        summary[status] += 1

    return {
        "_v": "1.0",
        "summary": summary,
        "checks": results,
    }
