"""Kairn diagnostic registry.

Read-only health checks for Kairn workspaces. Exposed via the
`kn_doctor` MCP tool (Phase 4 of the judgment-envelope plan) and the
`kairn doctor` CLI subcommand. Each check returns a structured envelope:

    {
        "check_id": "snake_case_id",
        "severity": "info | warn | error",
        "status": "ok | warn | fail",
        "evidence": "one-line factual observation",
        "safe_next_step": "what an operator/agent should do",
    }

The envelope shape is stable across CLI `--json` output and MCP
responses (envelope parity is a Phase 4 gate criterion). Checks are
strictly read-only - they MUST NOT mutate workspace state, even
transiently. Mutating checks belong in a separate write-allowed
diagnostic surface.

Adding a new check: implement an async function with the signature
`async def name(store: StorageBackend) -> dict` and register it in
`CHECK_REGISTRY` in `registry.py`. The check_id must equal the
function name.
"""

from kairn.diagnostic.checks import (
    check_fts_index_health,
    check_lock_mode,
    check_namespace_distribution,
    check_orphan_edges,
    check_promoted_experience_consistency,
)
from kairn.diagnostic.registry import CHECK_REGISTRY, run_checks

__all__ = [
    "CHECK_REGISTRY",
    "check_fts_index_health",
    "check_lock_mode",
    "check_namespace_distribution",
    "check_orphan_edges",
    "check_promoted_experience_consistency",
    "run_checks",
]
