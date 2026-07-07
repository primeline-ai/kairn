"""Zero-LLM bulk importers - WOW-9 (kairn import).

Backfill a user's real history (git commits, Claude Code transcripts) into
Kairn at $0 marginal cost: no LLM calls, no network calls, deterministic
rule-based extraction only. Each importer writes into its own dedicated
namespace so a bad import is always reversible via a namespace-scoped
delete (see storage.base.StorageBackend.delete_experiences_by_namespace).
"""

from __future__ import annotations
