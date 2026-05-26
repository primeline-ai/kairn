"""Edge model for graph relationships.

Canonical relationship vocabulary (5-verb judgment system):

| Verb            | When to use                                                            |
|-----------------|------------------------------------------------------------------------|
| conflicts_with  | Two nodes assert contradictory facts; one must lose                    |
| supersedes      | Source replaces target; target is now historical                       |
| compatible      | Two nodes agree / coexist / one validates or implements the other      |
| scoped          | One node is a scoped specialization, refinement, or sub-part of other  |
| related         | Generic association with no stronger claim (default for unclassified)  |

This vocabulary is enforced strictly only on edges created via the
`kn_judge` MCP tool (Phase 3). Legacy edges created via `kn_connect`
(GraphEngine.connect) pass through `validate_relation` in lax mode:
a one-line `logger.warning` is emitted on unknown verbs but the edge
persists. `GraphEngine._auto_link` bypasses `connect()` entirely and
writes via `store.insert_edge()` directly, so its `auto_related` edges
are not validated at all (this is the hot path - 98.7% of historical
edges - and was deliberately kept zero-overhead). Combined, this
preserves backward compatibility with the historical edge population
(6086 `auto_related` system edges + 83 manual taxonomy edges across
35 distinct types as of 2026-05-26) while pulling new judgment edges
toward the canonical 5-verb set.

Reference: `.claude/plans/2026-05-26-kairn-judgment-envelope-and-doctor.md`
Phase 0 edge-type audit: `_autonomous/benchmarks/kairn-edge-type-audit.md`
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Canonical 5-verb judgment vocabulary. Strict-mode callers (kn_judge)
# must pick one of these; lax-mode callers (kn_connect, _auto_link) may
# pass any string but unknown verbs get a one-line warning.
RELATION_VERBS: frozenset[str] = frozenset(
    {"conflicts_with", "supersedes", "compatible", "scoped", "related"}
)


def validate_relation(verb: str, *, strict: bool = False) -> bool:
    """Validate that a relation verb is in the canonical 5-verb vocabulary.

    Two modes:

    - `strict=False` (lax, default): unknown verbs return False and emit a
      one-line `logger.warning`. Caller proceeds with the legacy verb.
      Used by `GraphEngine.connect()` to preserve backward compatibility
      with the 6169-edge historical population.
    - `strict=True`: unknown verbs raise `ValueError`. Used by `kn_judge`
      to force new judgment edges onto the canonical vocabulary.

    Empty / whitespace-only strings are always rejected (raise ValueError
    even in lax mode) - an unclassified empty verb is a caller bug, not
    a legacy artifact.

    Args:
        verb: Relationship verb to validate.
        strict: If True, raise on unknown verbs. If False, warn and return False.

    Returns:
        True if verb is in `RELATION_VERBS`, False if lax-mode passthrough.

    Raises:
        ValueError: If verb is empty/whitespace, or strict=True and verb is unknown.
    """
    if not isinstance(verb, str) or not verb.strip():
        raise ValueError("Relation verb must be a non-empty string")

    if verb in RELATION_VERBS:
        return True

    if strict:
        raise ValueError(
            f"Invalid relation verb: {verb!r}. "
            f"Must be one of {sorted(RELATION_VERBS)} in strict mode."
        )

    logger.warning(
        "Legacy relation verb %r is not in the canonical 5-verb vocabulary; "
        "edge will persist. Use kn_judge for new judgment edges.",
        verb,
    )
    return False


class Edge(BaseModel):
    """A typed, weighted edge between two nodes.

    The `type` field accepts any string for backward compatibility, but
    new judgment edges are expected to use one of the 5 canonical verbs
    in `RELATION_VERBS`. See module docstring for the vocabulary.
    """

    source_id: str
    target_id: str
    type: str
    weight: float = 1.0
    properties: dict | None = None
    created_by: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_storage(self) -> dict:
        return self.model_dump()

    def to_response(self) -> dict:
        return {
            "_v": "1.0",
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type,
            "weight": self.weight,
            "created_by": self.created_by,
        }
