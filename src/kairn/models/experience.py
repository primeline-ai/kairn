"""Experience model with decay mechanics."""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from kairn.relevance import RELEVANCE_KIND_MATCH_RECENCY, RELEVANCE_KIND_RECENCY

VALID_TYPES = {"solution", "pattern", "decision", "workaround", "gotcha", "preference"}
VALID_CONFIDENCES = {"high", "medium", "low"}


class Experience(BaseModel):
    """A temporal, decaying experience with promotion capability."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    namespace: str = "knowledge"
    type: str
    content: str
    context: str | None = None
    confidence: str = "high"
    score: float = 1.0
    # Match-aware relevance for THIS recall, set by ExperienceEngine.search()
    # when a text query was given. Never persisted - `relevance()` remains pure
    # time-decay, so decay and match strength stay orthogonal in the model and
    # only the RECALL composes them.
    recall_relevance: float | None = None
    decay_rate: float
    tags: list[str] | None = None
    properties: dict | None = None
    created_by: str | None = None
    access_count: int = 0
    promoted_to_node_id: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_accessed: str | None = None
    # Bi-temporal valid-time window (when the fact was TRUE in the world).
    # ORTHOGONAL to created_at (transaction-time) and to decay: valid_from /
    # valid_to NEVER feed relevance(). Both nullable; NULL = no validity bound.
    valid_from: str | None = None
    valid_to: str | None = None
    # Rule-based cross-session entity grouping key (no LLM, no embeddings).
    # Populated at save() time; used by the bi-temporal recall path to
    # diversify/aggregate experiences about the same subject across sessions.
    entity_key: str | None = None

    def relevance(self, *, at: datetime | None = None) -> float:
        """Calculate current relevance using exponential decay."""
        at = at or datetime.now(UTC)
        created = datetime.fromisoformat(self.created_at)
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        days = (at - created).total_seconds() / 86400
        return self.score * math.exp(-self.decay_rate * days)

    def is_expired(self, threshold: float = 0.01) -> bool:
        return self.relevance() < threshold

    def reported_relevance(
        self, *, at: datetime | None = None, decimals: int = 4
    ) -> float:
        """The relevance a CALLER sees. THE ONLY implementation of that rule.

        Prefers the match-aware value a recall computed and attached as
        recall_relevance; falls back to pure time-decay for a browse query
        and for any object that never went through a recall. So this is
        additive: on the no-text path it is exactly what the old
        round(self.relevance(), decimals) returned.

        It lives on the MODEL rather than in core.intelligence because the
        model is the one module every reporting surface can already import.
        The previous arrangement had the rule in intelligence and a second,
        older copy here in to_response - so the wire surfaces were fixed
        and the model method kept emitting pure decay, which is the N-1-of-N
        shape (Kairn d5272a91) with the model as the missing site.
        test_f3_census_no_decay_scale_report_left_anywhere now sweeps the
        whole package and exempts this definition BY NAME, so a third copy
        cannot appear quietly.
        """
        # `recall_relevance` is a DECLARED field with a None default, so the
        # rule keys on the value, not on the attribute's presence.
        if self.recall_relevance is not None:
            return self.recall_relevance
        return round(self.relevance(at=at), decimals)

    def reported_relevance_kind(self) -> str:
        """WHICH quantity `reported_relevance` just returned.

        Same input, same branch, one method away - so the label cannot drift
        away from the number it describes. Reporting a match-aware composite
        under the RECENCY label is the mislabelling `kairn.relevance` was
        written to prevent, and after this branch made the experience path
        match-aware that label became wrong on every text query.
        """
        if self.recall_relevance is not None:
            return RELEVANCE_KIND_MATCH_RECENCY
        return RELEVANCE_KIND_RECENCY

    def to_storage(self) -> dict:
        return self.model_dump()

    def to_response(self, *, detail: str = "summary") -> dict:
        # Note: the validity window (valid_from/valid_to) and entity_key are
        # included only in the "full" detail branch below, not the summary
        # branch, to keep summary responses token-lean. Callers that need
        # validity/entity data must request detail="full".
        data = {
            "_v": "1.0",
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "confidence": self.confidence,
            # 3 decimals is this method's own wire contract and is kept;
            # what changed is WHICH quantity gets rounded - and the label
            # moves with it.
            "relevance": self.reported_relevance(decimals=3),
            "relevance_kind": self.reported_relevance_kind(),
        }
        if detail != "summary":
            data.update(
                {
                    "namespace": self.namespace,
                    "context": self.context,
                    "score": self.score,
                    "decay_rate": self.decay_rate,
                    "tags": self.tags,
                    "access_count": self.access_count,
                    "promoted_to_node_id": self.promoted_to_node_id,
                    "created_at": self.created_at,
                    "last_accessed": self.last_accessed,
                    "valid_from": self.valid_from,
                    "valid_to": self.valid_to,
                }
            )
        return data
