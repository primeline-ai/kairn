"""Experience model with decay mechanics."""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field

VALID_TYPES = {"solution", "pattern", "decision", "workaround", "gotcha"}
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
    decay_rate: float
    tags: list[str] | None = None
    properties: dict | None = None
    created_by: str | None = None
    access_count: int = 0
    promoted_to_node_id: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_accessed: str | None = None

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

    def to_storage(self) -> dict:
        return self.model_dump()

    def to_response(self, *, detail: str = "summary") -> dict:
        data = {
            "_v": "1.0",
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "confidence": self.confidence,
            "relevance": round(self.relevance(), 3),
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
                }
            )
        return data
