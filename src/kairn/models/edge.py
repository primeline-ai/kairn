"""Edge model for graph relationships."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class Edge(BaseModel):
    """A typed, weighted edge between two nodes."""

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
