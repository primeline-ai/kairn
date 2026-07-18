"""Local embedding support for the optional ``semantic_recall`` path.

Deliberately dependency-light: the standard library only (``urllib``,
``struct``, ``math``). No numpy, no sentence-transformers - turning the flag on
adds no new *pip* dependency, only a running local Ollama. Embeddings are
produced by a LOCAL Ollama server (``localhost`` by default), so corpus content
never leaves the machine and engram's air-gap posture is preserved even with
semantic recall enabled.

This module is imported ONLY on the flag-ON path (lazily, from the wiring in
``cli``/``server`` and from ``IntelligenceLayer`` when an embedder is present).
With ``semantic_recall`` OFF nothing here is loaded, so the default recall path
is byte-identical to the keyword-only product.

Vectors are stored as a packed little-endian float32 BLOB on ``nodes.embedding``
with the model name in ``nodes.embedding_model`` (see migration 006). Storing
the model name lets recall compare only vectors produced by the SAME model as
the live query embedding - a model swap invalidates old vectors instead of
silently comparing incompatible spaces.
"""

from __future__ import annotations

import json
import math
import struct
import urllib.request
from collections.abc import Callable

DEFAULT_MODEL = "bge-m3"
DEFAULT_HOST = "http://localhost:11434"

# A node is embedded from its name + the head of its description. Embedding the
# salient summary rather than the full body is standard retrieval practice
# (a long heterogeneous body dilutes the vector) and keeps embed cost bounded.
# The cap is a round, model-agnostic value, not tuned to any benchmark.
EMBED_TEXT_CHARS = 512


def node_embedding_text(name: str | None, description: str | None) -> str:
    """The text embedded for a node: name + description head, capped."""
    return f"{name or ''} {description or ''}".strip()[:EMBED_TEXT_CHARS]

# An embedder maps a batch of texts to a batch of vectors. Prod uses
# ``OllamaEmbedder``; tests inject a deterministic callable so CI needs no
# Ollama. ``None`` everywhere means the flag is OFF and no embedding happens.
Embedder = Callable[[list[str]], list[list[float]]]


def pack_vector(vec: list[float]) -> bytes:
    """Pack a float vector into a little-endian float32 BLOB."""
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_vector(blob: bytes) -> list[float]:
    """Unpack a float32 BLOB back into a list of floats (dim = len // 4)."""
    if not blob:
        return []
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def normalize(vec: list[float]) -> list[float]:
    """Return the L2-normalized vector; a zero vector is returned unchanged."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 for empty, zero, or
    mismatched-length vectors (never raises, never NaN)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def embedder_from_config(config: object) -> tuple[Embedder | None, str | None]:
    """Build the (embedder, model) pair for the wiring layer.

    Returns ``(None, None)`` when ``semantic_recall`` is off so the store and
    intelligence layer stay on the keyword path. Duck-typed on ``config`` to
    avoid importing the Config dataclass (no import cycle).
    """
    if not getattr(config, "semantic_recall", False):
        return None, None
    model = getattr(config, "embedding_model", DEFAULT_MODEL)
    host = getattr(config, "embedding_host", DEFAULT_HOST)
    return OllamaEmbedder(model=model, host=host), model


class OllamaEmbedder:
    """Callable that batches texts to a local Ollama ``/api/embed`` endpoint.

    ``transport`` is an injection seam for tests: a callable taking the batch
    and returning the vectors, bypassing the HTTP call. In production it is
    ``None`` and the real endpoint is used.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 30.0,
        transport: Embedder | None = None,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._transport = transport

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._transport is not None:
            return self._transport(texts)
        body = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/embed",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        return data["embeddings"]
