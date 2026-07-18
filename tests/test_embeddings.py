"""Tests for the optional local-embedding support (semantic_recall path).

These never touch a real Ollama server: the HTTP embedder takes an injectable
transport, and everything else is pure stdlib math. CI has no Ollama and must
still pass.
"""

from __future__ import annotations

import math

import pytest

from kairn.core.embeddings import (
    OllamaEmbedder,
    cosine,
    normalize,
    pack_vector,
    unpack_vector,
)


class TestVectorCodec:
    def test_pack_unpack_roundtrip(self):
        vec = [0.1, -0.2, 0.3333, 1.0, -1.0, 0.0]
        blob = pack_vector(vec)
        assert isinstance(blob, bytes)
        assert len(blob) == len(vec) * 4  # float32
        out = unpack_vector(blob)
        assert len(out) == len(vec)
        for a, b in zip(vec, out, strict=False):
            assert a == pytest.approx(b, abs=1e-6)

    def test_unpack_empty(self):
        assert unpack_vector(b"") == []

    def test_dim_recoverable_from_blob(self):
        vec = [0.0] * 1024
        assert len(unpack_vector(pack_vector(vec))) == 1024


class TestCosine:
    def test_identical_is_one(self):
        v = [1.0, 2.0, 3.0]
        assert cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_is_zero(self):
        assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_is_minus_one(self):
        assert cosine([1.0, 1.0], [-1.0, -1.0]) == pytest.approx(-1.0)

    def test_zero_vector_is_zero_not_nan(self):
        assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_mismatched_length_is_zero(self):
        assert cosine([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0

    def test_normalize_unit_length(self):
        n = normalize([3.0, 4.0])
        assert math.sqrt(sum(x * x for x in n)) == pytest.approx(1.0)
        assert normalize([0.0, 0.0]) == [0.0, 0.0]


class TestOllamaEmbedder:
    def test_calls_transport_and_returns_embeddings(self):
        captured = {}

        def fake_transport(texts):
            captured["texts"] = texts
            return [[0.1, 0.2], [0.3, 0.4]]

        emb = OllamaEmbedder(model="bge-m3", transport=fake_transport)
        out = emb(["hello", "world"])  # callable interface
        assert out == [[0.1, 0.2], [0.3, 0.4]]
        assert captured["texts"] == ["hello", "world"]

    def test_empty_input_returns_empty_without_transport_call(self):
        def fake_transport(texts):  # pragma: no cover - must not be called
            raise AssertionError("transport called on empty input")

        emb = OllamaEmbedder(transport=fake_transport)
        assert emb([]) == []

    def test_model_name_exposed(self):
        emb = OllamaEmbedder(model="bge-m3", transport=lambda t: [])
        assert emb.model == "bge-m3"
