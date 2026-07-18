"""Recall-path tests for the semantic_recall flag (cosine rerank + abstain).

A substring-mapped fake embedder gives deterministic vectors so the rerank and
the cosine abstain floor are testable without Ollama. Both nodes share a
lexical token so both are FTS candidates; the embedding decides the order and
whether the floor abstains.
"""

from __future__ import annotations

import pytest

from kairn.core.experience import ExperienceEngine
from kairn.core.graph import GraphEngine
from kairn.core.ideas import IdeaEngine
from kairn.core.intelligence import IntelligenceLayer
from kairn.core.memory import ProjectMemory
from kairn.core.router import ContextRouter
from kairn.events.bus import EventBus
from kairn.storage.sqlite_store import SQLiteStore


def _mapped_embedder(mapping: dict[str, list[float]], default: list[float]):
    """First substring key found in the (lowercased) text wins its vector."""

    def embed(texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            tl = t.lower()
            vec = default
            for key, v in mapping.items():
                if key in tl:
                    vec = v
                    break
            out.append(list(vec))
        return out

    return embed


# A: "concurrent writers" -> aligned with a concurrent-writers query.
# B: "backup cron" -> orthogonal to A. "postgres" is the shared FTS token but
# is deliberately NOT mapped, so a bare-"postgres" alien query falls to the
# orthogonal default and the cosine floor abstains on it.
_EMBEDDER = _mapped_embedder(
    {
        "concurrent writers": [1.0, 0.0, 0.0],
        "backup cron": [0.0, 1.0, 0.0],
        # A partial-overlap vector: cosine ~0.707 with a [1,0,0] query - above
        # the 0.5 floor but below a 0.9 min_relevance, so it exercises the
        # min_relevance-tightens-the-node-gate path.
        "partialvec": [0.7, 0.7, 0.0],
    },
    default=[0.0, 0.0, 1.0],
)


async def _make_intel(tmp_path, *, semantic_recall: bool, floor: float = 0.5):
    store = SQLiteStore(
        tmp_path / "sem.db",
        embedder=_EMBEDDER if semantic_recall else None,
        embedder_model="fake-3" if semantic_recall else None,
    )
    await store.initialize()
    bus = EventBus()
    graph = GraphEngine(store, bus)
    intel = IntelligenceLayer(
        store=store,
        event_bus=bus,
        graph=graph,
        router=ContextRouter(store, bus),
        memory=ProjectMemory(store, bus),
        experience=ExperienceEngine(store, bus),
        ideas=IdeaEngine(store, bus),
        embedder=_EMBEDDER if semantic_recall else None,
        embedder_model="fake-3" if semantic_recall else None,
        semantic_recall=semantic_recall,
        semantic_floor=floor,
    )
    return store, intel


async def _seed(intel):
    # Both contain "postgres" (shared FTS token) so both are candidates.
    await intel.learn(
        content="Postgres chosen for concurrent writers in the analytics service",
        type="decision",
        confidence="high",
    )
    await intel.learn(
        content="Postgres backup cron mentions nothing else of note",
        type="pattern",
        confidence="high",
    )


class TestSemanticRerank:
    @pytest.mark.asyncio
    async def test_cosine_rerank_orders_by_meaning(self, tmp_path):
        store, intel = await _make_intel(tmp_path, semantic_recall=True, floor=0.25)
        try:
            await _seed(intel)
            results = await intel.recall(topic="concurrent writers")
            nodes = [r for r in results if r["source"] == "node"]
            assert nodes, "expected node results"
            # The concurrent-writers node ranks first; relevance is the cosine.
            assert "concurrent writers" in (nodes[0]["description"] or "")
            assert nodes[0]["relevance"] == pytest.approx(1.0, abs=1e-4)
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_cosine_floor_abstains_on_semantically_alien_query(self, tmp_path):
        # Query lexically matches "postgres" (FTS candidate) but its embedding
        # is orthogonal to both nodes -> every candidate is below the floor.
        store, intel = await _make_intel(tmp_path, semantic_recall=True, floor=0.5)
        try:
            await _seed(intel)
            results = await intel.recall(topic="postgres unrelatable gibberish")
            nodes = [r for r in results if r["source"] == "node"]
            assert nodes == []  # abstained: no node cleared the cosine floor
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_weak_match_below_floor_is_dropped(self, tmp_path):
        store, intel = await _make_intel(tmp_path, semantic_recall=True, floor=0.5)
        try:
            await _seed(intel)
            results = await intel.recall(topic="concurrent writers")
            nodes = [r for r in results if r["source"] == "node"]
            # Only the aligned node survives; the orthogonal backup-cron node
            # (cosine 0) is filtered by the 0.5 floor.
            descs = " ".join(n["description"] or "" for n in nodes)
            assert "concurrent writers" in descs
            assert "backup cron" not in descs
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_min_relevance_tightens_the_semantic_node_gate(self, tmp_path):
        """A caller's min_relevance must still tighten node results on the
        semantic path (it can only raise the effective floor), instead of being
        silently ignored the way it was before."""
        store, intel = await _make_intel(tmp_path, semantic_recall=True, floor=0.5)
        try:
            # Exact-match node (cosine 1.0) + a partial node (cosine ~0.707).
            await intel.learn(
                content="concurrent writers analytics service",
                type="decision",
                confidence="high",
            )
            await intel.learn(
                content="concurrent partialvec adjacent note",
                type="pattern",
                confidence="high",
            )
            # min_relevance below both cosines: both survive the 0.5 floor.
            loose = await intel.recall(topic="concurrent writers", min_relevance=0.0)
            loose_nodes = [r for r in loose if r["source"] == "node"]
            assert any("partialvec" in (n["description"] or "") for n in loose_nodes)
            # min_relevance 0.9 (> 0.707) drops the partial node, keeps the exact one.
            strict = await intel.recall(topic="concurrent writers", min_relevance=0.9)
            strict_nodes = [r for r in strict if r["source"] == "node"]
            assert strict_nodes, "the 1.0-cosine node must survive"
            assert all("partialvec" not in (n["description"] or "") for n in strict_nodes)
        finally:
            await store.close()
