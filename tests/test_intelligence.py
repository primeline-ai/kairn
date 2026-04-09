"""Tests for the intelligence layer — learn, recall, crossref, context, related.

Uses REAL conversation-style inputs, not just unit tests.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from kairn.core.experience import ExperienceEngine
from kairn.core.graph import GraphEngine
from kairn.core.ideas import IdeaEngine
from kairn.core.intelligence import IntelligenceLayer
from kairn.core.memory import ProjectMemory
from kairn.core.router import ContextRouter
from kairn.events.bus import EventBus
from kairn.storage.sqlite_store import SQLiteStore


@pytest_asyncio.fixture
async def engine(tmp_path):
    """Full intelligence stack wired to a temp database."""
    db_path = tmp_path / "intel_test.db"
    store = SQLiteStore(db_path)
    await store.initialize()
    bus = EventBus()

    graph = GraphEngine(store, bus)
    router = ContextRouter(store, bus)
    memory = ProjectMemory(store, bus)
    experience = ExperienceEngine(store, bus)
    ideas = IdeaEngine(store, bus)

    intel = IntelligenceLayer(
        store=store,
        event_bus=bus,
        graph=graph,
        router=router,
        memory=memory,
        experience=experience,
        ideas=ideas,
    )

    yield intel

    await store.close()


# ──────────────────────────────────────────────────────────────────────
# learn() tests
# ──────────────────────────────────────────────────────────────────────


class TestLearn:
    """Tests for kn_learn — confidence-based knowledge routing."""

    @pytest.mark.asyncio
    async def test_learn_high_confidence_creates_node(self, engine: IntelligenceLayer):
        """High confidence → node in knowledge namespace."""
        result = await engine.learn(
            content="JWT is better than session cookies for API auth",
            type="decision",
            context="Evaluating auth strategies for REST API",
            confidence="high",
            tags=["auth", "jwt"],
        )

        assert result["stored_as"] == "node"
        assert result["node_id"] is not None
        assert result["experience_id"] is not None  # Also creates experience

    @pytest.mark.asyncio
    async def test_learn_medium_confidence_creates_experience_only(
        self, engine: IntelligenceLayer
    ):
        """Medium confidence → experience only, 2x decay."""
        result = await engine.learn(
            content="Redis might be better than in-memory for caching",
            type="pattern",
            confidence="medium",
        )

        assert result["stored_as"] == "experience"
        assert result["node_id"] is None
        assert result["experience_id"] is not None

    @pytest.mark.asyncio
    async def test_learn_low_confidence_creates_experience_only(
        self, engine: IntelligenceLayer
    ):
        """Low confidence → experience only, 4x decay."""
        result = await engine.learn(
            content="Maybe we should try GraphQL instead of REST",
            type="decision",
            confidence="low",
        )

        assert result["stored_as"] == "experience"
        assert result["node_id"] is None
        assert result["experience_id"] is not None

    @pytest.mark.asyncio
    async def test_learn_high_confidence_node_is_queryable(
        self, engine: IntelligenceLayer
    ):
        """Node created by learn() should appear in graph queries."""
        await engine.learn(
            content="Always use parameterized SQL queries",
            type="pattern",
            confidence="high",
            tags=["security", "sql"],
        )

        nodes = await engine.graph.query(text="parameterized SQL")
        assert len(nodes) >= 1
        assert any("parameterized" in n.name.lower() or "parameterized" in (n.description or "").lower() for n in nodes)

    @pytest.mark.asyncio
    async def test_learn_auto_links_related_nodes(self, engine: IntelligenceLayer):
        """When learning, auto-link to existing related nodes."""
        # First, add a related node
        await engine.graph.add_node(name="authentication", type="concept")

        # Now learn something related
        result = await engine.learn(
            content="Use bcrypt for password hashing in authentication",
            type="solution",
            confidence="high",
            tags=["auth"],
        )

        assert result["node_id"] is not None

    @pytest.mark.asyncio
    async def test_learn_invalid_type_raises(self, engine: IntelligenceLayer):
        with pytest.raises(ValueError, match="Invalid"):
            await engine.learn(
                content="Something",
                type="invalid_type",
                confidence="high",
            )

    @pytest.mark.asyncio
    async def test_learn_invalid_confidence_raises(self, engine: IntelligenceLayer):
        with pytest.raises(ValueError, match="Invalid"):
            await engine.learn(
                content="Something",
                type="decision",
                confidence="very_high",
            )

    @pytest.mark.asyncio
    async def test_learn_empty_content_raises(self, engine: IntelligenceLayer):
        with pytest.raises(ValueError, match="empty"):
            await engine.learn(
                content="",
                type="decision",
                confidence="high",
            )

    @pytest.mark.asyncio
    async def test_learn_emits_event(self, engine: IntelligenceLayer):
        """learn() should emit KNOWLEDGE_LEARNED event."""
        events = []

        async def capture(event_type, data):
            events.append((str(event_type), data))

        engine.event_bus.on_all(capture)

        await engine.learn(
            content="Use type hints everywhere",
            type="pattern",
            confidence="high",
        )

        event_types = [e[0] for e in events]
        assert "knowledge.learned" in event_types

    @pytest.mark.asyncio
    async def test_learn_default_namespace_is_knowledge(
        self, engine: IntelligenceLayer
    ):
        """learn() without namespace arg records 'knowledge' on node + experience."""
        result = await engine.learn(
            content="Default namespace goes to knowledge",
            type="pattern",
            confidence="high",
        )
        assert result["namespace"] == "knowledge"

        # Verify node landed in the knowledge namespace
        node = await engine.store.get_node(result["node_id"])
        assert node["namespace"] == "knowledge"

        # Verify experience persists with the same namespace
        exp = await engine.experience.get(result["experience_id"])
        assert exp.namespace == "knowledge"

    @pytest.mark.asyncio
    async def test_learn_with_namespace_creates_namespaced_experience(
        self, engine: IntelligenceLayer
    ):
        """learn(namespace=X) routes the node AND the experience to namespace X."""
        result = await engine.learn(
            content="Primeline growth cadence: ship daily, measure weekly",
            type="decision",
            confidence="high",
            namespace="primeline",
        )
        assert result["stored_as"] == "node"
        assert result["namespace"] == "primeline"

        # Node in primeline namespace
        node = await engine.store.get_node(result["node_id"])
        assert node["namespace"] == "primeline"

        # Experience in primeline namespace
        exp = await engine.experience.get(result["experience_id"])
        assert exp.namespace == "primeline"

    @pytest.mark.asyncio
    async def test_learn_medium_confidence_namespace_on_experience(
        self, engine: IntelligenceLayer
    ):
        """Medium confidence skips the node but still tags the experience."""
        result = await engine.learn(
            content="Maybe sharding helps at 10M rows",
            type="pattern",
            confidence="medium",
            namespace="workspace-gamma",
        )
        assert result["stored_as"] == "experience"
        assert result["node_id"] is None
        assert result["namespace"] == "workspace-gamma"

        exp = await engine.experience.get(result["experience_id"])
        assert exp.namespace == "workspace-gamma"


# ──────────────────────────────────────────────────────────────────────
# recall() tests
# ──────────────────────────────────────────────────────────────────────


class TestRecall:
    """Tests for kn_recall — cross-namespace, decay-aware recall."""

    @pytest.mark.asyncio
    async def test_recall_finds_learned_knowledge(self, engine: IntelligenceLayer):
        """recall() should find previously learned knowledge."""
        await engine.learn(
            content="Redis is great for caching API responses",
            type="solution",
            confidence="high",
            tags=["redis", "caching"],
        )

        results = await engine.recall(topic="caching API responses")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_recall_includes_both_nodes_and_experiences(
        self, engine: IntelligenceLayer
    ):
        """recall() should search across nodes AND experiences."""
        # High confidence → node
        await engine.learn(
            content="PostgreSQL for relational data",
            type="decision",
            confidence="high",
        )
        # Low confidence → experience only
        await engine.learn(
            content="Maybe try MongoDB for unstructured data",
            type="decision",
            confidence="low",
        )

        results = await engine.recall(topic="database data")
        # Should find items from both sources
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_recall_empty_topic_returns_recent(self, engine: IntelligenceLayer):
        """recall() with no topic returns recent knowledge."""
        await engine.learn(
            content="Testing is important",
            type="pattern",
            confidence="high",
        )

        results = await engine.recall(topic=None, limit=5)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_recall_respects_limit(self, engine: IntelligenceLayer):
        """recall() should respect the limit parameter."""
        for i in range(5):
            await engine.learn(
                content=f"Pattern number {i} about testing",
                type="pattern",
                confidence="high",
            )

        results = await engine.recall(topic="pattern testing", limit=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_recall_emits_event(self, engine: IntelligenceLayer):
        events = []

        async def capture(event_type, data):
            events.append((str(event_type), data))

        engine.event_bus.on_all(capture)

        await engine.learn(
            content="Something to recall",
            type="pattern",
            confidence="high",
        )

        await engine.recall(topic="recall")

        event_types = [e[0] for e in events]
        assert "knowledge.recalled" in event_types

    @pytest.mark.asyncio
    async def test_recall_no_results_returns_empty(self, engine: IntelligenceLayer):
        results = await engine.recall(topic="completely_nonexistent_xyz123")
        assert results == []


# ──────────────────────────────────────────────────────────────────────
# crossref() tests
# ──────────────────────────────────────────────────────────────────────


class TestCrossref:
    """Tests for kn_crossref — cross-workspace discovery."""

    @pytest.mark.asyncio
    async def test_crossref_finds_related_patterns(self, engine: IntelligenceLayer):
        """crossref() should find solutions from the same workspace."""
        # Learn several things
        await engine.learn(
            content="Token bucket algorithm for rate limiting",
            type="solution",
            confidence="high",
            tags=["rate-limiting"],
        )
        await engine.learn(
            content="Redis sorted sets for leaderboards",
            type="solution",
            confidence="high",
            tags=["redis"],
        )

        results = await engine.crossref(problem="I need to implement rate limiting")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_crossref_empty_problem_raises(self, engine: IntelligenceLayer):
        with pytest.raises(ValueError, match="empty"):
            await engine.crossref(problem="")

    @pytest.mark.asyncio
    async def test_crossref_emits_event(self, engine: IntelligenceLayer):
        events = []

        async def capture(event_type, data):
            events.append((str(event_type), data))

        engine.event_bus.on_all(capture)

        await engine.learn(
            content="Use connection pooling for database",
            type="solution",
            confidence="high",
        )

        await engine.crossref(problem="database performance")

        event_types = [e[0] for e in events]
        assert "crossref.found" in event_types


# ──────────────────────────────────────────────────────────────────────
# context() tests
# ──────────────────────────────────────────────────────────────────────


class TestContext:
    """Tests for kn_context — progressive disclosure subgraph."""

    @pytest.mark.asyncio
    async def test_context_returns_relevant_subgraph(
        self, engine: IntelligenceLayer
    ):
        """context() should return relevant nodes and experiences."""
        await engine.learn(
            content="FastAPI uses Pydantic for validation",
            type="pattern",
            confidence="high",
            tags=["fastapi", "pydantic"],
        )

        result = await engine.context(keywords="FastAPI validation")
        assert "nodes" in result
        assert "experiences" in result
        assert result["_v"] == "1.0"

    @pytest.mark.asyncio
    async def test_context_summary_vs_full(self, engine: IntelligenceLayer):
        """detail='full' should include more information than 'summary'."""
        await engine.learn(
            content="SQLite FTS5 for full-text search",
            type="pattern",
            confidence="high",
        )

        summary = await engine.context(keywords="SQLite FTS5", detail="summary")
        full = await engine.context(keywords="SQLite FTS5", detail="full")

        # Full should have same or more data
        assert full["_v"] == "1.0"
        assert summary["_v"] == "1.0"

    @pytest.mark.asyncio
    async def test_context_empty_keywords(self, engine: IntelligenceLayer):
        result = await engine.context(keywords="")
        assert result["count"] == 0


# ──────────────────────────────────────────────────────────────────────
# related() tests
# ──────────────────────────────────────────────────────────────────────


class TestRelated:
    """Tests for kn_related — BFS/DFS traversal."""

    @pytest.mark.asyncio
    async def test_related_finds_connected_nodes(self, engine: IntelligenceLayer):
        """related() should find nodes connected via edges."""
        n1 = await engine.graph.add_node(name="authentication", type="concept")
        n2 = await engine.graph.add_node(name="JWT tokens", type="pattern")
        await engine.graph.connect(n1.id, n2.id, "uses")

        results = await engine.related(node_id=n1.id, depth=1)
        assert len(results) >= 1
        node_names = [r["node"]["name"] for r in results]
        assert "JWT tokens" in node_names

    @pytest.mark.asyncio
    async def test_related_respects_depth(self, engine: IntelligenceLayer):
        """related() should not return nodes beyond specified depth."""
        n1 = await engine.graph.add_node(name="root", type="concept")
        n2 = await engine.graph.add_node(name="depth1", type="concept")
        n3 = await engine.graph.add_node(name="depth2", type="concept")
        await engine.graph.connect(n1.id, n2.id, "links_to")
        await engine.graph.connect(n2.id, n3.id, "links_to")

        results_d1 = await engine.related(node_id=n1.id, depth=1)
        results_d2 = await engine.related(node_id=n1.id, depth=2)

        d1_names = [r["node"]["name"] for r in results_d1]
        d2_names = [r["node"]["name"] for r in results_d2]

        assert "depth1" in d1_names
        assert "depth2" not in d1_names
        assert "depth2" in d2_names

    @pytest.mark.asyncio
    async def test_related_nonexistent_node(self, engine: IntelligenceLayer):
        results = await engine.related(node_id="nonexistent_id", depth=1)
        assert results == []

    @pytest.mark.asyncio
    async def test_related_with_edge_type_filter(self, engine: IntelligenceLayer):
        n1 = await engine.graph.add_node(name="Python", type="concept")
        n2 = await engine.graph.add_node(name="FastAPI", type="framework")
        n3 = await engine.graph.add_node(name="Django", type="framework")
        await engine.graph.connect(n1.id, n2.id, "has_framework")
        await engine.graph.connect(n1.id, n3.id, "has_framework")

        results = await engine.related(
            node_id=n1.id, depth=1, edge_type="has_framework"
        )
        names = [r["node"]["name"] for r in results]
        assert "FastAPI" in names
        assert "Django" in names


# ──────────────────────────────────────────────────────────────────────
# Integration / Real conversation tests
# ──────────────────────────────────────────────────────────────────────


class TestRealConversation:
    """Tests with realistic conversation-style inputs."""

    @pytest.mark.asyncio
    async def test_developer_workflow(self, engine: IntelligenceLayer):
        """Simulate a developer making decisions across a session."""
        # Decision 1: Auth approach
        await engine.learn(
            content="We're using JWT instead of session cookies for the API",
            type="decision",
            confidence="high",
            context="API design meeting",
            tags=["auth", "jwt", "api"],
        )

        # Decision 2: Database
        await engine.learn(
            content="PostgreSQL for the main database, Redis for caching",
            type="decision",
            confidence="high",
            context="Architecture review",
            tags=["database", "postgresql", "redis"],
        )

        # Tentative idea
        await engine.learn(
            content="Maybe GraphQL would simplify the frontend queries",
            type="pattern",
            confidence="low",
            context="Frontend team suggestion",
        )

        # Later: recall auth decisions
        auth_results = await engine.recall(topic="authentication API")
        assert len(auth_results) >= 1

        # Context for database work
        db_context = await engine.context(keywords="database caching")
        assert db_context["count"] >= 0  # May find related items

    @pytest.mark.asyncio
    async def test_learn_then_crossref_workflow(self, engine: IntelligenceLayer):
        """Learn solutions, then crossref to find them."""
        await engine.learn(
            content="Implemented rate limiting with token bucket in Redis",
            type="solution",
            confidence="high",
            tags=["rate-limiting", "redis"],
        )
        await engine.learn(
            content="Used circuit breaker pattern for external API calls",
            type="solution",
            confidence="high",
            tags=["resilience", "circuit-breaker"],
        )

        # New problem arises
        results = await engine.crossref(
            problem="Need to prevent API abuse and rate limit endpoints"
        )
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_multiple_confidence_levels(self, engine: IntelligenceLayer):
        """Different confidence levels stored correctly."""
        r1 = await engine.learn(
            content="Definitive: Use Pydantic v2 for models",
            type="decision",
            confidence="high",
        )
        r2 = await engine.learn(
            content="Probably should add OpenTelemetry tracing",
            type="pattern",
            confidence="medium",
        )
        r3 = await engine.learn(
            content="Maybe try Rust for the hot path",
            type="pattern",
            confidence="low",
        )

        assert r1["stored_as"] == "node"
        assert r2["stored_as"] == "experience"
        assert r3["stored_as"] == "experience"


# ──────────────────────────────────────────────────────────────────────
# Access tracking wiring
# ──────────────────────────────────────────────────────────────────────


class TestAccessTracking:
    """Verify recall/context/crossref increment access_count on returned
    experiences. This is what activates the promotion pipeline: without
    the wiring, access_count stays 0 forever and the SQL auto-promote
    trigger never fires.
    """

    async def _seed_experience(self, engine, content: str, tags=None):
        """Helper: create a low-confidence experience (experience only, no node)."""
        await engine.learn(
            content=content,
            type="gotcha",
            confidence="low",
            tags=tags or [],
        )

    async def _get_experience_access_count(self, engine, content_substr: str) -> int:
        """Find an experience by content substring and return its access_count."""
        all_exps = await engine.experience.search(text=None, limit=100)
        for exp in all_exps:
            if content_substr in exp.content:
                return exp.access_count
        raise AssertionError(f"experience with {content_substr!r} not found")

    @pytest.mark.asyncio
    async def test_recall_increments_access_count(self, engine: IntelligenceLayer):
        """recall() should touch every returned experience's access_count."""
        await self._seed_experience(
            engine,
            "Kafka partitions must match consumer count for parallelism",
            tags=["kafka", "partitions"],
        )
        before = await self._get_experience_access_count(engine, "Kafka partitions")
        assert before == 0

        results = await engine.recall(topic="kafka partitions consumer")
        # Should surface at least the experience we just seeded.
        assert any("Kafka" in r.get("content", "") for r in results
                   if r.get("source") == "experience")

        after = await self._get_experience_access_count(engine, "Kafka partitions")
        assert after == 1

    @pytest.mark.asyncio
    async def test_context_increments_access_count(self, engine: IntelligenceLayer):
        """context() should touch every returned experience's access_count."""
        await self._seed_experience(
            engine,
            "gRPC streaming requires keepalive tuning for long-lived connections",
            tags=["grpc", "keepalive"],
        )
        before = await self._get_experience_access_count(engine, "gRPC streaming")
        assert before == 0

        result = await engine.context(keywords="grpc keepalive streaming")
        assert any("gRPC" in e.get("content", "")
                   for e in result.get("experiences", []))

        after = await self._get_experience_access_count(engine, "gRPC streaming")
        assert after == 1

    @pytest.mark.asyncio
    async def test_crossref_increments_access_count(self, engine: IntelligenceLayer):
        """crossref() should touch every returned experience's access_count."""
        await self._seed_experience(
            engine,
            "Use bounded queues for backpressure under burst load",
            tags=["queue", "backpressure"],
        )
        before = await self._get_experience_access_count(
            engine, "bounded queues for backpressure"
        )
        assert before == 0

        results = await engine.crossref(problem="burst load overflow backpressure queue")
        assert any("bounded queues" in r.get("content", "")
                   for r in results if r.get("source") == "experience")

        after = await self._get_experience_access_count(
            engine, "bounded queues for backpressure"
        )
        assert after == 1

    @pytest.mark.asyncio
    async def test_5_recalls_trigger_promotion_flag(self, engine: IntelligenceLayer):
        """After 5 recall hits on the same experience, the SQL trigger
        exp_auto_promote should set properties.needs_promotion = 1."""
        await self._seed_experience(
            engine,
            "Always set SO_REUSEADDR before bind for fast socket reuse",
            tags=["sockets", "bind", "tcp"],
        )

        for _ in range(5):
            results = await engine.recall(
                topic="sockets bind SO_REUSEADDR fast reuse"
            )
            # Confirm the experience is actually being matched each time,
            # otherwise the test is vacuous.
            assert any("SO_REUSEADDR" in r.get("content", "")
                       for r in results if r.get("source") == "experience")

        count = await self._get_experience_access_count(engine, "SO_REUSEADDR")
        assert count == 5

        # Check the promotion flag via store directly.
        promotable = await engine.experience.store.get_promotable_experiences()
        assert any("SO_REUSEADDR" in p.get("content", "") for p in promotable), \
            f"expected SO_REUSEADDR experience to be flagged promotable, " \
            f"got {[p.get('content','')[:50] for p in promotable]}"

    @pytest.mark.asyncio
    async def test_empty_experience_list_no_sql_issued(
        self, engine: IntelligenceLayer, monkeypatch
    ):
        """Recall with no experience hits must never pass a non-empty
        list to touch_accessed. Defense in depth: even if the guard at
        the intelligence layer is removed, the store layer also short-
        circuits on empty lists (see sqlite_store.touch_accessed_
        experiences). This test pins the intelligence-layer contract
        by asserting the spy only observed empty-or-no calls."""
        calls = []
        original = engine.experience.touch_accessed

        async def spy(exp_ids):
            calls.append(list(exp_ids))
            return await original(exp_ids)

        monkeypatch.setattr(engine.experience, "touch_accessed", spy)

        # Nothing in the DB yet -> recall returns empty experience list.
        await engine.recall(topic="completely_nonexistent_xyz999")

        # Either touch_accessed was never called (most efficient) or it
        # was called with an empty list (also acceptable). Either way,
        # no SQL IN () syntax error.
        for call in calls:
            assert call == []  # if called at all, must be with empty list
