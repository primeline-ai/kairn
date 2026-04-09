"""Tests for CLI subcommands added in Gate 3 automation-tier1-cli.

Covers the 5 new intelligence subcommands that mirror MCP tools:
learn, recall, context, memories, crossref.

Uses subprocess to exercise the CLI as a shell user would, verifying JSON
output shape and round-trip behavior. Each test creates an isolated workspace
via tmp_path to avoid cross-test contamination.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run_kairn(*args: str, check: bool = True) -> tuple[int, str, str]:
    """Run `kairn` CLI as a subprocess. Returns (returncode, stdout, stderr)."""
    cmd = [sys.executable, "-m", "kairn.cli", *args]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"CLI command failed: {' '.join(cmd)}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    return result.returncode, result.stdout, result.stderr


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Initialize a fresh kairn workspace in tmp_path and return its directory."""
    ws = tmp_path / "ws"
    ws.mkdir()
    _run_kairn("init", str(ws))
    assert (ws / "kairn.db").exists(), "init did not create kairn.db"
    return ws


# ──────────────────────────────────────────────────────────
# learn
# ──────────────────────────────────────────────────────────


class TestLearn:
    def test_learn_high_confidence_creates_node_and_experience(self, workspace: Path):
        _, out, _ = _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "Use parameterized SQL to prevent injection",
            "--type",
            "pattern",
            "--confidence",
            "high",
        )
        result = json.loads(out)
        assert result["_v"] == "1.0"
        assert result["stored_as"] == "node"
        assert result["node_id"] is not None
        assert result["experience_id"] is not None
        assert result["type"] == "pattern"
        assert result["confidence"] == "high"

    def test_learn_medium_confidence_creates_experience_only(self, workspace: Path):
        _, out, _ = _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "Maybe Redis is better than in-memory for caching",
            "--type",
            "pattern",
            "--confidence",
            "medium",
        )
        result = json.loads(out)
        assert result["stored_as"] == "experience"
        assert result["node_id"] is None
        assert result["experience_id"] is not None

    def test_learn_low_confidence_creates_experience_only(self, workspace: Path):
        _, out, _ = _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "Perhaps GraphQL simplifies frontend queries",
            "--type",
            "decision",
            "--confidence",
            "low",
        )
        result = json.loads(out)
        assert result["stored_as"] == "experience"
        assert result["node_id"] is None

    def test_learn_with_tags(self, workspace: Path):
        _, out, _ = _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "Token bucket rate limiting",
            "--type",
            "solution",
            "--tags",
            "rate-limiting,api,security",
        )
        result = json.loads(out)
        assert result["stored_as"] == "node"

    def test_learn_invalid_confidence_rejected_by_click(self, workspace: Path):
        rc, _, stderr = _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "x",
            "--type",
            "pattern",
            "--confidence",
            "bogus",
            check=False,
        )
        assert rc != 0
        assert "bogus" in stderr.lower() or "invalid" in stderr.lower()

    def test_learn_invalid_type_surfaces_error(self, workspace: Path):
        rc, _, stderr = _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "x",
            "--type",
            "not_a_valid_type",
            check=False,
        )
        assert rc != 0
        err = json.loads(stderr)
        assert "error" in err
        assert "not_a_valid_type" in err["error"]

    def test_learn_default_namespace_is_knowledge(self, workspace: Path):
        _, out, _ = _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "Default namespace via CLI",
            "--type",
            "pattern",
        )
        result = json.loads(out)
        assert result["namespace"] == "knowledge"

    def test_learn_with_explicit_namespace(self, workspace: Path):
        _, out, _ = _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "Circuit breaker pattern for fault tolerance",
            "--type",
            "pattern",
            "--namespace",
            "tenant-alpha",
        )
        result = json.loads(out)
        assert result["_v"] == "1.0"
        assert result["namespace"] == "tenant-alpha"
        assert result["stored_as"] == "node"
        assert result["node_id"] is not None


# ──────────────────────────────────────────────────────────
# recall
# ──────────────────────────────────────────────────────────


class TestRecall:
    def test_recall_returns_previously_learned_knowledge(self, workspace: Path):
        _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "Use connection pooling for database performance",
            "--type",
            "pattern",
        )
        _, out, _ = _run_kairn(
            "recall",
            str(workspace),
            "--topic",
            "database pooling",
        )
        result = json.loads(out)
        assert result["_v"] == "1.0"
        assert result["count"] >= 1
        assert isinstance(result["results"], list)

    def test_recall_respects_limit(self, workspace: Path):
        for i in range(5):
            _run_kairn(
                "learn",
                str(workspace),
                "--content",
                f"Pattern number {i} about testing strategies",
                "--type",
                "pattern",
            )
        _, out, _ = _run_kairn(
            "recall",
            str(workspace),
            "--topic",
            "testing",
            "--limit",
            "2",
        )
        result = json.loads(out)
        assert result["count"] <= 2

    def test_recall_without_topic_returns_empty_or_recent(self, workspace: Path):
        _, out, _ = _run_kairn("recall", str(workspace))
        result = json.loads(out)
        assert "count" in result
        assert isinstance(result["results"], list)


# ──────────────────────────────────────────────────────────
# context
# ──────────────────────────────────────────────────────────


class TestContext:
    def test_context_returns_nodes_and_experiences(self, workspace: Path):
        _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "FastAPI uses Pydantic for request validation",
            "--type",
            "pattern",
        )
        _, out, _ = _run_kairn(
            "context",
            str(workspace),
            "--keywords",
            "FastAPI validation",
        )
        result = json.loads(out)
        assert result["_v"] == "1.0"
        assert "nodes" in result
        assert "experiences" in result
        assert "query" in result
        assert result["query"] == "FastAPI validation"

    def test_context_full_detail_includes_description(self, workspace: Path):
        _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "SQLite FTS5 for full-text search",
            "--type",
            "pattern",
        )
        _, out, _ = _run_kairn(
            "context",
            str(workspace),
            "--keywords",
            "SQLite FTS5",
            "--detail",
            "full",
        )
        result = json.loads(out)
        assert result["detail"] == "full"


# ──────────────────────────────────────────────────────────
# memories
# ──────────────────────────────────────────────────────────


class TestMemories:
    def test_memories_returns_experiences(self, workspace: Path):
        _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "Circuit breaker for external API calls",
            "--type",
            "solution",
            "--tags",
            "resilience",
        )
        _, out, _ = _run_kairn("memories", str(workspace))
        result = json.loads(out)
        assert result["_v"] == "1.0"
        assert result["count"] >= 1
        exp = result["experiences"][0]
        assert "id" in exp
        assert "type" in exp
        assert "content" in exp
        assert "confidence" in exp
        assert "relevance" in exp
        assert "tags" in exp

    def test_memories_text_filter(self, workspace: Path):
        _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "UNIQUE_CLI_TOKEN_XYZ pattern for testing",
            "--type",
            "pattern",
        )
        _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "Unrelated solution about queues",
            "--type",
            "solution",
        )
        _, out, _ = _run_kairn(
            "memories",
            str(workspace),
            "--text",
            "UNIQUE_CLI_TOKEN_XYZ",
        )
        result = json.loads(out)
        assert result["count"] >= 1
        assert any(
            "UNIQUE_CLI_TOKEN_XYZ" in e["content"] for e in result["experiences"]
        )

    def test_memories_type_filter(self, workspace: Path):
        _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "Test solution one",
            "--type",
            "solution",
        )
        _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "Test pattern one",
            "--type",
            "pattern",
        )
        _, out, _ = _run_kairn(
            "memories",
            str(workspace),
            "--type",
            "solution",
        )
        result = json.loads(out)
        assert all(e["type"] == "solution" for e in result["experiences"])


# ──────────────────────────────────────────────────────────
# crossref
# ──────────────────────────────────────────────────────────


class TestCrossref:
    def test_crossref_finds_related_solutions(self, workspace: Path):
        _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "Token bucket algorithm for rate limiting",
            "--type",
            "solution",
            "--tags",
            "rate-limiting",
        )
        _, out, _ = _run_kairn(
            "crossref",
            str(workspace),
            "--problem",
            "Need to prevent API abuse with rate limiting",
        )
        result = json.loads(out)
        assert result["_v"] == "1.0"
        assert result["count"] >= 1

    def test_crossref_empty_problem_errors(self, workspace: Path):
        # Click requires --problem, so pass empty string explicitly
        rc, _, stderr = _run_kairn(
            "crossref",
            str(workspace),
            "--problem",
            "",
            check=False,
        )
        assert rc != 0
        err = json.loads(stderr)
        assert "error" in err


# ──────────────────────────────────────────────────────────
# Cross-command round trips
# ──────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_learn_then_recall_then_context(self, workspace: Path):
        """End-to-end: learn something, recall it by topic, then get its context."""
        _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "OAuth2 PKCE for mobile client authentication",
            "--type",
            "decision",
            "--confidence",
            "high",
            "--tags",
            "auth,oauth2,mobile",
        )

        _, out, _ = _run_kairn("recall", str(workspace), "--topic", "mobile auth")
        recall_result = json.loads(out)
        assert recall_result["count"] >= 1

        _, out, _ = _run_kairn(
            "context",
            str(workspace),
            "--keywords",
            "oauth2 mobile",
        )
        context_result = json.loads(out)
        assert context_result["_v"] == "1.0"
        assert "nodes" in context_result
        assert "experiences" in context_result
        assert "query" in context_result
        assert context_result["query"] == "oauth2 mobile"

    def test_multiple_learns_appear_in_memories(self, workspace: Path):
        contents = [
            "First solution about databases",
            "Second solution about caching",
            "Third pattern about queues",
        ]
        for c in contents:
            _run_kairn(
                "learn",
                str(workspace),
                "--content",
                c,
                "--type",
                "solution",
            )
        _, out, _ = _run_kairn("memories", str(workspace), "--limit", "50")
        result = json.loads(out)
        assert result["count"] >= 3


# ──────────────────────────────────────────────────────────
# Error paths
# ──────────────────────────────────────────────────────────


class TestErrors:
    def test_learn_on_missing_workspace_exits_nonzero(self, tmp_path: Path):
        nonexistent = tmp_path / "does-not-exist"
        rc, _, _ = _run_kairn(
            "learn",
            str(nonexistent),
            "--content",
            "x",
            "--type",
            "pattern",
            check=False,
        )
        # click's exists=True guard catches this
        assert rc != 0

    def test_recall_on_missing_workspace_exits_nonzero(self, tmp_path: Path):
        nonexistent = tmp_path / "does-not-exist"
        rc, _, _ = _run_kairn(
            "recall",
            str(nonexistent),
            check=False,
        )
        assert rc != 0


# ──────────────────────────────────────────────────────────
# Phase 4: Graph CRUD + query + project/idea/log parity
# ──────────────────────────────────────────────────────────


class TestAddConnectRemove:
    def test_add_creates_node(self, workspace: Path):
        _, out, _ = _run_kairn(
            "add",
            str(workspace),
            "--name",
            "Auth Gateway",
            "--type",
            "concept",
            "--description",
            "Handles all authentication flows",
            "--tags",
            "auth,security",
        )
        result = json.loads(out)
        assert result["_v"] == "1.0"
        assert result["name"] == "Auth Gateway"
        assert result["id"]

    def test_add_with_explicit_namespace(self, workspace: Path):
        _, out, _ = _run_kairn(
            "add",
            str(workspace),
            "--name",
            "Isolated Concept",
            "--type",
            "concept",
            "--namespace",
            "tenant-alpha",
        )
        result = json.loads(out)
        assert result["id"]

    def test_connect_creates_edge(self, workspace: Path):
        # Build two nodes to link
        _, out_a, _ = _run_kairn(
            "add", str(workspace), "--name", "Cache Layer", "--type", "concept"
        )
        _, out_b, _ = _run_kairn(
            "add", str(workspace), "--name", "API Server", "--type", "concept"
        )
        node_a = json.loads(out_a)["id"]
        node_b = json.loads(out_b)["id"]

        _, out, _ = _run_kairn(
            "connect",
            str(workspace),
            "--source-id",
            node_a,
            "--target-id",
            node_b,
            "--edge-type",
            "depends_on",
            "--weight",
            "0.8",
        )
        result = json.loads(out)
        assert result["source_id"] == node_a
        assert result["target_id"] == node_b
        assert result["type"] == "depends_on"

    def test_remove_node_succeeds(self, workspace: Path):
        _, out, _ = _run_kairn(
            "add", str(workspace), "--name", "Disposable", "--type", "concept"
        )
        node_id = json.loads(out)["id"]

        _, out, _ = _run_kairn("remove", str(workspace), "--node-id", node_id)
        result = json.loads(out)
        assert result["removed"] == "node"
        assert result["id"] == node_id

    def test_remove_missing_node_errors(self, workspace: Path):
        rc, _, stderr = _run_kairn(
            "remove",
            str(workspace),
            "--node-id",
            "nonexistent-id",
            check=False,
        )
        assert rc != 0
        err = json.loads(stderr)
        assert "nonexistent-id" in err["error"]

    def test_remove_requires_target(self, workspace: Path):
        rc, _, _ = _run_kairn("remove", str(workspace), check=False)
        assert rc != 0


class TestQuery:
    def test_query_nodes_by_text(self, workspace: Path):
        _run_kairn(
            "add",
            str(workspace),
            "--name",
            "Redis Cache Pattern",
            "--type",
            "pattern",
            "--description",
            "In-memory cache with TTL",
        )
        _, out, _ = _run_kairn("query", str(workspace), "--text", "redis")
        result = json.loads(out)
        assert result["_v"] == "1.0"
        assert result["count"] >= 1
        assert isinstance(result["nodes"], list)

    def test_query_nodes_by_namespace(self, workspace: Path):
        _run_kairn(
            "add",
            str(workspace),
            "--name",
            "Scoped Node",
            "--type",
            "concept",
            "--namespace",
            "team-beta",
        )
        _, out, _ = _run_kairn(
            "query", str(workspace), "--namespace", "team-beta"
        )
        result = json.loads(out)
        assert result["count"] >= 1

    def test_query_since_returns_experiences(self, workspace: Path):
        # Learn something so there's an experience after epoch
        _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "Time-based query demo",
            "--type",
            "pattern",
            "--confidence",
            "medium",
        )
        _, out, _ = _run_kairn(
            "query",
            str(workspace),
            "--since",
            "2000-01-01T00:00:00",
        )
        result = json.loads(out)
        assert result["_v"] == "1.0"
        assert result["count"] >= 1
        assert isinstance(result["experiences"], list)

    def test_query_since_format_json_emits_bare_list(self, workspace: Path):
        """`--format json --since X` emits a JSON array directly.

        Shell-based replication consumers expect to pipe this into jq
        or json.load without needing to unwrap an envelope.
        """
        _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "Bidirectional sync input",
            "--type",
            "decision",
            "--confidence",
            "medium",
        )
        _, out, _ = _run_kairn(
            "query",
            str(workspace),
            "--since",
            "2000-01-01T00:00:00",
            "--format",
            "json",
        )
        parsed = json.loads(out)
        assert isinstance(parsed, list), "Expected bare JSON list for sync consumers"
        assert len(parsed) >= 1
        # Each entry should look like an experience row
        assert "id" in parsed[0]
        assert "content" in parsed[0]
        assert "created_at" in parsed[0]

    def test_query_since_combined_with_namespace_filter(self, workspace: Path):
        """`--since` + `--namespace` should narrow to that tenant only.

        Locks in the combined filter path through storage.query_experiences_since,
        so a replication consumer scoped to one namespace only sees its own rows.
        """
        # Seed two experiences in different namespaces
        _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "Alpha-tenant signal",
            "--type",
            "pattern",
            "--confidence",
            "medium",
            "--namespace",
            "alpha",
        )
        _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "Beta-tenant signal",
            "--type",
            "pattern",
            "--confidence",
            "medium",
            "--namespace",
            "beta",
        )

        _, out, _ = _run_kairn(
            "query",
            str(workspace),
            "--since",
            "2000-01-01T00:00:00",
            "--namespace",
            "alpha",
            "--format",
            "json",
        )
        parsed = json.loads(out)
        assert isinstance(parsed, list)
        assert len(parsed) >= 1
        # Every returned row must belong to the filtered namespace
        for row in parsed:
            assert row["namespace"] == "alpha"

    def test_query_since_empty_window_returns_empty_list(self, workspace: Path):
        """Future `--since` should produce an empty list (still valid JSON)."""
        _, out, _ = _run_kairn(
            "query",
            str(workspace),
            "--since",
            "2099-12-31T23:59:59",
            "--format",
            "json",
        )
        parsed = json.loads(out)
        assert parsed == []


class TestPrune:
    def test_prune_returns_envelope(self, workspace: Path):
        # Nothing to prune on a fresh workspace
        _, out, _ = _run_kairn("prune", str(workspace), "--threshold", "0.5")
        result = json.loads(out)
        assert result["_v"] == "1.0"
        assert "pruned_count" in result
        assert isinstance(result["pruned_ids"], list)


class TestPromotePending:
    def test_promote_pending_empty_returns_envelope(self, workspace: Path):
        """Fresh workspace: nothing flagged, promote-pending is a no-op."""
        _, out, _ = _run_kairn("promote-pending", str(workspace))
        result = json.loads(out)
        assert result["_v"] == "1.0"
        assert result["flagged_total"] == 0
        assert result["attempted"] == 0
        assert result["promoted"] == 0
        assert result["raced"] == 0
        assert result["failed"] == 0
        assert result["nodes_created"] == []

    def test_promote_pending_promotes_after_5_recalls(self, workspace: Path):
        """End-to-end: learn -> recall 5x -> promote-pending creates a node."""
        # Seed a low-confidence experience (no node yet)
        _run_kairn(
            "learn", str(workspace),
            "--content", "Promotion sweeper end-to-end CLI test",
            "--type", "gotcha",
            "--confidence", "low",
            "--tags", "sweeper-cli-test",
        )

        # Recall 5 times so the access tracking wiring fires the trigger.
        for _ in range(5):
            _run_kairn(
                "recall", str(workspace),
                "--topic", "promotion sweeper end-to-end",
            )

        # Now promote-pending should find and promote the experience.
        _, out, _ = _run_kairn("promote-pending", str(workspace))
        result = json.loads(out)
        assert result["flagged_total"] >= 1
        assert result["attempted"] >= 1
        assert result["promoted"] >= 1
        assert result["raced"] == 0
        # Invariant: attempted == promoted + raced + failed
        assert result["attempted"] == (
            result["promoted"] + result["raced"] + result["failed"]
        )
        assert len(result["nodes_created"]) == result["promoted"]

    def test_promote_pending_respects_limit_flag(self, workspace: Path):
        _, out, _ = _run_kairn(
            "promote-pending", str(workspace), "--limit", "5"
        )
        result = json.loads(out)
        assert result["_v"] == "1.0"
        assert isinstance(result["promoted"], int)
        assert isinstance(result["attempted"], int)
        assert isinstance(result["raced"], int)


class TestProjectLog:
    def test_project_create_and_log_progress(self, workspace: Path):
        _, out, _ = _run_kairn(
            "project",
            str(workspace),
            "--name",
            "Launch Playbook",
            "--goals",
            "Ship v1,Validate users,Measure churn",
        )
        proj = json.loads(out)
        assert proj["_v"] == "1.0"
        assert proj["name"] == "Launch Playbook"
        assert proj["phase"] == "planning"
        project_id = proj["id"]

        _, out, _ = _run_kairn("projects", str(workspace))
        result = json.loads(out)
        assert any(p["id"] == project_id for p in result["projects"])

        _, out, _ = _run_kairn(
            "log",
            str(workspace),
            "--project-id",
            project_id,
            "--action",
            "Built the onboarding flow",
            "--type",
            "progress",
        )
        entry = json.loads(out)
        assert entry["project_id"] == project_id
        assert entry["type"] == "progress"
        assert entry["action"] == "Built the onboarding flow"

    def test_log_failure_entry(self, workspace: Path):
        _, out, _ = _run_kairn(
            "project", str(workspace), "--name", "Tracked Project"
        )
        project_id = json.loads(out)["id"]

        _, out, _ = _run_kairn(
            "log",
            str(workspace),
            "--project-id",
            project_id,
            "--action",
            "Deploy step failed",
            "--type",
            "failure",
            "--result",
            "Network timeout",
            "--next-step",
            "Retry with exponential backoff",
        )
        entry = json.loads(out)
        assert entry["type"] == "failure"


class TestIdeasCli:
    def test_idea_create_then_list(self, workspace: Path):
        _, out, _ = _run_kairn(
            "idea",
            str(workspace),
            "--title",
            "Add dark mode",
            "--category",
            "ux",
            "--score",
            "0.85",
        )
        created = json.loads(out)
        assert created["title"] == "Add dark mode"
        assert created["status"] == "draft"

        _, out, _ = _run_kairn("ideas", str(workspace))
        listing = json.loads(out)
        assert listing["count"] >= 1
        assert any(i["title"] == "Add dark mode" for i in listing["ideas"])


class TestRelatedCli:
    def test_related_returns_envelope(self, workspace: Path):
        _, out_a, _ = _run_kairn(
            "add", str(workspace), "--name", "Anchor", "--type", "concept"
        )
        _, out_b, _ = _run_kairn(
            "add", str(workspace), "--name", "Neighbour", "--type", "concept"
        )
        node_a = json.loads(out_a)["id"]
        node_b = json.loads(out_b)["id"]
        _run_kairn(
            "connect",
            str(workspace),
            "--source-id",
            node_a,
            "--target-id",
            node_b,
            "--edge-type",
            "references",
        )

        _, out, _ = _run_kairn(
            "related", str(workspace), "--node-id", node_a, "--depth", "1"
        )
        result = json.loads(out)
        assert result["_v"] == "1.0"
        assert "count" in result
        assert isinstance(result["results"], list)


class TestReplicationOutputShape:
    """Integration test: `query --since --format json` output must be
    parseable by shell-based replication consumers.

    A representative consumer does:
        echo "$EXPORT" | python3 -c 'import json,sys; \\
            d=json.load(sys.stdin); \\
            print(len(d) if isinstance(d,list) else "unknown")'

    That pattern requires the output to be a bare JSON list, not an envelope.
    """

    def test_output_parses_as_json_list_with_len(self, workspace: Path):
        _run_kairn(
            "learn",
            str(workspace),
            "--content",
            "Sync integration content",
            "--type",
            "pattern",
            "--confidence",
            "medium",
        )
        _, out, _ = _run_kairn(
            "query",
            str(workspace),
            "--since",
            "2000-01-01T00:00:00",
            "--format",
            "json",
        )
        # Exactly what the bash script does:
        parsed = json.loads(out)
        assert isinstance(parsed, list)
        count = len(parsed) if isinstance(parsed, list) else "unknown"
        assert isinstance(count, int)
        assert count >= 1
