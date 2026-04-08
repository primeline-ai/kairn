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
