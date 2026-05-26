"""Tests for the 5-verb relation vocabulary validation.

Phase 1 of `.claude/plans/2026-05-26-kairn-judgment-envelope-and-doctor.md`.

Validation modes:
- lax (`strict=False`, default for `GraphEngine.connect` / `kn_connect`):
  unknown verbs emit a one-line warning and pass through. Legacy
  vocabulary (incl. ~6086 historical `auto_related` edges) keeps working.
- strict (`strict=True`, used by `kn_judge` in Phase 3): unknown verbs
  raise `ValueError`. New judgment edges must use one of the five
  canonical verbs.

Empty / whitespace verbs raise in either mode (caller bug, not legacy).
"""

from __future__ import annotations

import logging

import pytest

from kairn.core.graph import GraphEngine
from kairn.events.bus import EventBus
from kairn.models.edge import RELATION_VERBS, validate_relation
from kairn.storage.sqlite_store import SQLiteStore


@pytest.fixture
async def graph(store: SQLiteStore) -> GraphEngine:
    bus = EventBus()
    return GraphEngine(store, bus)


# ----------------------------------------------------------------------
# Module-level validate_relation helper
# ----------------------------------------------------------------------


def test_relation_verbs_set_is_canonical_five() -> None:
    """The frozenset must be exactly the 5 verbs called out in the plan."""
    assert RELATION_VERBS == frozenset(
        {"conflicts_with", "supersedes", "compatible", "scoped", "related"}
    )


@pytest.mark.parametrize("verb", sorted(RELATION_VERBS))
def test_validate_relation_accepts_canonical_verbs_in_strict_mode(verb: str) -> None:
    """All 5 canonical verbs pass strict validation and return True."""
    assert validate_relation(verb, strict=True) is True


def test_validate_relation_strict_rejects_unknown_verb() -> None:
    """Strict mode raises ValueError on legacy / unknown vocabulary."""
    with pytest.raises(ValueError, match="Invalid relation verb"):
        validate_relation("auto_related", strict=True)


def test_validate_relation_lax_warns_but_passes_unknown_verb(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Lax mode returns False + warns; does NOT raise. Preserves legacy edges."""
    with caplog.at_level(logging.WARNING, logger="kairn.models.edge"):
        result = validate_relation("auto_related", strict=False)
    assert result is False
    assert any(
        "Legacy relation verb" in rec.message and "auto_related" in rec.message
        for rec in caplog.records
    ), "Expected one-line warning for legacy verb"


@pytest.mark.parametrize("bad_verb", ["", "   ", "\t\n"])
def test_validate_relation_rejects_empty_or_whitespace_in_both_modes(bad_verb: str) -> None:
    """Empty/whitespace is a caller bug, not a legacy verb. Raises always."""
    with pytest.raises(ValueError, match="non-empty string"):
        validate_relation(bad_verb, strict=False)
    with pytest.raises(ValueError, match="non-empty string"):
        validate_relation(bad_verb, strict=True)


@pytest.mark.parametrize("non_str", [None, 42, 3.14, ["related"]])
def test_validate_relation_rejects_non_string(non_str: object) -> None:
    """Non-string verbs are caller bugs in either mode.

    Both lax and strict are checked so a future reorder of the guards
    cannot silently make one mode accept non-strings while tests stay
    green (mirrors the pattern in
    `test_validate_relation_rejects_empty_or_whitespace_in_both_modes`).
    """
    with pytest.raises(ValueError, match="non-empty string"):
        validate_relation(non_str, strict=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty string"):
        validate_relation(non_str, strict=True)  # type: ignore[arg-type]


def test_validate_relation_lax_returns_true_for_canonical() -> None:
    """Canonical verbs return True in lax mode too (not just strict)."""
    for verb in RELATION_VERBS:
        assert validate_relation(verb, strict=False) is True


# ----------------------------------------------------------------------
# GraphEngine.connect() integration with validation
# ----------------------------------------------------------------------


async def test_connect_accepts_canonical_verb_lax(graph: GraphEngine) -> None:
    """Default lax connect persists a canonical-verb edge cleanly."""
    n1 = await graph.add_node(name="A canon", type="concept")
    n2 = await graph.add_node(name="B canon", type="concept")

    edge = await graph.connect(n1.id, n2.id, "supersedes")
    assert edge.type == "supersedes"


async def test_connect_accepts_legacy_verb_lax_with_warning(
    graph: GraphEngine, caplog: pytest.LogCaptureFixture
) -> None:
    """Backward-compat: legacy verbs persist; warning emitted; no exception."""
    n1 = await graph.add_node(name="A legacy", type="concept")
    n2 = await graph.add_node(name="B legacy", type="concept")

    with caplog.at_level(logging.WARNING, logger="kairn.models.edge"):
        edge = await graph.connect(n1.id, n2.id, "has_finding")
    assert edge.type == "has_finding"
    assert any("has_finding" in rec.message for rec in caplog.records)


async def test_connect_strict_rejects_legacy_verb(graph: GraphEngine) -> None:
    """Strict mode (used by kn_judge in Phase 3) raises on legacy vocabulary."""
    n1 = await graph.add_node(name="A strict", type="concept")
    n2 = await graph.add_node(name="B strict", type="concept")

    with pytest.raises(ValueError, match="Invalid relation verb"):
        await graph.connect(n1.id, n2.id, "auto_related", strict_relation=True)


async def test_connect_strict_accepts_canonical(graph: GraphEngine) -> None:
    """Strict mode happy path: a canonical verb persists."""
    n1 = await graph.add_node(name="A strict ok", type="concept")
    n2 = await graph.add_node(name="B strict ok", type="concept")

    edge = await graph.connect(
        n1.id, n2.id, "conflicts_with", strict_relation=True
    )
    assert edge.type == "conflicts_with"


async def test_connect_empty_verb_raises_in_lax_mode(graph: GraphEngine) -> None:
    """Empty verb is caller bug - raise even in default lax mode."""
    n1 = await graph.add_node(name="A empty", type="concept")
    n2 = await graph.add_node(name="B empty", type="concept")

    with pytest.raises(ValueError, match="non-empty string"):
        await graph.connect(n1.id, n2.id, "  ")


async def test_connect_strict_does_not_break_existing_lax_callers(
    graph: GraphEngine,
) -> None:
    """Regression: existing call sites (no strict_relation arg) still work."""
    n1 = await graph.add_node(name="A regress", type="concept")
    n2 = await graph.add_node(name="B regress", type="concept")

    # Mirrors existing test_connect_nodes shape - keyword-only strict param
    # must default to False without forcing call-site changes.
    edge = await graph.connect(n1.id, n2.id, "uses", weight=0.9)
    assert edge.weight == 0.9
    assert edge.type == "uses"
