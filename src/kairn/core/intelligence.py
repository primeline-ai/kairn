"""Intelligence layer — learn, recall, crossref, context, related.

Bridges graph, experience, and router engines into unified knowledge operations.
"""

from __future__ import annotations

import re

import asyncio
import logging
import re
import sqlite3
import unicodedata
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from kairn.core.experience import ExperienceEngine
from kairn.core.fts import (  # used here + re-exported for back-compat
    BM25_RELEVANCE_MIDPOINT,
    _to_fts_query,
    bm25_match,
    bm25_to_relevance,
    term_coverage,
)
from kairn.core.graph import GraphEngine
from kairn.core.ideas import IdeaEngine
from kairn.core.memory import ProjectMemory
from kairn.core.router import ContextRouter
from kairn.events.bus import EventBus
from kairn.events.types import EventType
from kairn.models.experience import VALID_CONFIDENCES, VALID_TYPES
from kairn.models.node import Node
from kairn.relevance import (
    RELEVANCE_KIND_MATCH,
    RELEVANCE_KIND_RECENCY,
    RELEVANCE_KIND_SIMILARITY,
    RELEVANCE_KIND_UNSCORED,
    RELEVANCE_KINDS,
)
from kairn.storage.base import StorageBackend

logger = logging.getLogger(__name__)

# FTS5 query shaping (_to_fts_query, _STOP_WORDS) now lives in core.fts and is
# re-exported via the top-of-module import so the experience path can share it
# without an import cycle. See core/fts.py.


# Max candidates returned by kn_learn FTS5 scan. Set conservatively;
# Phase 0 bench confirmed p95 = 1.258ms for the 5-row LIMIT shape at
# the current 4923-node scale (`_autonomous/benchmarks/kairn-fts5-latency.json`).
_CANDIDATES_LIMIT = 5
_CANDIDATE_SNIPPET_CHARS = 160

# Both live in core.fts: the experience path needs the same transform, and
# `intelligence` imports `experience`, so the shared helper cannot live here.
_BM25_RELEVANCE_MIDPOINT = BM25_RELEVANCE_MIDPOINT
_bm25_to_relevance = bm25_to_relevance



# Reporting resolution for a match-derived relevance. SIX decimals, not four.
# MEASURED on this branch: a one- or two-document store puts the FTS5 bm25
# rank at ~-4e-06 (there is no corpus for IDF to work with), so the saturating
# transform yields ~1.1e-06 and `round(match, 4)` is exactly 0.0 - including
# for a verbatim 4-of-4 term match. `round(match, 6)` is 1e-06, and the whole
# effect disappears at three documents (match 0.29). Four decimals therefore
# fabricated a zero in exactly the case a fresh workspace is in.
#
# Two consumers were misled by that zero: anything filtering `relevance > 0`
# (the review's own example), and `min_relevance` itself, which is compared
# against this same ROUNDED value in `_keyword_node_recall` - so any positive
# `min_relevance` emptied the node side of a new workspace.
#
# What more decimals do NOT fix, and no rounding can: bm25 magnitude scales
# with corpus size, so the same query/document pair reports 1e-06 here and
# 0.92 on a 500-document store. A consumer holding an ABSOLUTE threshold is
# still misled. Fixing that needs a corpus-independent quantity (term coverage
# is one) reported alongside, which is a wire-schema change, not a rounding.
#
# The experience path already reports at 6 decimals (`ExperienceEngine.search`
# stores `round(sort_value, 6)`), so matching it here also puts both kinds on
# ONE resolution - which `crossref` now depends on, because it sorts them
# against each other.
_RELEVANCE_DECIMALS = 6

# How many candidates to pull from the store before reranking by the score we
# actually report. The store truncates in RAW bm25 order (`ORDER BY rank
# LIMIT ?`), so a rerank that only ever sees `limit` rows cannot recover a
# document the raw order already dropped - it can only shuffle the survivors.
# MEASURED on a 26-node store, query "postgres connection pool exhaustion",
# one document answering all four terms against five one-word notes holding
# only the rarest term:
#
#     limit=3   the 4-of-4 document is ABSENT, three one-word notes fill it
#     limit=5   still absent, five one-word notes fill it
#     limit=10  it leads, at 0.101312 against 0.061231
#
# so at a realistic `limit` the rerank was decorative. `_semantic_node_recall`
# already had this right with `semantic_top_n` (default 30); this is the same
# over-fetch for the keyword path. The floor keeps a small `limit` from
# reranking a pool too shallow to contain the answer.
_RERANK_POOL_FACTOR = 4
_RERANK_POOL_MIN = 30


def _rerank_pool(limit: int) -> int:
    """Candidate count to fetch before reranking down to `limit`."""
    return max(_RERANK_POOL_MIN, limit * _RERANK_POOL_FACTOR)


def _node_relevance(
    rank: float | None,
    terms: list[str],
    name: str | None,
    description: str | None,
) -> float:
    """The ONE reported-and-ordering relevance for a node on a text query.

    bm25 match strength scaled by how much of the question the node actually
    answers. Written once and shared by `_keyword_node_recall` and `crossref`
    because those two were the N-1-of-N pair: recall measured its nodes while
    crossref labelled every node with the constant 1.0, and then sorted the
    two kinds against each other. A placeholder always beats a measurement,
    which is how a 4-of-4 experience lost its slot to a 1-of-4 node.

    A browse query (no terms) has no match signal to report, so it keeps the
    1.0 that `bm25_to_relevance` documents for `rank is None`.
    """
    if not terms:
        return bm25_to_relevance(rank)
    return round(
        bm25_match(rank) * term_coverage(terms, name, description),
        _RELEVANCE_DECIMALS,
    )


def _reported_relevance(exp: Any, now: datetime) -> float:
    """The relevance a caller sees for an experience.

    Prefers the match-aware value the recall computed (`recall_relevance`,
    set by ExperienceEngine.search when the query had text). Falls back to
    pure decay for browse queries and for any path that did not go through
    that gate - so this is additive, never a behaviour change on the no-text
    path. Reporting raw decay for a text query printed ~0.98 against alien
    queries, which is the same fake-relevance defect the node path had.
    """
    # DELEGATES. The rule lives on the model (Experience.reported_relevance)
    # because the model is the one module every reporting surface can import,
    # and because a second copy here is what let models/experience.py go on
    # emitting pure decay after the wire surfaces were fixed.
    return exp.reported_relevance(at=now)


def _fts_terms(fts_query: str | None) -> list[str]:
    """Recover the individual terms from a `to_fts_query` output.

    `to_fts_query` emits `"a" OR "b" OR "c"`, so the quoted spans are exactly
    the searchable terms. Returns [] for a browse query (None) or a malformed
    string, which callers treat as "no coverage signal available".
    """
    if not fts_query:
        return []
    return re.findall(r'"([^"]+)"', fts_query)


# Moved to `core.fts` so the EXPERIENCE path can use it too (it could not
# import this module - `intelligence` imports `experience`). The private name
# stays as the alias every call site here already uses.
_term_coverage = term_coverage


def _fold_diacritics(text: str) -> str:
    """Strip combining marks, the way FTS5's default `unicode61` tokenizer does.

    `unicode61` folds diacritics when it indexes, so `Zürich` is stored as
    `zurich` and an ASCII query for `zurich` matches it. Any code comparing a
    query term against raw text has to fold too, or it disagrees with the index
    it is scoring.

    The ASCII short-circuit is not a heuristic and does not change any result:
    NFD leaves an all-ASCII string unchanged, and no ASCII codepoint has a
    non-zero combining class, so the loop below provably returns `text` itself.
    It matters because this runs once per TOKEN of every candidate document -
    hundreds of calls per `context()` - and the overwhelming majority of those
    tokens are ASCII. Measured on a 12.9k-node store: dropping the needless NFD
    pass cut the coverage-scoring step of a ranked `context()` call from 38ms to
    under 3ms, which is the difference between fitting and not fitting inside a
    caller's latency budget. `test_fold_diacritics_ascii_shortcut_is_equivalent`
    pins the equivalence against the unoptimised definition.
    """
    if text.isascii():
        return text
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)
    )


# Token class matching FTS5's `unicode61`: alphanumeric runs, and NOTHING else.
# Deliberately not `\w`, which keeps `_` inside a token - see `_term_coverage`.
_INDEX_TOKEN_RE = re.compile(r"[^\W_]+")


def _index_tokens(text: str) -> set[str]:
    """The tokens FTS5's default tokenizer would index `text` as, folded.

    Any code SCORING an FTS5 match has to agree with the tokenizer that
    produced it. Two ways to disagree have already shipped and been fixed:
    diacritics (`Zürich` vs `zurich`) and underscores (`feedback_kairn_first`
    as one token instead of three). Both live here now, in one place, so the
    query side and the document side cannot drift apart again.
    """
    return {_fold_diacritics(tok) for tok in _INDEX_TOKEN_RE.findall(text.lower())}


def _fts_terms(fts_query: str | None) -> list[str]:
    """Recover the individual terms from a `to_fts_query` output.

    `to_fts_query` emits `"a" OR "b" OR "c"`, so the quoted spans are exactly
    the searchable terms. Returns [] for a browse query (None) or a malformed
    string, which callers treat as "no coverage signal available".
    """
    if not fts_query:
        return []
    return re.findall(r'"([^"]+)"', fts_query)


def _term_coverage(terms: list[str], *fields: str | None) -> float:
    """Fraction of distinct query terms that actually occur in `fields`.

    WHY THIS EXISTS. `to_fts_query` joins terms with OR so that ANY keyword can
    match - that is deliberate, and it is what gives Kairn its recall. But bm25
    then scores the document on whatever did match, and the saturating
    transform above only ever sees that aggregate score. It has no way to tell
    a 1-of-6 match from a 6-of-6 one.

    Measured consequence before this fix (12,866-node store, 2026-08-22): the
    query "baroque harpsichord tuning temperament werckmeister" - which has no
    real overlap with the corpus - returned hits at relevance 0.60 by matching
    the single word "tuning" against "autoevolve self-tuning". Genuinely
    relevant queries scored 0.67-0.73. An 0.08 separation band makes
    `min_relevance` decorative and abstention structurally impossible, which is
    exactly what the `_BM25_RELEVANCE_MIDPOINT` docstring above promises it is
    not ("weak keyword overlaps fall under a strict min_relevance floor while
    strong multi-term matches clear it").

    Scaling relevance by coverage implements that promise: match strength times
    how much of the question you actually answered. Ordering within a single
    query is preserved for equal-coverage candidates, and recall is unchanged -
    a partial match is still RETURNED, it is just no longer scored as if it
    were a full one.
    THIS MUST MODEL FTS5's MATCH SEMANTICS, NOT PYTHON's `in`. A literal
    substring test diverges from the thing it is scoring, in BOTH directions,
    and both were measured on a real FTS5 table:

      FALSE NEGATIVE - `unicode61` FOLDS DIACRITICS. A node holding "Zürich
      delegation workflow" is indexed as `zurich`, so the query term "zurich"
      genuinely MATCHES (bm25 rank -1e-06). A literal `"zurich" in "zürich..."`
      is False, so coverage returns 0.0 and the multiplication drives a REAL
      match to relevance 0 - dropped under any positive `min_relevance`. This
      is the case that makes "recall is unchanged" false for accented content.

      FALSE POSITIVE - substring, not token. The term "cat" is NOT matched by
      FTS5 against "Harpsichord repair category notes" (0 rows), but `"cat" in
      "...category..."` is True, so a 1-of-2 match scores 2-of-2 - restoring
      exactly the inflation this function exists to remove.

      THIRD TRAP, SAME FAMILY - `unicode61` SPLITS ON UNDERSCORE, `\\w` DOES NOT.
      `\\w` keeps `_` inside a token, so `feedback_kairn_first` stays one opaque
      token in Python while FTS5 indexes it as `feedback`, `kairn`, `first`.
      Measured on a real FTS5 table: `MATCH '"kairn"'` returns the row, and a
      set-membership test against the `\\w+` token would not. That is the
      dominant name shape in a Kairn store (`feedback_*`, `gotcha_*`,
      `project_*`), and once `context()` started SORTING on coverage it began
      inverting real orderings: a node matching 3 of 3 terms at bm25 -10.75 fell
      below one matching 2 of 3 at -8.73, because its match hid inside an
      underscored identifier. Found by internal review after two external
      lenses had passed the same code.

    So: fold diacritics on both sides, split on the same class `unicode61` uses,
    and compare TOKEN SETS. A query term that itself contains `_` is several
    FTS5 tokens, so it counts as covered when all of its tokens are present.
    (FTS5 would additionally require them ADJACENT, since a quoted multi-token
    term is a phrase query. Not modelled here: it would only ever make coverage
    stricter, and the gap this closes is the one that reorders results.)
    """

    if not terms:
        return 1.0
    want = {frozenset(_index_tokens(t)) for t in terms}
    want.discard(frozenset())
    if not want:
        return 0.0
    have = _index_tokens(" ".join(f for f in fields if f))
    if not have:
        return 0.0
    return sum(1 for tokens in want if tokens <= have) / len(want)
def _allocate_across_sources(
    results: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Take one row from each source in turn, nodes first, up to `limit`.

    THE BUDGET IS THE POINT, NOT THE ORDER. It was written when node
    match-strength and experience time-decay were different scales and the
    union therefore could not be sorted at all; both now carry bm25 match times
    term coverage, and `crossref` does sort them before calling this. The
    allocation is still needed, because sorting fairly does not make the budget
    fair: the other half of the defect was a plain `results[:limit]` with nodes
    appended first, which let the node list eat the whole budget. Measured on
    12 real queries: limit=3 -> 36 nodes / 0 experiences, limit=6 -> 72 / 0;
    experiences only appeared at limit=20.

    Rank INSIDE each group is preserved exactly - this only interleaves two
    already-ordered lists. One empty group still fills the whole budget.

    HONEST LIMITS, because "neither source can starve the other" is not true in
    general: at limit=1 a node wins whenever one exists, which is inherent to
    "nodes first"; and a caller who wanted the `limit` best NODES now gets about
    half that many. This is an allocation POLICY, not a pure correctness fix,
    and it is stated here rather than implied.

    ONE MORE CONSEQUENCE, now that the union arrives sorted: slot 0 is a node
    whenever any node matched, so "the best row is first" does NOT hold at
    `crossref` even though the rows are ranked. That trade was taken
    deliberately - losing an answer to truncation is a real harm, not leading
    with the top row is a cosmetic one - and it is pinned in
    `test_the_best_match_leads_its_own_group` rather than left to be
    rediscovered.

    Rows whose `source` is neither "node" nor "experience" are appended after
    both groups rather than dropped - the previous slice would have kept them.
    """
    nodes_out = [r for r in results if r.get("source") == "node"]
    exps_out = [r for r in results if r.get("source") == "experience"]
    other = [r for r in results if r.get("source") not in ("node", "experience")]
    out: list[dict[str, Any]] = []
    for i in range(max(len(nodes_out), len(exps_out))):
        if i < len(nodes_out):
            out.append(nodes_out[i])
        if i < len(exps_out):
            out.append(exps_out[i])
    return (out + other)[:limit]


class IntelligenceLayer:
    """Unified intelligence operations over graph, experience, and routing."""

    def __init__(
        self,
        *,
        store: StorageBackend,
        event_bus: EventBus,
        graph: GraphEngine,
        router: ContextRouter,
        memory: ProjectMemory,
        experience: ExperienceEngine,
        ideas: IdeaEngine,
        embedder: Callable[[list[str]], list[list[float]]] | None = None,
        embedder_model: str | None = None,
        semantic_recall: bool = False,
        semantic_floor: float = 0.5,
        experience_min_match: float = 0.0,
        semantic_top_n: int = 30,
    ) -> None:
        self.store = store
        self.event_bus = event_bus
        self.graph = graph
        self.router = router
        self.memory = memory
        self.experience = experience
        self.ideas = ideas
        # Optional semantic_recall (opt-in flag, default OFF). When on, recall's
        # node path reranks the FTS5 top-N by local-embedding cosine and abstains
        # below semantic_floor. All-None/False => the keyword path runs unchanged.
        self.embedder = embedder
        self.embedder_model = embedder_model
        self.semantic_recall = semantic_recall
        self.semantic_floor = semantic_floor
        # Abstention floor for the EXPERIENCE path. Default 0.0 keeps recall
        # byte-identical; the node path can abstain while experiences still
        # flood the result set, which is what made the semantic win invisible.
        self.experience_min_match = experience_min_match
        self.semantic_top_n = semantic_top_n

    async def _log_node_access(
        self, activity_type: str, node_ids: list[str]
    ) -> None:
        """Best-effort batch-log of node accesses to activity_log.

        Fires after recall/context/crossref return nodes so downstream
        analytics can track which nodes are actually queried. Failures
        are logged and swallowed to keep the read path fail-open.
        """
        if not node_ids:
            return
        now = datetime.now(UTC).isoformat()
        entries = [
            {
                "id": str(uuid.uuid4())[:8],
                "user_id": None,
                "activity_type": activity_type,
                "entity_type": "node",
                "entity_id": nid,
                "description": None,
                "created_at": now,
            }
            for nid in node_ids
        ]
        try:
            await self.store.log_activities(entries)
        except Exception:
            logger.debug("Failed to log node access for %s", activity_type, exc_info=True)

    async def learn(
        self,
        *,
        content: str,
        type: str,
        context: str | None = None,
        confidence: str = "high",
        tags: list[str] | None = None,
        namespace: str = "knowledge",
        with_candidates: bool = True,
    ) -> dict[str, Any]:
        """Store knowledge from conversation.

        High confidence creates a permanent node + experience.
        Medium/low confidence creates a decaying experience only.

        The `namespace` parameter isolates knowledge across tenants/projects.
        It is applied to both the high-confidence graph node and the
        backing experience record.

        When `with_candidates=True` (default), runs a follow-up FTS5 scan
        over existing nodes using the saved content as the seed query.
        Returns up to `_CANDIDATES_LIMIT` semantically-related node
        snippets in the response envelope under the `candidates` key so
        the caller can decide whether to invoke `kn_judge` to assert a
        relationship verb (conflicts_with / supersedes / compatible /
        scoped / related). The just-created node (if any) is excluded
        from candidates. Set `with_candidates=False` for high-volume
        bulk-save scripts that do not need the judgment hook.
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")
        if type not in VALID_TYPES:
            raise ValueError(f"Invalid type: {type}. Must be one of {VALID_TYPES}")
        if confidence not in VALID_CONFIDENCES:
            raise ValueError(
                f"Invalid confidence: {confidence}. Must be one of {VALID_CONFIDENCES}"
            )

        content = content.strip()
        node_id: str | None = None
        experience_id: str | None = None

        if confidence == "high":
            # Create permanent node
            node = await self.graph.add_node(
                name=f"{type.capitalize()}: {content[:60]}",
                type=f"learned_{type}",
                namespace=namespace,
                description=content,
                tags=tags,
                source_type="learn",
            )
            node_id = node.id

            # Update routes for discoverability
            await self.router.update_routes_for_node(node.id, node.name, node.description)

        # Always create experience (for decay tracking)
        exp = await self.experience.save(
            content=content,
            type=type,
            context=context,
            confidence=confidence,
            tags=tags,
            namespace=namespace,
        )
        experience_id = exp.id

        stored_as = "node" if confidence == "high" else "experience"

        await self.event_bus.emit(
            EventType.KNOWLEDGE_LEARNED,
            {
                "stored_as": stored_as,
                "node_id": node_id,
                "experience_id": experience_id,
                "type": type,
                "confidence": confidence,
            },
        )

        logger.info(
            "Learned %s (confidence=%s, stored_as=%s)",
            type,
            confidence,
            stored_as,
        )

        response: dict[str, Any] = {
            "_v": "1.0",
            "stored_as": stored_as,
            "node_id": node_id,
            "experience_id": experience_id,
            "type": type,
            "confidence": confidence,
            "namespace": namespace,
        }

        if with_candidates:
            response["candidates"] = await self._scan_candidates(
                content=content,
                exclude_node_id=node_id,
                namespace=namespace,
            )

        return response

    async def _scan_candidates(
        self,
        *,
        content: str,
        exclude_node_id: str | None,
        namespace: str,
    ) -> list[dict[str, Any]]:
        """FTS5-scan for semantically-related existing nodes.

        Mirrors the Phase 0 benchmark query shape (validated p95 1.258ms
        at 4923-node scale). Used by `learn()` to surface judgment
        candidates without forcing the caller to issue a separate
        `kn_context` / `kn_recall` query.

        Returns up to `_CANDIDATES_LIMIT` candidate dicts, each shaped
        `{id, name, type, snippet, sim_rank}`. `snippet` is the node
        description truncated to `_CANDIDATE_SNIPPET_CHARS`; `sim_rank`
        is the integer position 0..N-1 in FTS5 rank order (lower is
        more relevant). Empty list when no FTS5 hits or the content
        produced no usable keywords.
        """
        fts_query = _to_fts_query(content)
        if not fts_query:
            return []

        # Over-fetch by one so we can drop the just-created node and
        # still return up to _CANDIDATES_LIMIT.
        # Fail-open: the save already persisted (lines above). A scan
        # error must not abort the caller, otherwise a retry would
        # create a duplicate node. Mirrors GraphEngine._auto_link
        # guard at graph.py.
        try:
            nodes = await self.graph.query(
                text=fts_query,
                namespace=namespace,
                limit=_CANDIDATES_LIMIT + 1,
            )
        except (OSError, RuntimeError, sqlite3.Error):
            logger.warning(
                "FTS5 candidate scan failed for content (len=%d); returning []",
                len(content),
                exc_info=True,
            )
            return []

        candidates: list[dict[str, Any]] = []
        for node in nodes:
            if exclude_node_id and node.id == exclude_node_id:
                continue
            description = node.description or ""
            snippet = description[:_CANDIDATE_SNIPPET_CHARS]
            if len(description) > _CANDIDATE_SNIPPET_CHARS:
                snippet += "..."
            candidates.append(
                {
                    "id": node.id,
                    "name": node.name,
                    "type": node.type,
                    "snippet": snippet,
                    "sim_rank": len(candidates),
                }
            )
            if len(candidates) >= _CANDIDATES_LIMIT:
                break
        return candidates

    def _node_result(
        self, *, node_id, name, type_, namespace, description, relevance, relevance_kind
    ):
        # namespace travels in every item shape so downstream namespace-based
        # access filters can enforce their allowlists on this surface.
        #
        # relevance_kind is a REQUIRED argument, not a default. This factory is a
        # funnel for three different numbers - bm25, embedding cosine, and the
        # literal 1.0 a text-less browse produces - and an earlier version
        # hardcoded MATCH here, which stamped "lexical match strength" on all
        # three. A default would have hidden the next one the same way.
        assert relevance_kind in RELEVANCE_KINDS, relevance_kind
        return {
            "source": "node",
            "id": node_id,
            "name": name,
            "type": type_,
            "namespace": namespace,
            "description": description,
            "relevance": relevance,
            "relevance_kind": relevance_kind,
        }

    async def _keyword_node_recall(
        self, *, fts_query: str | None, limit: int, min_relevance: float
    ) -> list[dict[str, Any]]:
        """Keyword node path: FTS5 bm25 relevance, min_relevance gate. This is
        the default recall for nodes (semantic_recall OFF)."""
        if fts_query:
            ranked = await self.graph.query_ranked(
                text=fts_query, limit=_rerank_pool(limit)
            )
        else:
            ranked = await self.graph.query_ranked(limit=limit)
        terms = _fts_terms(fts_query)
        out: list[dict[str, Any]] = []
        for node, rank in ranked:
            # The same computation this used to inline, moved into one helper
            # because `crossref` needs it too and had a constant 1.0 instead.
            # It also drops the intermediate 4-decimal round: on a small store
            # that round collapses every match to 0.0, at which point a sort by
            # the value is insertion order wearing a ranking's name.
            relevance = _node_relevance(rank, terms, node.name, node.description)
            if relevance < min_relevance:
                continue
            # A text-less query has no rank, and _bm25_to_relevance(None)
            # returns the literal 1.0. That is the same non-score crossref
            # reports, so it gets the same label - calling it "match" would
            # advertise a perfect lexical hit on a query with no text in it.
            out.append(
                self._node_result(
                    node_id=node.id,
                    name=node.name,
                    type_=node.type,
                    namespace=node.namespace,
                    description=node.description,
                    relevance=relevance,
                    relevance_kind=(
                        RELEVANCE_KIND_MATCH if rank is not None else RELEVANCE_KIND_UNSCORED
                    ),
                )
            )
        # The reported score IS the order. `query_ranked` hands back raw bm25
        # order, but the score reported here is bm25 times term coverage, and
        # those two orders disagree exactly when coverage does its job: a
        # one-word note holding only the rarest query term scores 0.387 on raw
        # bm25 against 0.175 for the document that answers all four terms, and
        # arrived FIRST while reporting the lower number. Found by a mutation
        # control - deleting `term_coverage` from the node score changed
        # nothing any test could see, because nothing ordered by it.
        out.sort(key=lambda r: r["relevance"], reverse=True)
        # Truncate AFTER the rerank, never before - see `_rerank_pool`.
        return out[:limit]

    async def _semantic_node_recall(
        self, topic: str, fts_query: str, limit: int, min_relevance: float = 0.0
    ) -> list[dict[str, Any]]:
        """Semantic node path (semantic_recall ON): rerank the FTS5 top-N by
        local-embedding cosine and abstain below semantic_floor.

        Uses each candidate's stored vector when its model matches the live
        embedder; embeds any missing/stale candidate on the fly (correctness
        over speed - works before a backfill). Fail-open: any embedding error
        falls back to the keyword node path so recall never crashes and never
        silently returns nothing because of an embedder outage."""
        from kairn.core.embeddings import (
            cosine,
            node_embedding_text,
            normalize,
            unpack_vector,
        )

        candidates = await self.store.query_nodes_with_embeddings(
            text=fts_query, limit=self.semantic_top_n
        )
        if not candidates:
            return []
        loop = asyncio.get_running_loop()
        try:
            query_vectors = await loop.run_in_executor(None, self.embedder, [topic])
        except Exception:
            logger.warning(
                "semantic recall query-embed failed; falling back to keyword", exc_info=True
            )
            return await self._keyword_node_recall(
                fts_query=fts_query, limit=limit, min_relevance=min_relevance
            )
        if not query_vectors or not query_vectors[0]:
            return []
        qvec = normalize(query_vectors[0])

        scored: list[tuple[float, dict[str, Any]]] = []
        missing: list[dict[str, Any]] = []
        for row in candidates:
            blob = row.get("embedding")
            if blob and row.get("embedding_model") == self.embedder_model:
                scored.append((cosine(qvec, unpack_vector(blob)), row))
            else:
                missing.append(row)
        if missing:
            texts = [
                node_embedding_text(row.get("name"), row.get("description"))
                for row in missing
            ]
            try:
                fresh = await loop.run_in_executor(None, self.embedder, texts)
            except Exception:
                logger.warning("semantic recall candidate-embed failed", exc_info=True)
                fresh = []
            for row, vec in zip(missing, fresh, strict=False):
                if vec:
                    scored.append((cosine(qvec, normalize(vec)), row))

        scored.sort(key=lambda item: item[0], reverse=True)
        # A caller's min_relevance still tightens the node gate (it can only
        # raise the floor, never lower it), so kn_recall(min_relevance=...)
        # affects nodes as it does experiences instead of being silently
        # ignored on the semantic path. Default min_relevance=0 => the cosine
        # floor alone decides.
        effective_floor = max(min_relevance, self.semantic_floor)
        out: list[dict[str, Any]] = []
        for score, row in scored:
            if score < effective_floor:
                continue
            out.append(
                self._node_result(
                    node_id=row["id"],
                    name=row["name"],
                    type_=row["type"],
                    namespace=row["namespace"],
                    description=row.get("description"),
                    relevance=round(score, 4),
                    # Embedding cosine, not bm25. A different scale and
                    # corpus-independent, so it does not share MATCH's label.
                    relevance_kind=RELEVANCE_KIND_SIMILARITY,
                )
            )
            if len(out) >= limit:
                break
        return out

    async def recall(
        self,
        *,
        topic: str | None = None,
        limit: int = 10,
        min_relevance: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Surface relevant past knowledge for context.

        Searches across both nodes (permanent) and experiences (decaying).
        Returns combined, ranked results.
        """
        results: list[dict[str, Any]] = []
        fts_query = _to_fts_query(topic) if topic else None
        # THREE STATES, NOT TWO. `_to_fts_query` returns None both when NO
        # topic was given (a deliberate browse) and when a topic was given
        # whose every word is a stop word (a question with nothing searchable
        # in it). The node path reads None as "browse", so the second case
        # answered an unanswerable question with the most recent nodes at
        # relevance 1.0 - `bm25_to_relevance(None)` is a REPORTING contract
        # ("no match to report") being read as a perfect score, one file over
        # from where the same defect was removed from the experience gate.
        # Measured at the SHIPPED DEFAULT, so no floor could have caught it:
        # recall("is it ok") returned 0 experiences and 5 unrelated nodes at
        # 1.0. The experience half took a different fix (raw text, because
        # ExperienceEngine shapes text itself); the node path genuinely needs
        # the pre-shaped query, since GraphEngine.query_ranked forwards it
        # straight to `nodes_fts MATCH ?`. So the discrimination belongs here.
        asked_but_unsearchable = bool(topic) and not fts_query

        # Node path. Default = keyword (honest bm25 relevance + min_relevance
        # gate). With the opt-in semantic_recall flag on AND a query present,
        # rerank the FTS5 top-N by local-embedding cosine and abstain below the
        # cosine floor. Flag OFF runs the keyword path unchanged.
        if self.semantic_recall and self.embedder is not None and fts_query and topic:
            node_results = await self._semantic_node_recall(
                topic, fts_query, limit, min_relevance
            )
        elif asked_but_unsearchable:
            # A question was asked and nothing searchable survived it. The
            # honest answer is nothing, not the newest rows (Kairn 8a3a11af).
            node_results = []
        else:
            node_results = await self._keyword_node_recall(
                fts_query=fts_query, limit=limit, min_relevance=min_relevance
            )
        results.extend(node_results)
        kept_node_ids = [r["id"] for r in node_results]

        # NOTE: access logging and touch_accessed now happen AFTER allocation,
        # over the rows the caller actually receives. See the block below.

        # Search experiences (decay-aware)
        experiences = await self.experience.search(
            # RAW TEXT, not the shaped query. `_to_fts_query` maps BOTH "no
            # topic was given" and "a topic whose words are all stop words"
            # onto None, and None is the engine's BROWSE signal - so an
            # unanswerable question was answered with the most recent rows.
            # Measured: `recall("is it ok")` returned 5 unrelated experiences,
            # AT THE DEFAULT FLOOR, so no abstention gate could ever have
            # closed it (Kairn `00f674bd` reproduced it at 0.65 and read the
            # cause one layer too low). `ExperienceEngine.search` shapes raw
            # text itself and returns [] when no keyword survives, and
            # re-shaping an already-shaped query is idempotent by its own
            # contract, so passing the raw text is both safe and the only
            # place the two meanings can be told apart.
            text=topic,
            min_relevance=min_relevance,
            min_match=self.experience_min_match,
            limit=limit,
        )

        now = datetime.now(UTC)
        for exp in experiences:
            results.append(
                {
                    "source": "experience",
                    "id": exp.id,
                    "type": exp.type,
                    "namespace": exp.namespace,
                    "content": exp.content,
                    "confidence": exp.confidence,
                    "relevance": _reported_relevance(exp, now),
                    "relevance_kind": exp.reported_relevance_kind(),
                }
            )

        # Nodes (curated, permanent) lead, then experiences (decaying). Each
        # group is already ranked internally - nodes by bm25 match strength
        # times term coverage, experiences by the experience engine's search
        # order. We deliberately do NOT merge-sort the union here. The original
        # reason was that the two were different SCALES - node match strength
        # against experience time-decay - and that reason is now gone: both
        # sides carry bm25 match times coverage. What remains is the reason
        # that outlives it. `recall` is the surface that must not bury curated
        # nodes under fresh experiences, so the two groups keep their own order
        # and the budget is split between them. `crossref` DOES sort the union,
        # and may, because there nothing has to lead.
        #
        # BUT REFUSING TO MERGE-SORT IS ONLY HALF THE JOB, and the other half
        # was a plain `results[:limit]`. Nodes are appended first, so at the
        # limits callers actually use the node list ate the entire budget and
        # NO experience was ever returned. Measured against a real store on 12
        # queries: limit=3 -> 36 nodes / 0 experiences; limit=6 -> 72 / 0;
        # experiences only start appearing at limit=20. A caller asking for 6
        # "results across nodes and experiences" got one source, silently.
        #
        # So the budget is ALLOCATED rather than consumed: take one from each
        # group in turn, nodes first, until `limit` is reached. Rank inside each
        # group is preserved exactly (this only interleaves two already-ordered
        # lists), neither source can starve the other, and when one group is
        # empty the other fills the whole budget as before. No score from one
        # scale is ever compared against a score from the other.
        results = _allocate_across_sources(results, limit)

        # CREDIT ONLY WHAT THE CALLER ACTUALLY RECEIVED.
        #
        # Both sub-queries fetch `limit` rows, but the allocation shows at most
        # about half of each. Logging and touching the full fetch credited rows
        # nobody saw - and `exp_auto_promote` fires on `access_count`, so at
        # limit=10 five unseen experiences were pushed toward promotion on every
        # single call. The node access feed drives the same decay/promotion
        # pipeline. Both now run over `results`, after truncation.
        kept_node_ids = [r["id"] for r in results if r.get("source") == "node"]
        if kept_node_ids:
            await self._log_node_access("node_recall", kept_node_ids)
        kept_exp_ids = [r["id"] for r in results if r.get("source") == "experience"]
        if kept_exp_ids:
            await self.experience.touch_accessed(kept_exp_ids)
            kept = set(kept_exp_ids)
            for exp in experiences:
                if exp.id in kept:
                    exp.access_count += 1

        await self.event_bus.emit(
            EventType.KNOWLEDGE_RECALLED,
            {"topic": topic, "result_count": len(results)},
        )

        return results

    async def crossref(
        self,
        *,
        problem: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find similar solutions from current workspace.

        In V1 (single workspace), searches nodes and experiences.
        In team mode, will search across accessible workspaces.
        """
        if not problem or not problem.strip():
            raise ValueError("Problem description cannot be empty")

        problem = problem.strip()
        fts_query = _to_fts_query(problem)
        results: list[dict[str, Any]] = []

        # Search nodes for solutions/patterns. `query_ranked` is the same
        # store query as `query` with the bm25 rank additionally exposed, so
        # the row set is unchanged - what changes is that the node now carries
        # a MEASUREMENT instead of the constant 1.0 it used to be labelled
        # with. The sort below compares these against experience scores, and a
        # placeholder in a comparison is not a tie-break, it is a guaranteed
        # win: three nodes sharing one query term out-ranked a verbatim 4-of-4
        # experience and the `[:limit]` truncation then dropped it (Kairn
        # `00f674bd`). Both sides are now bm25 match times term coverage.
        terms = _fts_terms(fts_query)
        if fts_query:
            ranked = await self.graph.query_ranked(
                text=fts_query, limit=_rerank_pool(limit)
            )
        else:
            ranked = []
        for node, rank in ranked:
            results.append(
                {
                    "source": "node",
                    "workspace": "default",
                    "id": node.id,
                    "name": node.name,
                    "type": node.type,
                    "namespace": node.namespace,
                    "description": node.description,
                    # This used to be the constant 1.0, labelled UNSCORED
                    # because `graph.query` returned nodes in its own order and
                    # there was no score to report. The query above is now
                    # `query_ranked`, so there IS one: bm25 match strength
                    # scaled by term coverage, the same quantity `recall` uses.
                    # The placeholder was not a tie-break but a guaranteed win
                    # - three nodes sharing one query term out-ranked a
                    # verbatim 4-of-4 experience and the truncation dropped it
                    # (Kairn `00f674bd`). A measurement is what removes that.
                    "relevance": _node_relevance(
                        rank, terms, node.name, node.description
                    ),
                    "relevance_kind": RELEVANCE_KIND_MATCH,
                }
            )

        # Access logging moved below, over the allocated rows only.

        # Search experiences for solutions
        experiences = await self.experience.search(
            # RAW TEXT, not the shaped query. `_to_fts_query` maps BOTH "no
            # topic was given" and "a topic whose words are all stop words"
            # onto None, and None is the engine's BROWSE signal - so an
            # unanswerable question was answered with the most recent rows.
            # Measured: `recall("is it ok")` returned 5 unrelated experiences,
            # AT THE DEFAULT FLOOR, so no abstention gate could ever have
            # closed it (Kairn `00f674bd` reproduced it at 0.65 and read the
            # cause one layer too low). `ExperienceEngine.search` shapes raw
            # text itself and returns [] when no keyword survives, and
            # re-shaping an already-shaped query is idempotent by its own
            # contract, so passing the raw text is both safe and the only
            # place the two meanings can be told apart.
            text=problem,
            min_relevance=0.1,
            min_match=self.experience_min_match,
            limit=limit,
        )

        now = datetime.now(UTC)
        for exp in experiences:
            results.append(
                {
                    "source": "experience",
                    "workspace": "default",
                    "id": exp.id,
                    "type": exp.type,
                    "namespace": exp.namespace,
                    "content": exp.content,
                    "confidence": exp.confidence,
                    "relevance": _reported_relevance(exp, now),
                    "relevance_kind": exp.reported_relevance_kind(),
                }
            )

        # A SINGLE SORT ON `relevance` IS LEGITIMATE NOW, AND ONLY NOW. It was
        # replaced by an explicit nodes-then-experiences key precisely because
        # a node carried the literal 1.0 and an experience carried decay: two
        # numbers sharing a range and nothing else, where the placeholder
        # always won. Both sides now carry bm25 match strength times term
        # coverage, at one resolution (`_RELEVANCE_DECIMALS`); experiences
        # additionally carry the bounded +-10% recency nudge, which cannot
        # promote a weak fresh hit over a strong old one. The condition that
        # made the two-level key necessary is gone, so the honest key is the
        # measurement.
        results.sort(key=lambda r: r["relevance"], reverse=True)
        # SAME DEFECT AS recall(), one function away - measured side by side on
        # one store with 10 matching nodes and 10 matching experiences at
        # limit=6:  recall -> 3 nodes / 3 experiences,  crossref -> 6 / 0.
        # Sorting fairly does not make the budget fair; that needs allocating.
        results = _allocate_across_sources(results, limit)

        # Credit only what the caller received - see the note in recall(). This
        # replaces a node-only version of the same block that used to sit after
        # the truncation: the node pool is over-fetched for the rerank
        # (`_rerank_pool`), so crediting before this point charged ~30 nodes for
        # a query that returned 3, and the experience half was missing entirely
        # while `exp_auto_promote` fires on `access_count`.
        kept_node_ids = [r["id"] for r in results if r.get("source") == "node"]
        if kept_node_ids:
            await self._log_node_access("node_crossref", kept_node_ids)
        kept_exp_ids = [r["id"] for r in results if r.get("source") == "experience"]
        if kept_exp_ids:
            await self.experience.touch_accessed(kept_exp_ids)
            kept = set(kept_exp_ids)
            for exp in experiences:
                if exp.id in kept:
                    exp.access_count += 1

        await self.event_bus.emit(
            EventType.CROSSREF_FOUND,
            {"problem": problem, "result_count": len(results)},
        )

        return results

    async def context(
        self,
        *,
        keywords: str,
        detail: str = "summary",
        limit: int = 10,
    ) -> dict[str, Any]:
        """Get relevant context subgraph with progressive disclosure.

        Combines router-based node discovery with experience search.

        BEHAVIOUR CHANGE (node ranking): `nodes` used to arrive in the router's
        own order, which on a real store is arbitrary - see the long comment
        below. They now arrive best-match first and each carries `relevance` +
        `relevance_kind` alongside the unchanged `confidence`. A caller that
        relied on the previous order relied on luck; a caller that reads
        `nodes[0]` gets a better node than before. No field was removed.

        `relevance` on a node is bm25 match strength scaled by query-term
        coverage. `relevance` on an experience is that SAME quantity with a
        bounded +-10% recency nudge when the query had text, and pure time
        decay only on a browse - which is what `relevance_kind` now reports as
        `match` against `match_recency` or `recency`. The two lists are still
        returned separately rather than merge-sorted: this method's contract is
        two named lists, and collapsing them would be a wire change, not a
        ranking improvement.
        """
        if not keywords or not keywords.strip():
            return {
                "_v": "1.0",
                "query": keywords,
                "detail": detail,
                "count": 0,
                "nodes": [],
                "experiences": [],
            }

        keywords = keywords.strip()
        fts_query = _to_fts_query(keywords)

        # NODE DISCOVERY, THEN NODE RANKING. These are two steps and the second
        # one used to be missing.
        #
        # Discovery is the context router: keywords -> routes -> candidate ids.
        # What the router CANNOT do is order them. Every route in a real store
        # carries the same confidence:
        #     sqlite3 kairn.db "select confidence, count(*) from routes
        #                       group by confidence"
        #     -> 0.5|23353        (one row - every route in the store)
        # so `min_confidence` excludes nothing at its default and the old code's
        # `route(limit=limit)` returned an ARBITRARY `limit` of the candidates.
        # On 40 real queries against a 12.9k-node store the median candidate
        # pool was 510 live nodes. Ten were returned. Which ten was luck.
        #
        # Ranking them is a bm25 pass restricted to exactly those ids. The
        # restriction is the point: an EARLIER ATTEMPT ranked the candidates by
        # intersecting them with a GLOBAL top-K (`query_ranked(limit=limit*4)`)
        # and called everything outside that window `unscored`. That number was
        # a property of the window, not of the data - same 12 queries, 72 nodes:
        #     fts limit    20 ->  9 matched, 63 "unscored"   12.5%
        #     fts limit   100 -> 29 matched, 43 "unscored"   40.3%
        #     fts limit  1000 -> 61 matched, 11 "unscored"   84.7%
        #     fts limit 20000 -> 72 matched,  0 "unscored"  100.0%
        # Restricting the MATCH to the candidate ids has no window: every
        # candidate is scored or provably does not match.
        #
        # THE COST IS LINEAR IN THE POOL, AND THE POOL IS NOT BOUNDED. Measured
        # end to end through the CLI on that store: median 177 -> 203 ms, and
        # 231 -> 402 ms on the largest pool observed (3350 candidates). A route
        # array grows with the store, so a caller on a hot path with a hard
        # timeout should size that timeout against its own worst pool, not the
        # median.
        #
        # CAPPING THE POOL IS NOT AVAILABLE HERE, and it is worth saying why
        # rather than leaving it to look like an oversight: the obvious cap is
        # "rank the best N candidates by route confidence", but route confidence
        # is the constant this whole path exists to work around. "Best N" would
        # be an arbitrary N, which is the defect, not the fix.
        # THE EXACT BOUND, if this ever becomes binding: coverage is a
        # multiplier in [0, 1], so a candidate's final score can never exceed
        # its raw bm25 relevance. Fetch the restricted set ORDER BY rank in
        # batches and stop as soon as the `limit`-th best SCALED score is at or
        # above the next unfetched row's RAW relevance - proven complete, no
        # window. Measured as worth ~75ms of ~150ms, which did not justify an
        # adaptive loop in a published package on the day the ranking landed.
        #
        # Ranking is by the SAME scale the recall path uses (`_bm25_to_relevance`
        # scaled by `_term_coverage`), so a node's `relevance` here means what it
        # means there. It is NOT comparable to an experience's `relevance` in
        # this same payload - that one is a time-decay score, which is exactly
        # what `relevance_kind` is there to say. Do not sort the two together.
        candidates = await self.router.route_candidates(keywords)
        terms = _fts_terms(fts_query)
        nodes: list[dict[str, Any]] = []

        def _node_out(
            node_data: Mapping[str, Any],
            confidence: float,
            relevance: float,
            relevance_kind: str,
        ) -> dict[str, Any]:
            out: dict[str, Any] = {
                "id": node_data["id"],
                "name": node_data["name"],
                "type": node_data["type"],
                "namespace": node_data.get("namespace"),
                "confidence": confidence,
                "relevance": relevance,
                "relevance_kind": relevance_kind,
            }
            if detail != "summary":
                out["description"] = node_data.get("description")
                out["tags"] = node_data.get("tags")
                out["properties"] = node_data.get("properties")
            return out

        if candidates and fts_query:
            ranked = await self.graph.query_ranked(
                text=fts_query,
                node_ids=list(candidates),
                limit=len(candidates),
            )
            scored: list[tuple[float, Node]] = []
            for node, rank in ranked:
                relevance = _bm25_to_relevance(rank)
                if terms:
                    relevance = round(
                        relevance * _term_coverage(terms, node.name, node.description), 4
                    )
                scored.append((relevance, node))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            for relevance, node in scored[:limit]:
                nodes.append(
                    _node_out(
                        node.model_dump(),
                        candidates.get(node.id, 0.0),
                        relevance,
                        RELEVANCE_KIND_MATCH,
                    )
                )

        # Top up from the routed candidates when ranking could not fill the
        # slots. They are REAL routed nodes and dropping them would cost recall,
        # so they are returned last, at relevance 0.0, labelled `unscored`.
        #
        # `unscored` MEANS "NO SCORE AVAILABLE", NOT "NO LEXICAL MATCH", and the
        # difference matters to a consumer deciding whether to abstain. Two
        # states land here:
        #   * the candidate did not match the text
        #   * no FTS query was built at all, so nothing was scored. The router
        #     and `to_fts_query` use DIFFERENT stop-word sets, so `keywords="need
        #     about"` routes to real candidates while `to_fts_query` returns
        #     None - and a node whose text contains both words comes back
        #     `unscored`. Reading that as "the store provably has no match" is
        #     wrong.
        # This is the same meaning `_keyword_node_recall` gives the label for a
        # text-less query, so the two paths agree. An earlier version of this
        # comment claimed the label meant one thing only; internal review
        # produced the counterexample above.
        if len(nodes) < limit:
            already = {n["id"] for n in nodes}
            for r in await self.router.take_live(
                candidates, limit=limit - len(nodes), skip=already
            ):
                nodes.append(_node_out(r["node"], r["confidence"], 0.0, RELEVANCE_KIND_UNSCORED))

        # Also search by FTS5 if the router found nothing.
        #
        # This path has NO candidate set to restrict to, so unlike the ranked
        # path above it genuinely is a top-K of the whole corpus - `limit` rows
        # in raw bm25 order. Coverage scaling can only REORDER those rows, never
        # pull in a row that bm25 ranked lower, so which rows appear here does
        # depend on `limit`. Say so rather than implying the same completeness.
        # The re-sort is not optional: without it the returned order is raw bm25
        # while the reported `relevance` is the coverage-scaled score, and the
        # two disagree (external review, first version of this method).
        if not nodes and fts_query:
            ranked = await self.graph.query_ranked(text=fts_query, limit=limit)
            fallback: list[tuple[float, Node]] = []
            for n, rank in ranked:
                relevance = _bm25_to_relevance(rank)
                if terms:
                    relevance = round(
                        relevance * _term_coverage(terms, n.name, n.description), 4
                    )
                fallback.append((relevance, n))
            fallback.sort(key=lambda pair: pair[0], reverse=True)
            for relevance, n in fallback:
                nodes.append(_node_out(n.model_dump(), 0.5, relevance, RELEVANCE_KIND_MATCH))

        # Log node access for activity tracking
        if nodes:
            await self._log_node_access(
                "node_context", [n["id"] for n in nodes]
            )

        # Experience search. The abstention floor is forwarded HERE too:
        # `kn_context` is a primary daily surface, and a floor that reaches
        # `recall` alone means the same query abstains on one surface and
        # floods on another (reproduced at 0.65 in Kairn `00f674bd` - recall
        # 0 experiences, crossref and context 2).
        experiences = await self.experience.search(
            # RAW TEXT, not the shaped query. `_to_fts_query` maps BOTH "no
            # topic was given" and "a topic whose words are all stop words"
            # onto None, and None is the engine's BROWSE signal - so an
            # unanswerable question was answered with the most recent rows.
            # Measured: `recall("is it ok")` returned 5 unrelated experiences,
            # AT THE DEFAULT FLOOR, so no abstention gate could ever have
            # closed it (Kairn `00f674bd` reproduced it at 0.65 and read the
            # cause one layer too low). `ExperienceEngine.search` shapes raw
            # text itself and returns [] when no keyword survives, and
            # re-shaping an already-shaped query is idempotent by its own
            # contract, so passing the raw text is both safe and the only
            # place the two meanings can be told apart.
            text=keywords,
            min_relevance=0.1,
            min_match=self.experience_min_match,
            limit=limit,
        )

        # Batch-increment access_count for all returned experiences.
        if experiences:
            await self.experience.touch_accessed([e.id for e in experiences])
            for exp in experiences:
                exp.access_count += 1

        now = datetime.now(UTC)
        exp_items = []
        for e in experiences:
            exp_out: dict[str, Any] = {
                "id": e.id,
                "type": e.type,
                "namespace": e.namespace,
                "content": e.content[:200] if detail == "summary" else e.content,
                "relevance": _reported_relevance(e, now),
                "relevance_kind": e.reported_relevance_kind(),
            }
            if detail != "summary":
                exp_out["confidence"] = e.confidence
                exp_out["tags"] = e.tags
                exp_out["context"] = e.context
            exp_items.append(exp_out)

        total_count = len(nodes) + len(exp_items)

        return {
            "_v": "1.0",
            "query": keywords,
            "detail": detail,
            "count": total_count,
            "nodes": nodes,
            "experiences": exp_items,
        }

    async def related(
        self,
        *,
        node_id: str,
        depth: int = 1,
        edge_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Find nodes connected to a starting point via BFS."""
        return await self.graph.get_related(node_id, depth=depth, edge_type=edge_type)
