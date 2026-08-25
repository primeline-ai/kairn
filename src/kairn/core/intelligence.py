"""Intelligence layer — learn, recall, crossref, context, related.

Bridges graph, experience, and router engines into unified knowledge operations.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import unicodedata
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from kairn.core.experience import ExperienceEngine
from kairn.core.fts import _to_fts_query  # used here + re-exported for back-compat
from kairn.core.graph import GraphEngine
from kairn.core.ideas import IdeaEngine
from kairn.core.memory import ProjectMemory
from kairn.core.router import ContextRouter
from kairn.events.bus import EventBus
from kairn.events.types import EventType
from kairn.models.experience import VALID_CONFIDENCES, VALID_TYPES
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

# bm25 score at which node relevance = 0.5. Larger => the same bm25 match maps
# to a lower relevance, so weak keyword overlaps fall under a strict
# min_relevance floor while strong multi-term matches clear it.
_BM25_RELEVANCE_MIDPOINT = 5.0


def _bm25_to_relevance(rank: float | None) -> float:
    """Map an FTS5 bm25 `rank` to a bounded (0, 1] relevance.

    SQLite FTS5 exposes bm25 as a negative score where a more-negative value
    means a stronger match. A saturating transform (score / (score + K))
    preserves the raw bm25 ordering while yielding an absolute-ish relevance
    the min_relevance gate can act on. `rank is None` (a non-text browse query
    with no MATCH) has no match strength to report, so it stays 1.0.
    """
    if rank is None:
        return 1.0
    score = max(0.0, -float(rank))
    return round(score / (score + _BM25_RELEVANCE_MIDPOINT), 4)


def _fold_diacritics(text: str) -> str:
    """Strip combining marks, the way FTS5's default `unicode61` tokenizer does.

    `unicode61` folds diacritics when it indexes, so `Zürich` is stored as
    `zurich` and an ASCII query for `zurich` matches it. Any code comparing a
    query term against raw text has to fold too, or it disagrees with the index
    it is scoring.
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)
    )


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

    So: fold diacritics on both sides, and compare TOKEN SETS built by the same
    shared tokenizer the query came from.
    """

    if not terms:
        return 1.0
    want = {_fold_diacritics(t.lower()) for t in terms}
    have = {
        _fold_diacritics(tok)
        for tok in re.findall(r"\w+", " ".join(f for f in fields if f).lower())
    }
    if not have:
        return 0.0
    return sum(1 for term in want if term in have) / len(want)
def _allocate_across_sources(
    results: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Take one row from each source in turn, nodes first, up to `limit`.

    REFUSING TO MERGE-SORT IS ONLY HALF THE JOB. Node bm25 match-strength and
    experience time-decay are different scales, so the union is deliberately not
    sorted on `relevance` - but the other half was a plain `results[:limit]`
    with nodes appended first, which let the node list eat the whole budget.
    Measured on 12 real queries: limit=3 -> 36 nodes / 0 experiences,
    limit=6 -> 72 / 0; experiences only appeared at limit=20.

    Rank INSIDE each group is preserved exactly - this only interleaves two
    already-ordered lists. One empty group still fills the whole budget.

    HONEST LIMITS, because "neither source can starve the other" is not true in
    general: at limit=1 a node wins whenever one exists, which is inherent to
    "nodes first"; and a caller who wanted the `limit` best NODES now gets about
    half that many. This is an allocation POLICY, not a pure correctness fix,
    and it is stated here rather than implied.

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
            ranked = await self.graph.query_ranked(text=fts_query, limit=limit)
        else:
            ranked = await self.graph.query_ranked(limit=limit)
        terms = _fts_terms(fts_query)
        out: list[dict[str, Any]] = []
        for node, rank in ranked:
            relevance = _bm25_to_relevance(rank)
            if terms:
                relevance = round(
                    relevance * _term_coverage(terms, node.name, node.description), 4
                )
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
        return out

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

        # Node path. Default = keyword (honest bm25 relevance + min_relevance
        # gate). With the opt-in semantic_recall flag on AND a query present,
        # rerank the FTS5 top-N by local-embedding cosine and abstain below the
        # cosine floor. Flag OFF runs the keyword path unchanged.
        if self.semantic_recall and self.embedder is not None and fts_query and topic:
            node_results = await self._semantic_node_recall(
                topic, fts_query, limit, min_relevance
            )
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
            text=fts_query,
            min_relevance=min_relevance,
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
                    "relevance": round(exp.relevance(at=now), 4),
                    "relevance_kind": RELEVANCE_KIND_RECENCY,
                }
            )

        # Nodes (curated, permanent) lead, then experiences (decaying). Each
        # group is already ranked internally - nodes by bm25 match strength
        # (query_ranked order), experiences by the experience engine's search
        # order. We deliberately do NOT merge-sort the union by a single
        # "relevance" float: node bm25 match-strength and experience time-decay
        # are different scales, and sorting them together buries curated nodes
        # under fresh (high-decay-relevance) experiences.
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

        # Search nodes for solutions/patterns
        if fts_query:
            nodes = await self.graph.query(text=fts_query, limit=limit)
        else:
            nodes = []
        for node in nodes:
            results.append(
                {
                    "source": "node",
                    "workspace": "default",
                    "id": node.id,
                    "name": node.name,
                    "type": node.type,
                    "namespace": node.namespace,
                    "description": node.description,
                    # Not a ranking. `graph.query` returns nodes in its own
                    # order and this surface has no score to report, so it
                    # fills in a constant. Labelled UNSCORED so a caller does
                    # not read 1.0 as a perfect match - and so the sort below,
                    # which compares this constant against experience recency,
                    # is visible rather than implied.
                    "relevance": 1.0,
                    "relevance_kind": RELEVANCE_KIND_UNSCORED,
                }
            )

        # Access logging moved below, over the allocated rows only.

        # Search experiences for solutions
        experiences = await self.experience.search(
            text=fts_query,
            min_relevance=0.1,
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
                    "relevance": round(exp.relevance(at=now), 4),
                    "relevance_kind": RELEVANCE_KIND_RECENCY,
                }
            )

        # ORDER-PRESERVING, and that is provable rather than hoped for. This
        # used to sort every row on `relevance` alone - comparing a node's
        # literal 1.0 against an experience's decay, two numbers that share a
        # range and nothing else. It is replaced by an explicit two-level key
        # that reproduces the old ORDER exactly:
        #   nodes are all 1.0, so no experience could ever outrank one; the
        #   only tie is a brand-new experience whose decay is also exactly 1.0,
        #   and Python's sort is stable, so that tie already resolved in
        #   insertion order - nodes are appended first, above.
        # Same output, without the cross-scale comparison that made the number
        # look meaningful. Changing the order is a behaviour change and belongs
        # in the phase that owns ranking, not in a labelling change.
        results.sort(
            key=lambda r: (0 if r["source"] == "node" else 1, -r["relevance"]),
        )
        # SAME DEFECT AS recall(), one function away - measured side by side on
        # one store with 10 matching nodes and 10 matching experiences at
        # limit=6:  recall -> 3 nodes / 3 experiences,  crossref -> 6 / 0.
        results = _allocate_across_sources(results, limit)

        # Credit only what the caller received - see the note in recall().
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

        # Route-based node discovery
        route_results = await self.router.route(keywords, limit=limit)

        # NODE RANKING FOR THIS SURFACE IS AN OPEN DEFECT - see below.
        #
        # These nodes come from `router.route()` and report the route's own
        # `confidence`. On a real store that field carries no information:
        #     sqlite3 kairn.db "select confidence, count(*) from routes
        #                       group by confidence"
        #     -> 0.5|23353        (one row - every route in the store)
        # So `min_confidence` excludes nothing at its default, `max(node_scores)`
        # ranks nothing, and the order a caller receives is whatever the id sort
        # produced. A consumer picking the best few nodes has nothing to pick on.
        #
        # AN ATTEMPT TO FIX THIS WAS REVERTED, and the reason is worth keeping.
        # It ranked the router's candidates by intersecting them with
        # `query_ranked(text=fts_query, limit=max(limit*4, 20))` - a GLOBAL
        # top-K - and labelled anything absent from that window UNSCORED. That
        # conflates four different states: no lexical match, a match below an
        # arbitrary cutoff, an id mismatch, and a failed query. Measured on 12
        # real queries, 72 routed nodes:
        #     fts limit    20 ->  9 matched, 63 "unscored"   12.5%
        #     fts limit   100 -> 29 matched, 43 "unscored"   40.3%
        #     fts limit  1000 -> 61 matched, 11 "unscored"   84.7%
        #     fts limit 20000 -> 72 matched,  0 "unscored"  100.0%
        # EVERY routed node matches - unsurprising, since routes are built from
        # node text. The "most candidates do not match" reading was an artefact
        # of the bound, and no fixed multiple of `limit` can fix it.
        #
        # THE CORRECT FIX is at the store layer: rank exactly the routed ids,
        # by adding an id restriction to `_query_nodes_fts`'s existing WHERE
        # clause, so coverage is complete by construction rather than by a
        # window size. Not done here - it is a storage-layer change to a
        # published package and belongs in its own reviewed commit.
        nodes = []
        for r in route_results:
            node_data = r["node"]
            node_out: dict[str, Any] = {
                "id": node_data["id"],
                "name": node_data["name"],
                "type": node_data["type"],
                "namespace": node_data.get("namespace"),
                "confidence": r["confidence"],
            }
            if detail != "summary":
                node_out["description"] = node_data.get("description")
                node_out["tags"] = node_data.get("tags")
                node_out["properties"] = node_data.get("properties")
            nodes.append(node_out)

        # Also search by FTS5 if router found nothing
        if not nodes and fts_query:
            fts_nodes = await self.graph.query(text=fts_query, limit=limit)
            for n in fts_nodes:
                node_out = {
                    "id": n.id,
                    "name": n.name,
                    "type": n.type,
                    "namespace": n.namespace,
                    "confidence": 0.5,
                }
                if detail != "summary":
                    node_out["description"] = n.description
                    node_out["tags"] = n.tags
                    node_out["properties"] = n.properties
                nodes.append(node_out)

        # Log node access for activity tracking
        if nodes:
            await self._log_node_access(
                "node_context", [n["id"] for n in nodes]
            )

        # Experience search
        experiences = await self.experience.search(
            text=fts_query,
            min_relevance=0.1,
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
                "relevance": round(e.relevance(at=now), 4),
                "relevance_kind": RELEVANCE_KIND_RECENCY,
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
