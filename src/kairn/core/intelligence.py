"""Intelligence layer — learn, recall, crossref, context, related.

Bridges graph, experience, and router engines into unified knowledge operations.
"""

from __future__ import annotations

import re

import asyncio
import logging
import sqlite3
import uuid
from collections.abc import Callable
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

    def _node_result(self, *, node_id, name, type_, namespace, description, relevance):
        # namespace travels in every item shape so downstream namespace-based
        # access filters can enforce their allowlists on this surface.
        return {
            "source": "node",
            "id": node_id,
            "name": name,
            "type": type_,
            "namespace": namespace,
            "description": description,
            "relevance": relevance,
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
            relevance = _node_relevance(rank, terms, node.name, node.description)
            if relevance < min_relevance:
                continue
            out.append(
                self._node_result(
                    node_id=node.id,
                    name=node.name,
                    type_=node.type,
                    namespace=node.namespace,
                    description=node.description,
                    relevance=relevance,
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

        # Log node access for activity tracking (only nodes we surfaced).
        if kept_node_ids:
            await self._log_node_access("node_recall", kept_node_ids)

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

        # Batch-increment access_count for all returned experiences so
        # the exp_auto_promote trigger can fire after repeated hits.
        # Mirror the increment on the in-memory objects so callers reading
        # exp.access_count from the result set are not off by one.
        if experiences:
            await self.experience.touch_accessed([e.id for e in experiences])
            for exp in experiences:
                exp.access_count += 1

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
                }
            )

        # Nodes (curated, permanent) lead, then experiences (decaying). Each
        # group is already ranked internally - nodes by bm25 match strength
        # times term coverage (`_keyword_node_recall` now sorts on the score
        # it reports, which raw query_ranked order did not), experiences by
        # the experience engine's search order. We deliberately do NOT
        # merge-sort the union here: `recall` is the surface that must not
        # bury curated nodes under fresh experiences. `crossref` DOES sort the
        # union, and can, because both kinds carry the same match-times-
        # coverage measurement there - see the note at its sort.
        results = results[:limit]

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
                    "relevance": _node_relevance(
                        rank, terms, node.name, node.description
                    ),
                }
            )

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

        # Batch-increment access_count for all returned experiences.
        if experiences:
            await self.experience.touch_accessed([e.id for e in experiences])
            for exp in experiences:
                exp.access_count += 1

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
                }
            )

        # Legitimate now, and only now: both kinds carry bm25 match strength
        # times term coverage, on one scale and at one resolution
        # (`_RELEVANCE_DECIMALS`). Experiences additionally carry the bounded
        # +-10% recency nudge, which is the whole point of a decaying store
        # and cannot promote a weak fresh hit over a strong old one.
        results.sort(key=lambda r: r["relevance"], reverse=True)
        results = results[:limit]

        # Log node access AFTER the truncation, so it records what was
        # actually surfaced. The node pool is now over-fetched for the rerank
        # (`_rerank_pool`), so logging it before this point would credit
        # ~30 nodes for a query that returned 3 - the same "only nodes we
        # surfaced" rule `recall` already follows with `kept_node_ids`.
        surfaced_node_ids = [r["id"] for r in results if r["source"] == "node"]
        if surfaced_node_ids:
            await self._log_node_access("node_crossref", surfaced_node_ids)

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
