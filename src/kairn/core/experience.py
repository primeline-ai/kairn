"""Experience/Decay Memory engine for Kairn.

Manages experiences with exponential decay based on type-specific half-lives
and confidence multipliers. Automatically promotes frequently accessed experiences
to the knowledge graph.
"""

import logging
import math
import os
import re
from datetime import UTC, datetime
from typing import Any

from kairn.core.fts import _STOP_WORDS, fts_keywords, to_fts_query
from kairn.events.bus import EventBus
from kairn.events.types import EventType
from kairn.models.experience import VALID_CONFIDENCES, VALID_TYPES, Experience
from kairn.models.node import Node
from kairn.storage.base import StorageBackend

logger = logging.getLogger(__name__)

# Default half-lives by type (in days). SINGLE SOURCE OF TRUTH for decay
# (config.py:Config.decay_rate_for_type delegates here).
#
# Calibrated 2026-06-13 against the REAL access tail of a 6483-experience
# production store (scripts/calibrate_halflives.py is the re-runnable
# derivation). The prior values (solution 200 / pattern 300 / decision 100 /
# workaround 50 / gotcha 200) were 3-21x longer than the observed p95 re-access
# interval, leaving decay effectively inactive. These are anchored at ~1.5-2x
# the observed p99 re-access interval per type (so a fact still re-accessed at
# its p99 tail sits at ~0.6-0.7 relevance, not near-zero - "calibrate on the
# tail, never the mean"). Observed p99 days: solution 53.5, pattern 40.5,
# decision 71.0, gotcha 32.4; workaround sparse (n=2) so kept conservative.
HALF_LIVES: dict[str, float] = {
    "solution": 120,   # p99 53.5d
    "pattern": 90,     # p99 40.5d
    "decision": 100,   # p99 71.0d (already near-calibrated)
    "workaround": 40,  # sparse re-access signal; conservative
    "gotcha": 70,      # p99 32.4d
}

# NOTE on the dead salience signals (audited 2026-06-13, kept honestly inert):
# `score` is uniformly 1.0 and `relevance()` ignores both `access_count` and
# `last_accessed`. An access-frequency/recency salience term was NOT activated:
# the LongMemEval harness ingests + recalls within one run (age ~minutes, no
# re-access), so it cannot measure a salience/access change - activating it
# would be an unmeasured guess. Left inert and documented rather than invented.

# Confidence multipliers for decay_rate
CONFIDENCE_MULTIPLIERS: dict[str, float] = {
    "high": 1.0,
    "medium": 2.0,  # 2x faster decay
    "low": 4.0,  # 4x faster decay
}


def decay_rate_from_half_life(half_life_days: float) -> float:
    """Convert half-life to decay rate: rate = ln(2) / half_life.

    Args:
        half_life_days: Half-life in days

    Returns:
        Decay rate constant
    """
    return math.log(2) / half_life_days


# Bucket width for quantizing decay-relevance when a text query is present.
# Exact relevance values are unique per row (microsecond created_at
# differences), so sorting by them collapses to pure recency and throws
# away match strength. Coarse buckets keep decay meaningful across large
# age gaps while letting BM25 order win within a bucket.
RELEVANCE_BUCKET_SIZE = 0.05


def relevance_bucket(value: float) -> int:
    """Quantize a relevance value into a coarse bucket index.

    Args:
        value: Relevance in [0.0, 1.0]

    Returns:
        Bucket index (higher = more relevant)
    """
    return round(value / RELEVANCE_BUCKET_SIZE)


# Leading calendar-day pattern shared by both date conventions seen in real
# data: ISO ("2023-05-20T03:39:00", "2023-05-20 ...") and the slash form
# LongMemEval and conversational ingest use ("2023/05/20 (Sat) 02:21").
_DATE_PREFIX_RE = re.compile(r"^\s*(\d{4})[-/](\d{2})[-/](\d{2})")


def normalize_date_prefix(value: str | None) -> str | None:
    """Extract a canonical ISO day prefix (YYYY-MM-DD) from a date-bearing string.

    Both separator conventions normalize to the same comparable form, so
    valid-time comparisons work across mixed conventions within one workspace
    (previously a documented silent-breakage caveat of raw string slicing).
    Returns None when the string does not start with a recognizable calendar
    day - callers treat that as "no validity signal" (always eligible),
    never as an error.
    """
    if value is None:
        return None
    m = _DATE_PREFIX_RE.match(value)
    if m is None:
        return None
    return "-".join(m.groups())


def derive_entity_key(content: str, tags: list[str] | None) -> str | None:
    """Derive a deterministic cross-session entity grouping key (rule-based).

    No LLM, no embeddings (air-gap + dependency-light constraint). The subject
    of an experience is taken from its FIRST tag, normalized (lowercased,
    trimmed). Tags are the explicit subject signal in Kairn; experiences about
    the same subject across sessions share the first tag and therefore group.
    Returns None when there are no tags (recall then falls back to the
    session/valid-time-diverse path, which is the validated multi-session
    lever - entity grouping is the secondary mechanism).

    Deterministic and seed-stable: same (content, tags) always yields the same
    key. `content` is accepted for signature stability and future heuristics
    but is intentionally not used in v1 (tag-only keying avoids the noise of
    free-text subject extraction).
    """
    if not tags:
        return None
    key = tags[0].strip().lower()
    return key or None


# Tokens that appear capitalized in conversational ingest but never denote a
# subject entity: the "[conversation on DATE (Sat) ...] User: ... Assistant:"
# framing that both real transcript imports and the benchmark harness produce.
# Deliberately separate from fts._STOP_WORDS (which IS also applied): these
# are proper-noun-shaped framing tokens, not common words.
_ENTITY_TOKEN_BLOCKLIST = {
    "user", "assistant", "conversation",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
}

# An entity token: a capitalized word, or a mixed-case product token whose
# capital is internal ("iPhone", "eBay", "macOS") - the lowercase lead means
# the capitalization is intrinsic, never sentence-induced.
_ENTITY_TOKEN = r"(?:[A-Z][\w'-]*|[a-z]{1,3}[A-Z][\w'-]*)"

# A proper-noun run: an entity token optionally continued by further entity
# tokens or number-bearing tokens joined by single spaces - "Dell XPS 13",
# "Adobe Premiere Pro", "GPT-4 Turbo".
_ENTITY_RUN_RE = re.compile(
    rf"(?<![\w'-])({_ENTITY_TOKEN}(?:[ ](?:{_ENTITY_TOKEN}|\d[\w-]*))*)"
)


def _is_sentence_lead(content: str, start: int) -> bool:
    """True when the token at `start` opens a sentence (or the whole string),
    where capitalization carries no proper-noun signal. Skippable leading
    punctuation and sentence terminators include the common Unicode
    typographic forms (curly quotes, em/en dashes) seen in real transcripts."""
    i = start - 1
    while i >= 0 and content[i] in " \t\"'([{*“”‘’„":
        i -= 1
    return i < 0 or content[i] in ".!?\n:;]—–…"


def _env_int(name: str, default: int) -> int:
    """Read an integer tuning knob from the environment, falling back to the
    default on missing OR malformed values - a stray non-numeric override
    left over from a benchmark sweep must degrade to defaults, never crash
    the recall path."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring malformed %s=%r; using default %d", name, raw, default)
        return default


def derive_entity_keys(
    content: str, tags: list[str] | None, *, max_keys: int = 16
) -> set[str]:
    """Derive the full rule-based entity-key SET for an experience.

    The multi-key, content-derived companion to `derive_entity_key` (which
    stays the single stored key for schema stability). Zero LLM, zero
    embeddings, deterministic: normalized tags plus proper-noun runs extracted
    from content ("Dell XPS 13" yields the run itself and its word components,
    so a later mention of just "XPS" still overlaps). Mixed-case product
    tokens with an internal capital ("iPhone", "eBay") count as entity tokens
    - their capitalization is intrinsic, never sentence-induced. Computed at
    query time by the density-preserving diversification pass - nothing is
    stored, so existing rows benefit without migration or backfill.

    Noise filters, in order: dialogue-turn labels (a run directly followed by
    ':' - "User:", "Human:", "Q:") are structural, never subjects; leading
    stop-words are stripped from runs so sentence-initial articles never leak
    in as keys ("The Rhine" -> "rhine", never "the"); single CAPITALIZED
    words at a sentence start are skipped (no proper-noun signal there);
    stop-words and conversational framing tokens are excluded everywhere.
    Capped at `max_keys` in first-occurrence order.
    """
    keys: set[str] = set()

    def _add(raw: str) -> None:
        k = raw.strip().lower()
        if (
            k
            and len(keys) < max_keys
            and k not in _ENTITY_TOKEN_BLOCKLIST
            and k not in _STOP_WORDS
        ):
            keys.add(k)

    for t in tags or []:
        _add(t)

    for m in _ENTITY_RUN_RE.finditer(content):
        run = m.group(1)
        end = m.end(1)
        # Dialogue-turn label: structural framing, never a subject.
        if end < len(content) and content[end] == ":":
            continue
        words = run.split(" ")
        # Strip leading stop-words ("The Rhine" at sentence start): the
        # survivors were capitalized without sentence pressure, so they keep
        # their proper-noun signal and skip the sentence-lead check below.
        stripped = False
        while words and words[0].lower() in _STOP_WORDS:
            words = words[1:]
            stripped = True
        if not words:
            continue
        if len(words) == 1:
            w = words[0]
            if len(w) < 3:
                continue
            if (
                not stripped
                and w[0].isupper()
                and _is_sentence_lead(content, m.start(1))
            ):
                continue
            _add(w)
        else:
            _add(" ".join(words))
            for w in words:
                if len(w) >= 3 and not w[0].isdigit():
                    _add(w)
    return keys


class ExperienceEngine:
    """Engine for managing experiences with decay and promotion."""

    def __init__(self, store: StorageBackend, event_bus: EventBus) -> None:
        """Initialize ExperienceEngine.

        Args:
            store: Storage backend
            event_bus: Event bus for publishing events
        """
        self.store = store
        self.event_bus = event_bus

    async def save(
        self,
        *,
        content: str,
        type: str,
        context: str | None = None,
        confidence: str = "high",
        tags: list[str] | None = None,
        namespace: str = "knowledge",
        valid_from: str | None = None,
        valid_to: str | None = None,
    ) -> Experience:
        """Save a new experience.

        Args:
            content: Experience content
            type: Experience type (must be in VALID_TYPES)
            context: Optional context information
            confidence: Confidence level (must be in VALID_CONFIDENCES)
            tags: Optional tags
            namespace: Namespace for multi-tenant isolation (default "knowledge")
            valid_from: Optional valid-time start (when the fact became true).
                Orthogonal to created_at and decay; NEVER feeds relevance().
            valid_to: Optional valid-time end (when the fact stopped being true,
                e.g. a supersession). Orthogonal to decay; NEVER feeds relevance().

        Returns:
            Created Experience

        Raises:
            ValueError: If type or confidence is invalid, or content is empty
        """
        # Validate content
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        # Validate type
        if type not in VALID_TYPES:
            raise ValueError(f"Invalid experience type: {type}. Must be one of {VALID_TYPES}")

        # Validate confidence
        if confidence not in VALID_CONFIDENCES:
            raise ValueError(
                f"Invalid confidence level: {confidence}. Must be one of {VALID_CONFIDENCES}"
            )

        # Calculate decay_rate
        half_life = HALF_LIVES[type]
        multiplier = CONFIDENCE_MULTIPLIERS[confidence]
        decay_rate = math.log(2) * multiplier / half_life

        # Create experience
        exp = Experience(
            namespace=namespace,
            type=type,
            content=content,
            context=context,
            confidence=confidence,
            score=1.0,
            decay_rate=decay_rate,
            tags=tags or [],
            valid_from=valid_from,
            valid_to=valid_to,
            entity_key=derive_entity_key(content, tags),
        )

        # Insert into store
        await self.store.insert_experience(exp.to_storage())

        # Emit event
        await self.event_bus.emit(
            EventType.EXPERIENCE_CREATED,
            {
                "exp_id": exp.id,
                "type": exp.type,
                "confidence": exp.confidence,
            },
        )

        logger.info(f"Created experience {exp.id} (type={type}, confidence={confidence})")

        return exp

    async def get(self, exp_id: str) -> Experience | None:
        """Get experience by ID.

        Args:
            exp_id: Experience ID

        Returns:
            Experience if found, None otherwise
        """
        data = await self.store.get_experience(exp_id)
        if data is None:
            return None
        return Experience(**data)

    async def search(
        self,
        *,
        text: str | None = None,
        exp_type: str | None = None,
        min_relevance: float = 0.0,
        limit: int = 10,
        offset: int = 0,
    ) -> list[Experience]:
        """Search for experiences.

        Args:
            text: Text to search for (FTS5)
            exp_type: Filter by experience type
            min_relevance: Minimum relevance threshold
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of matching experiences. With a text query, BM25 match
            strength is the primary order and decay-relevance acts as a
            coarse tiebreak; without text, sorted by relevance descending.
        """
        # Shape free-text into a safe FTS5 query (quoted OR of keywords). This
        # is the same helper the intelligence layer (kn_recall/kn_context) and
        # the LongMemEval benchmark use, so kn_memories now runs the identical,
        # crash-proof recall path. Raw text would let bare hyphens / reserved
        # words reach MATCH and raise OperationalError ("no such column: ...").
        # If the query has no searchable keyword, there is nothing to match.
        # Re-shaping an already-shaped query is idempotent (quotes/OR are
        # stripped by the tokenizer and the surviving keywords re-quote to the
        # same string), so callers that pre-shape stay correct.
        fts_text: str | None
        if text is not None:
            fts_text = to_fts_query(text)
            if fts_text is None:
                return []
        else:
            fts_text = None

        # Query from store (the FTS path returns rows in BM25 match order)
        results = await self.store.query_experiences(
            text=fts_text,
            exp_type=exp_type,
            limit=100000,  # Get all first, filter by relevance
            offset=0,
        )

        # Convert to Experience objects and calculate current relevance
        now = datetime.now(UTC)
        scored: list[tuple[Experience, float]] = []
        for data in results:
            exp = Experience(**data)
            current_relevance = exp.relevance(at=now)

            # Apply min_relevance filter
            if current_relevance >= min_relevance:
                scored.append((exp, current_relevance))

        if fts_text is not None:
            # Match strength stays primary: quantize relevance into coarse
            # buckets and stable-sort by bucket, so the store's BM25 order
            # survives within each bucket. Sorting by exact relevance would
            # collapse to pure recency (microsecond created_at differences
            # make every value unique) and discard match strength.
            scored.sort(key=lambda pair: relevance_bucket(pair[1]), reverse=True)
        else:
            # No match signal without text: exact relevance is the order.
            scored.sort(key=lambda pair: pair[1], reverse=True)

        experiences = [exp for exp, _ in scored]

        # Apply pagination
        return experiences[offset : offset + limit]

    async def search_bitemporal(
        self,
        *,
        text: str | None = None,
        exp_type: str | None = None,
        min_relevance: float = 0.0,
        limit: int = 10,
        as_of: str | None = None,
    ) -> list[Experience]:
        """Bi-temporal-aware recall over the experience layer.

        Same BM25 + decay-bucket ordering as `search()`, then two orthogonal
        bi-temporal refinements (both no-ops when the data lacks the signal, so
        this degrades to an identical BM25 result - the fallback-identity
        contract):

        1. **as-of validity** (temporal-reasoning): if `as_of` is given, drop
           experiences whose `valid_from` falls on a calendar day strictly
           AFTER `as_of`'s day (facts not yet true at the query time). The
           comparison is DAY-granular, not minute-granular: a fact valid later
           the same day as `as_of` is still eligible (LongMemEval-S dates carry
           clock time down to the minute, and 43/500 questions have a gold
           answer session dated the same day as the question but at a later
           clock time - a minute-granular compare silently dropped those).
           NULL-`valid_from` experiences are always eligible. `valid_to` is NOT
           consulted here - validity-END is a separate concern and never
           affects recall ordering, preserving the decay/validity orthogonality.
           Day prefixes are canonicalized via `normalize_date_prefix`, so the
           two date conventions seen in real data ("YYYY/MM/DD ..." and ISO
           "YYYY-MM-DD...") compare correctly even when mixed within one
           workspace; a `valid_from` with no recognizable leading calendar day
           carries no validity signal and stays always-eligible.
        2. **session/entity diversification** (multi-session): re-rank so the
           head of the result spreads across distinct subjects/sessions. The
           diversity key is `entity_key` when present, else `valid_from` (the
           session/valid-time key). The DEFAULT algorithm is the
           density-preserving pass (`_diversify_density_preserving`): an
           untouched BM25 anchor head, then a bounded, on-topic-gated
           coverage pass promoting the best hit per unrepresented group; the
           prior unconditional first-hit-per-key pass remains selectable via
           KAIRN_DIVERSIFY_MODE=legacy for A/B measurement. When every
           candidate already fits within `limit`, the density pass returns
           raw BM25 order unchanged - diversification exists to fix
           truncation loss, and nothing is truncated. Experiences with NO
           diversity key are never merged, so a store with no
           windows/entities is returned unchanged.

        This is the recall path the LongMemEval harness measures via
        `--recall-mode bitemporal`; the validated multi-session lever is the
        diversification step (Kairn `7784817d`).
        """
        fts_text: str | None
        if text is not None:
            fts_text = to_fts_query(text)
            if fts_text is None:
                return []
        else:
            fts_text = None

        results = await self.store.query_experiences(
            text=fts_text, exp_type=exp_type, limit=100000, offset=0
        )

        as_of_day = normalize_date_prefix(as_of)
        now = datetime.now(UTC)
        scored: list[tuple[Experience, float]] = []
        for data in results:
            exp = Experience(**data)
            # as-of validity filter (valid-time, NOT valid_to / decay) - see
            # criterion 1 above for the day-granularity rationale.
            if as_of_day is not None:
                vf_day = normalize_date_prefix(exp.valid_from)
                if vf_day is not None and vf_day > as_of_day:
                    continue
            current_relevance = exp.relevance(at=now)
            if current_relevance >= min_relevance:
                scored.append((exp, current_relevance))

        if fts_text is not None:
            scored.sort(key=lambda pair: relevance_bucket(pair[1]), reverse=True)
        else:
            scored.sort(key=lambda pair: pair[1], reverse=True)

        ordered = [exp for exp, _ in scored]
        # Session/entity diversification is an operator-toggleable kill-switch
        # (default ON). Set KAIRN_BITEMPORAL_DIVERSIFY=0 to use as-of validity
        # filtering alone without diversification. The default algorithm is the
        # density-preserving pass (anchored, on-topic-gated); the prior
        # unconditional first-hit-per-key pass (which diluted answer-content
        # density - the benchmarked regression, Kairn daf5cd8e) remains
        # selectable via KAIRN_DIVERSIFY_MODE=legacy for A/B measurement.
        if os.environ.get("KAIRN_BITEMPORAL_DIVERSIFY", "1") != "0":
            if os.environ.get("KAIRN_DIVERSIFY_MODE", "density") == "legacy":
                ordered = self._diversify_by_session(ordered)
            else:
                terms = set(fts_keywords(text)) if text is not None else None
                ordered = self._diversify_density_preserving(
                    ordered, limit=limit, query_terms=terms
                )
        return ordered[:limit]

    @staticmethod
    def _diversify_density_preserving(
        experiences: list[Experience],
        *,
        limit: int,
        query_terms: set[str] | None,
    ) -> list[Experience]:
        """Density-preserving session/entity diversification.

        Replaces the unconditional first-hit-per-key promotion (which collapsed
        answer-content density in the head - the benchmarked 0.31 regression,
        Kairn `daf5cd8e`: "right sessions, wrong content") with a bounded,
        on-topic coverage pass:

        1. ANCHOR - the top-A BM25 hits stay untouched, preserving the
           corroborating-rows density multi-excerpt answers need.
        2. COVERAGE - walk the next W*limit candidates in BM25 order and
           promote the best hit of each not-yet-represented session/entity
           group into the head, but only when the hit is plausibly on-topic:
           it shares >=2 query terms with the question (>=1 for single-term
           queries), OR >=1 derived entity key (`derive_entity_keys`) with the
           anchor set - the latter catches the aggregation case where the
           question names a category ("laptops") while sessions name instances
           ("Dell XPS 13").
        3. FILL - everything else follows in original BM25 order.

        Group identity matches the legacy pass (stored entity_key, else the
        full valid_from string = session identity); items with neither are
        never promoted or merged. Returns the input unchanged when it already
        fits within `limit` - diversification exists to fix truncation loss,
        and nothing is truncated in that case. Returns at most `limit` items.

        Benchmark-tuning knobs (read per call so harness sweeps work without
        code edits; malformed values degrade to defaults via `_env_int`, never
        crash recall): KAIRN_DIVERSIFY_ANCHOR (default 4, adaptively capped at
        limit//2), KAIRN_DIVERSIFY_WINDOW (default 3, in multiples of
        `limit`), KAIRN_DIVERSIFY_MIN_SHARED (default 2; clamped to the
        query's actual term count; 0 disables the gate).
        """
        if len(experiences) <= limit:
            return experiences

        anchor_n = max(1, min(_env_int("KAIRN_DIVERSIFY_ANCHOR", 4), limit // 2))
        window_mult = _env_int("KAIRN_DIVERSIFY_WINDOW", 3)

        # On-topic gate strength. Clamped to the terms actually available: a
        # 2-term query can never share 3 terms, and an operator-raised gate
        # must stay strict for term-rich queries instead of silently
        # collapsing to 1. Empirical note from the diag sweeps: scoring
        # candidates by raw shared-term count performed WORSE than plain BM25
        # order (verbose distractor rows out-count genuinely relevant ones;
        # BM25's length normalization already handles this) - promotion
        # therefore stays greedy in BM25 order and the term count is only a
        # gate.
        gate = _env_int("KAIRN_DIVERSIFY_MIN_SHARED", 2)
        min_shared = min(gate, len(query_terms)) if query_terms else gate

        def group(e: Experience) -> str | None:
            return e.entity_key or e.valid_from

        anchor = experiences[:anchor_n]
        seen_groups = {g for e in anchor if (g := group(e)) is not None}
        anchor_keys: set[str] | None = None  # computed lazily on first need

        window_end = min(len(experiences), anchor_n + window_mult * limit)
        slots = limit - anchor_n

        promoted: list[Experience] = []
        promoted_idx: set[int] = set()
        for idx in range(anchor_n, window_end):
            if len(promoted) >= slots:
                break
            e = experiences[idx]
            g = group(e)
            if g is None or g in seen_groups:
                continue
            if min_shared > 0:
                on_topic = False
                if query_terms:
                    # Token-set membership, not substring containment: "art"
                    # must not count inside "start"/"party".
                    content_tokens = set(
                        re.findall(r"[a-zA-Z0-9_]+", e.content.lower())
                    )
                    on_topic = len(query_terms & content_tokens) >= min_shared
                if not on_topic:
                    if anchor_keys is None:
                        anchor_keys = set()
                        for a in anchor:
                            anchor_keys |= derive_entity_keys(a.content, a.tags)
                    if anchor_keys & derive_entity_keys(e.content, e.tags):
                        on_topic = True
                if not on_topic:
                    continue
            seen_groups.add(g)
            promoted.append(e)
            promoted_idx.add(idx)

        # Fill the remaining slots in BM25 order without materializing the
        # full tail (the candidate list can be arbitrarily large) and using
        # positional identity, which is collision-proof (Experience.id is a
        # truncated UUID - id-based dedup could silently drop a distinct row
        # on an id collision).
        result = anchor + promoted
        for idx in range(anchor_n, len(experiences)):
            if len(result) >= limit:
                break
            if idx not in promoted_idx:
                result.append(experiences[idx])
        return result

    @staticmethod
    def _diversify_by_session(experiences: list[Experience]) -> list[Experience]:
        """Re-rank a BM25-ordered list for session/entity diversity.

        Round 1: best experience per distinct diversity key (entity_key else
        valid_from). Round 2: remaining experiences in original order. Keys that
        are None never merge (each such experience is its own group), so a list
        with no windows/entities is returned unchanged (fallback identity).
        """
        seen: set[str] = set()
        head: list[Experience] = []
        tail: list[Experience] = []
        for exp in experiences:
            key = exp.entity_key or exp.valid_from
            if key is None:
                head.append(exp)  # no diversity signal -> keep in place
            elif key not in seen:
                seen.add(key)
                head.append(exp)
            else:
                tail.append(exp)
        return head + tail

    async def group_by_entity(
        self, entity_key: str, *, limit: int = 1000
    ) -> list[Experience]:
        """Return the cross-session timeline for an entity.

        Given a normalized entity_key (see `derive_entity_key`), return every
        experience sharing it - across sessions/namespaces - ordered by
        valid-time (valid_from, falling back to created_at). This is the
        cross-session aggregation helper the bi-temporal recall path uses for
        multi-session questions. Empty list for an unknown key.
        """
        rows = await self.store.get_experiences_by_entity_key(entity_key, limit=limit)
        return [Experience(**data) for data in rows]

    async def touch_accessed(self, exp_ids: list[str]) -> int:
        """Batch-increment access_count for a list of experience IDs.

        Used by the intelligence layer read path so that returning N
        experiences from recall/context/crossref registers N access events
        in a single SQL round-trip. Fires the `exp_auto_promote` SQL trigger
        once per row when access_count crosses the threshold (SQLite
        triggers are always FOR EACH ROW, verified against the schema at
        `src/kairn/schema/triggers.sql`).

        Empty list is a no-op and returns 0 (the store layer is the
        load-bearing short-circuit — this wrapper does not pre-filter).

        Unknown IDs in the list are silently ignored (SQL WHERE IN filter).

        This does NOT perform the application-level promotion step (creating
        a node for a flagged experience). Promotion candidates are picked up
        later via `get_promotable()` and explicit `access()` calls, or by a
        background sweeper. The goal here is to keep the hot read path lean.

        Returns the number of rows affected.
        """
        return await self.store.touch_accessed_experiences(exp_ids)

    async def access(self, exp_id: str) -> Experience | None:
        """Access an experience (increments access count and checks for promotion).

        Args:
            exp_id: Experience ID

        Returns:
            Updated Experience if found, None otherwise
        """
        # Increment access count
        await self.store.increment_access_count(exp_id)

        # Get updated experience
        exp = await self.get(exp_id)
        if exp is None:
            return None

        # Emit access event
        await self.event_bus.emit(
            EventType.EXPERIENCE_ACCESSED,
            {
                "exp_id": exp.id,
                "access_count": exp.access_count,
            },
        )

        logger.info(f"Accessed experience {exp.id} (count={exp.access_count})")

        # Check for promotion
        if exp.promoted_to_node_id is None:
            # Check if needs promotion (flagged by trigger)
            promotable = await self.get_promotable()
            if any(p.id == exp.id for p in promotable):
                node = await self._promote(exp)
                if node:
                    # Refresh experience to get updated promoted_to_node_id
                    exp = await self.get(exp_id)

        return exp

    async def prune(self, *, threshold: float = 0.01) -> list[str]:
        """Prune experiences below relevance threshold.

        Args:
            threshold: Minimum relevance threshold

        Returns:
            List of pruned experience IDs
        """
        # Find all experiences below threshold
        now = datetime.now(UTC)
        all_experiences = await self.store.query_experiences(limit=100000, offset=0)

        pruned_ids = []
        for data in all_experiences:
            exp = Experience(**data)
            current_relevance = exp.relevance(at=now)
            if current_relevance < threshold:
                # Archive before deleting (log)
                logger.info(
                    f"Pruning experience {exp.id} "
                    f"(relevance={exp.relevance(at=now):.4f}, threshold={threshold})"
                )

                # Delete from store
                await self.store.delete_experience(exp.id)

                # Emit event
                await self.event_bus.emit(
                    EventType.EXPERIENCE_PRUNED,
                    {"exp_id": exp.id},
                )

                pruned_ids.append(exp.id)

        logger.info(f"Pruned {len(pruned_ids)} experiences")
        return pruned_ids

    async def promote_pending(self, *, limit: int = 100) -> dict[str, Any]:
        """Promote experiences flagged by the auto-promote SQL trigger.

        The `exp_auto_promote` trigger sets `properties.needs_promotion = 1`
        when `access_count` crosses 5. The flag stays set forever until
        something promotes the experience to a permanent graph node and
        clears the dead-letter state by setting `promoted_to_node_id`.

        This sweeper finds those flagged experiences and calls `_promote()`
        directly on each. Unlike `access()`, this does NOT increment
        access_count again (the trigger fired because the count already
        crossed the threshold via `touch_accessed()` from the read path).

        Race-safety: `_promote()` uses `store.promote_experience_atomic`
        which wraps INSERT + CAS UPDATE in one transaction. Concurrent
        sweepers cannot create duplicate nodes for the same source
        experience — the loser of the race gets `won=False` and the
        partially-inserted node is rolled back.

        Returns a stats dict:
            flagged_total: total experiences in the snapshot (pre-limit)
            attempted: number actually processed this call (= min(flagged_total, limit))
            promoted: attempted that became permanent nodes
            raced: attempted where another writer won the CAS (no orphan)
            failed: attempted where _promote raised an exception
            nodes_created: list of new node ids (length == promoted)
            errors: list of "exp_id: error" strings (capped at 10)

        Invariant: attempted == promoted + raced + failed.
        Idempotent: re-running on a clean DB is a no-op.

        Args:
            limit: maximum number of experiences to process per call.
                   Pushed down into SQL so a large backlog doesn't load
                   the full set into memory.
        """
        # Push the limit into SQL — bounded I/O, bounded memory.
        promotable = await self.get_promotable(limit=limit)
        flagged_total = len(promotable)
        attempted = 0
        promoted = 0
        raced = 0
        failed = 0
        nodes_created: list[str] = []
        errors: list[str] = []

        for exp in promotable:
            attempted += 1
            try:
                node = await self._promote(exp)
                if node is not None:
                    promoted += 1
                    nodes_created.append(node.id)
                else:
                    # _promote returned None: another writer won the CAS.
                    # The store has already rolled back the orphan node.
                    raced += 1
            except Exception as e:  # noqa: BLE001
                failed += 1
                errors.append(f"{exp.id}: {type(e).__name__}: {e}"[:200])
                logger.warning(
                    "promote_pending: failed to promote %s: %s", exp.id, e
                )

        if promoted:
            logger.info(
                "promote_pending: promoted %d/%d (raced=%d failed=%d)",
                promoted, attempted, raced, failed,
            )

        return {
            "flagged_total": flagged_total,
            "attempted": attempted,
            "promoted": promoted,
            "raced": raced,
            "failed": failed,
            "nodes_created": nodes_created,
            "errors": errors[:10],
        }

    async def _promote(self, experience: Experience) -> Node | None:
        """Promote experience to knowledge graph node.

        Atomic + race-safe via `store.promote_experience_atomic`. Returns
        None if another writer (e.g. a parallel sweeper, or a 2-MCP-server
        deployment) already claimed this experience between the
        get_promotable read and our write attempt. The atomic helper
        rolls back the partially-inserted node so the database stays
        orphan-free.

        Args:
            experience: Experience to promote

        Returns:
            Created Node on success, None if a race was lost or
            promotion failed mid-write.
        """
        # Build the candidate node (in-memory only at this point)
        node = Node(
            type="promoted_experience",
            name=f"{experience.type.capitalize()}: {experience.content[:50]}",
            namespace="knowledge",
            description=experience.content,
            properties={
                "source_experience_id": experience.id,
                "experience_type": experience.type,
                "confidence": experience.confidence,
                "access_count": experience.access_count,
                "tags": experience.tags,
            },
        )

        # Single transactional step: INSERT + CAS UPDATE.
        won = await self.store.promote_experience_atomic(
            experience.id, node.to_storage()
        )
        if not won:
            return None

        # Emit event
        await self.event_bus.emit(
            EventType.EXPERIENCE_PROMOTED,
            {
                "exp_id": experience.id,
                "node_id": node.id,
            },
        )

        logger.info(
            f"Promoted experience {experience.id} to node {node.id} "
            f"(access_count={experience.access_count})"
        )

        return node

    async def get_promotable(
        self, *, limit: int | None = None
    ) -> list[Experience]:
        """Get experiences flagged for promotion.

        Args:
            limit: optional SQL-side LIMIT pushed into the store query
                so a large promotion backlog does not load the full
                set into memory.

        Returns:
            List of experiences that need promotion (capped at `limit`
            if provided).
        """
        results = await self.store.get_promotable_experiences(limit=limit)
        return [Experience(**data) for data in results]
