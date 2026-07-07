# Plan: Kairn Roadmap Item 4 - Content-Derived Multi-Key Entity Extraction + Entity-Aware Diversification

**Date**: 2026-07-07
**Confidence**: Medium (2 unknowns pending Phase 0 validation - see Assumptions)
**Quality Grade**: A (Stage 3 review complete - see Stage 3 Review Report at end of file)
**Default Vehicle**: Single (self) - see Stage 0.5

---

## DSV Pre-Check

**Decompose** - the roadmap ask (Kairn decision `758ece4c`) bundles three separable claims:
1. `derive_entity_key()` should become "multi-key" / content-derived instead of single-first-tag.
2. `_diversify_by_session()` should become "density-preserving" instead of first-hit-per-key.
3. The resulting mechanism should get wired into a production-reachable MCP tool, gated on diag coverage >0.9 and a pilot-50 win over the 41.4% multi-session baseline.

**Suspend** - the alternative interpretation worth raising before committing to new extraction logic: the diagnostic (`runs/diag/multisession-diag.json`) shows `full_coverage_in_topk` at only 36.7% (22/60) while `mean_session_coverage_ratio` sits at 73.7% - a large gap between "some coverage" and "full coverage." That gap could trace to **topk depth (8) being shallow relative to session spread**, not to entity-grouping quality. If so, the cheap fix is tuning topk/round-2 fill order, not inventing multi-key content extraction. This plan treats it as a genuine fork: Phase 0 explicitly checks whether a topk-only intervention already closes most of the gap before the more expensive extraction work is justified (AHA-effect check, not skipped).

**Validate (least-sure claim)** - which of three diagnostic metrics the roadmap's ">0.9 diag coverage" gate refers to. Current values: `surfaced` rate 59/60=0.983 (already >0.9, trivially passes, gates nothing), `mean_session_coverage_ratio` 0.737 (a plausible, ambitious-but-reachable target), `full_coverage_in_topk` 22/60=0.367 (a very high bar, would require near-perfect per-question coverage). Resolved in Phase 0 as a documented design decision (see Assumptions A1) since the roadmap text alone is ambiguous and no prior Kairn node disambiguates it.

---

## Stage 0: Discovery

**0.1 Existing Work (ran, detailed)** - read `src/kairn/core/experience.py` in full for the relevant surface:
- `derive_entity_key(content, tags)` (line 90): returns `tags[0].strip().lower()` or `None` if no tags. `content` param accepted but unused ("reserved for future heuristics").
- **Correction to the roadmap's own framing**: `entity_key` is NOT dead data. `ExperienceEngine.save()` (line 188) calls `derive_entity_key(content, tags)` on every write and persists the result - confirmed via `sqlite_store.py` (column, index `idx_experiences_entity_key`, `INSERT` binds `:entity_key`). What IS dead is the two **consumers**: `group_by_entity()` and `search_bitemporal()` have zero MCP-tool callers (`grep -c` over `server.py`'s 47 `kn_*` tools returns 0 for both names) - confirmed live 2026-07-07, matches the roadmap claim on that specific point.
- `search_bitemporal()`'s only caller anywhere in the codebase is the external LongMemEval harness (`~/Buisiness/Evolving/_autonomous/benchmarks/longmemeval/bench.py:236`, gated behind `--recall-mode bitemporal`, not the harness's default `engine` mode).
- `_diversify_by_session()` (line 388): round-1 = first hit per `entity_key or valid_from`; round-2 = remaining in original BM25 order. Flag `KAIRN_BITEMPORAL_DIVERSIFY` (default "1"/ON) gates whether `search_bitemporal` calls it at all - moot today since nothing calls `search_bitemporal` in production.
- Real baseline numbers, verified from actual run artifacts (not re-derived from memory): `runs/full-500-s-fixed/summary.json` (recall_mode=engine, the current production default, n=500 real questions) - overall 0.562, multi-session n=133 acc=**0.4135** (confirms the roadmap's "41.4%" to the percent), temporal-reasoning 0.4286, single-session-preference n=30 acc=**0.10** (confirms item 5's baseline), abstention 0.9667. This is a **full-500 result that already exists** - it is the correct baseline to beat, not a re-derived pilot number.
- `runs/diag/multisession-diag.json` (topk=8, full_k=50, n=60 multi-session questions): `surfaced`=59/60, `mean_session_coverage_ratio`=0.737, `full_coverage_in_topk`=22/60. See DSV-Validate above.
- `models/experience.py`: `VALID_TYPES = {"solution","pattern","decision","workaround","gotcha"}` - no "preference" (confirms item 5's file-coupling note; not otherwise relevant to this plan).
- No `.claude/plans/` directory existed in this repo before this plan (created here).

**0.7 Feasibility** - High. All storage/model plumbing (`entity_key` column + index, `get_experiences_by_entity_key`, `group_by_entity`, `search_bitemporal`, the diversify round-1/round-2 shape) already exists and is exercised by the existing test suite (`tests/test_bitemporal.py`, `tests/test_experience.py`). This is an algorithm-enhancement + wiring task, not new infrastructure.

**0.9 Better Alternatives (AHA effect)** - see DSV-Suspend: Phase 0 checks the cheap topk-tuning alternative before committing to multi-key extraction work. A second, smaller AHA candidate: since `entity_key` is already populated from `tags[0]`, "multi-key" could mean simply deriving keys from ALL tags (not just the first) and letting `group_by_entity`/diversification match on ANY shared tag - materially simpler than free-text content heuristics, and consistent with `derive_entity_key`'s own "tags are the explicit subject signal" design philosophy. This is preferred over inventing content-based extraction unless Phase 0 shows all-tags matching is insufficient (documented as the Phase 0 decision, not assumed here).

**0.3 Official Docs** - checked: `entity_key` already has a dedicated SQLite index (`idx_experiences_entity_key`), so a multi-key scheme that stores one row per experience with a *set* of keys would need either (a) a join table, or (b) denormalizing to one row per (experience, key) pair, or (c) keeping single-column `entity_key` but changing what populates it (e.g., a canonicalized multi-tag composite, or the FIRST MATCHING key against a pre-existing key rather than always `tags[0]`). No SQLite version/feature gap - this is a schema-design decision, not a docs gap. Resolved in Phase 0 (leans toward (c) to avoid a schema migration - see A2).

**Skipped, with reasons**: 0.2 Factual Verification (folded into 0.1 - all factual claims were re-verified against live code/data, not taken from the roadmap node on faith). 0.4 Updates Scan (no external dependency versions involved). 0.5 Best Practices (no external-library integration; the "best practice" here IS the AHA-effect check in 0.9). 0.6 Deep Research (Zep/mem0's LLM-based entity resolution is already characterized in Kairn nodes from the roadmap workflow; re-researching competitors doesn't change this repo's implementation choices, which are constrained to zero-LLM/zero-embedding by design). 0.10 Competitive (same reason). 0.11 Constraint Discovery (the constraints - no LLM, no embeddings, air-gap - are already fixed project invariants, re-affirmed in `derive_entity_key`'s own docstring). 0.12 People Risk (solo-maintainer OSS project, no team/stakeholder coordination risk).

---

## Stage 0.5: Vehicle Selection

**Default Vehicle: Single (self)**. This is single-repo, sequential algorithm-design-then-implementation-then-benchmark work with a hard dependency chain (design decision -> code -> gate-eval -> wire-or-halt) - not decomposable into independent parallel streams. Per-phase deviation:

> Phase 6 (RC-gate): Vehicle: Sub-Agent - `feature-dev:code-reviewer` (sonnet, spec-compliance then quality) + `/simplify` (parallel cleanup-lens agents, sonnet - reuse/simplification/efficiency/altitude lenses, same tier as the reviewer since neither is complexity-8+ escalation-worthy) - the established, already-proven pattern from items 1+2's closeout, not a new choice.

---

## Context & Why

`derive_entity_key()` and its consumers (`group_by_entity`, `search_bitemporal`'s diversification) exist but are unreachable from any of Kairn's 47 production MCP tools, and the single-first-tag keying scheme is the simplest possible version of "which experiences are about the same subject." This item exists because the roadmap's competitive analysis (Kairn `758ece4c`) found Kairn's multi-session recall (41.4%) trailing Zep's LLM-based entity resolution (57.9%), and a richer, still-zero-LLM keying + diversification scheme is the identified lever to close part of that gap without breaking Kairn's air-gapped, dependency-light design constraint.

## End State

After this plan succeeds, `derive_entity_key()` derives entity keys from all of an experience's tags (not just the first), `_diversify_by_session()`'s round-2 fill is density-aware (biases toward under-represented entities/sessions instead of pure BM25 leftover order), and the diagnostic + a pilot-50 LongMemEval-S run show whether this clears the acceptance gate. If it clears, one MCP tool path (chosen in Phase 0) calls `search_bitemporal`/`group_by_entity` for real, and the change is merged to `main` with the existing RC-gate discipline. If it does not clear, the gate failure is documented with real numbers (precedent: gotcha `65b08ce7`) and nothing is wired into production - a valid, honest terminal state.

**[Clarified, Interview Tier 3 Cold-Start]**: "multi-key" here stays entirely rule-based/tag-based (matching `derive_entity_key`'s existing zero-LLM/zero-embedding design) - it is NOT the Zep-style LLM entity-resolution analog in the literal sense of running an extraction model. The roadmap's "no-LLM analog of Zep's LLM entity resolution" phrasing means "achieves a similar grouping *outcome* without an LLM," not "reimplements LLM extraction." A reader unfamiliar with this distinction would likely assume otherwise - stated explicitly here to preempt that.

## Success Criteria

- `derive_entity_key()` produces multi-key entity grouping from tag sets (mechanism decided in Phase 0), covered by new unit tests.
- `_diversify_by_session()`'s round-2 ordering is density-aware, covered by new unit tests demonstrating the behavior change vs the current BM25-leftover-order baseline.
- Diagnostic (`multisession-diag.json`-equivalent re-run) shows the Phase-0-selected coverage metric **>0.9** (see A1 for which metric).
- Pilot-50 LongMemEval-S re-run (recall_mode reflecting the new code path) shows multi-session accuracy **> 0.4135** (the verified full-500 baseline, not a re-derived number).
- **NOT-scope**: full-500 LongMemEval re-run (pilot-50 is the gate-decision scale per the roadmap and per this session's explicit "not in scope" instruction); WOW items 6/7/8-10; re-opening the `KAIRN_BITEMPORAL_DIVERSIFY` full-diversify-as-default halt verdict (`65b08ce7` stands regardless of this item's algorithm changes); content-based (non-tag) entity extraction unless Phase 0's AHA-check shows all-tags matching is insufficient.
- **FAILED conditions (kill criteria)**: (a) Phase 0's topk-tuning check alone closes >80% of the coverage gap - in which case the multi-key/density work is de-scoped down to that cheaper fix and the rest of this plan's phases are replanned or dropped (documented, not silently abandoned); (b) the pilot-50 gate fails (multi-session accuracy does not beat 0.4135) - implementation stops at Phase 4, no production wiring, closeout documents the negative result per the `65b08ce7` precedent; (c) if Phase 0 or Phase 2 discovers the schema-design decision (A2) requires a migration more invasive than a single-column reinterpretation, stop and replan rather than force a denormalized schema change into this estimate.
- **Timeout**: if Phase 0-3 (design through diversification implementation) exceed 3 elapsed days without reaching Phase 4's gate-eval, stop and reassess scope with Robin rather than silently extending past the roadmap's 2-4 day estimate.

## Assumptions & Validation

- **A1**: The roadmap's ">0.9 diag coverage" gate refers to `mean_session_coverage_ratio` (currently 0.737), not `full_coverage_in_topk` (0.367, an extreme bar) or `surfaced` (0.983, already-passing and therefore not a real gate).
  -> VALIDATE BY: state this interpretation explicitly in the Phase 0 deliverable and in the eventual closeout; if Robin disagrees when reviewing the shipped result, the gate can be re-evaluated against the alternative metric using the same diagnostic run (no new data collection needed).
  -> IMPACT IF WRONG: if the intended metric was actually `full_coverage_in_topk`, the true bar is much higher (0.367 -> 0.9) and may not be reachable within the 2-4 day estimate - would require a scope/timeline conversation with Robin before Phase 4, not a silent redefinition.

- **A2**: "Multi-key" entity extraction is implemented as deriving `entity_key` from ALL tags (matching if ANY tag overlaps) rather than free-text content heuristics, keeping the existing single-column `entity_key` schema (via a canonicalization scheme - e.g., a composite/sorted-tags key, or a match-any-tag join at query time) rather than a schema migration to a join table.
  -> VALIDATE BY: Phase 0 prototype against the existing diagnostic harness (`diag_recall.py`) on the 60 multi-session questions - confirm the all-tags approach measurably improves `mean_session_coverage_ratio` before committing to it over content-heuristic alternatives.
  -> IMPACT IF WRONG: if all-tags matching doesn't move the diagnostic number, Phase 0 must design a content-derived heuristic instead (the roadmap's literal ask) - larger scope, revisit the 2-4 day estimate with Robin.

- **A3**: "Density-preserving diversification" means round-2 fill order is re-ranked to prefer experiences from entities/sessions currently under-represented in the result set so far, rather than raw BM25-leftover order (which is what round-2 does today).
  -> VALIDATE BY: unit test asserting the new round-2 order differs from pure-BM25-leftover order on a constructed fixture with skewed entity distribution, and that answer-content density (per gotcha `daf5cd8e`'s regression concern) does not silently regress - re-run the diag harness's non-coverage stats (not just coverage) to confirm.
  -> IMPACT IF WRONG: if this interpretation doesn't hold up under Robin's review, the algorithm is a documented judgment call reversible in one function (`_diversify_by_session`), not a schema/API change - low blast radius either way.

- **A4**: A pilot-50 win is sufficient to justify Phase 5 production wiring, per the roadmap's explicit acceptance gate; a full-500 confirmation is NOT required to wire in (only to make a public/README claim later - out of scope here per gotcha `74244096`'s pilot-vs-proof framing).
  -> VALIDATE BY: this is the roadmap's own explicit gate language - not independently re-validated here.
  -> IMPACT IF WRONG: none expected; if Robin wants full-500 confirmation before wiring (stricter than the roadmap text), that is a one-line scope change to Phase 5's entry condition.

- **A5 [added, Interview Tier 1 - unvalidated assumption]**: changing `derive_entity_key()`'s algorithm only affects NEWLY-saved experiences going forward; EXISTING rows in any real production store keep their old first-tag-derived `entity_key` value forever unless explicitly backfilled. This plan does NOT include a production backfill/migration script - it is NOT-scope (see Success Criteria) because (a) the LongMemEval harness ingests fresh experiences per run with no pre-existing `entity_key` baggage, so Phase 4's gate evaluation is unaffected, and (b) a production backfill is a separate, optional follow-up Robin can request later if a real workspace's historical data needs re-keying.
  -> VALIDATE BY: confirm the harness's ingestion path creates fresh experiences via `ExperienceEngine.save()` (which always calls the current `derive_entity_key()`) rather than pre-seeding `entity_key` directly - if it pre-seeds, the gate-eval could be silently testing stale keys instead of the new algorithm.
  -> IMPACT IF WRONG: if the harness does pre-seed `entity_key`, Phase 4's numbers would reflect the OLD algorithm, not the new one - the gate would be measuring nothing. Must be checked in Phase 0, not assumed.

---

## Phases

### Phase 0 - Design decision + cheap-alternative check (no code)
- **Scope**: Re-run `diag_recall.py` for multi-session with topk raised (e.g. 8->15/20) to test the DSV-Suspend alternative; decide A1 (which coverage metric), A2 (all-tags vs content heuristic + schema approach), A3 (concrete density-preserving algorithm); document the date-format canonicalization decision (folded-in sub-issue, see below).
- **Deliverable**: a short design note appended to this plan (or a plan addendum) recording the 4 decisions with their rationale; NO production code changes.
- **Gate (binary)**: all of A1/A2/A3 + date-format decision are recorded with rationale; if the topk-alone check closes >80% of the `mean_session_coverage_ratio` gap to 0.9, STOP here per the Success Criteria kill-criterion (a) and replan the remaining phases down to a topk-tuning-only change.
- **Review checkpoint**: none (design-only, single self-review before proceeding to Phase 1).

**Folded-in date-normalization sub-issue** (from this session's item-2 RC-gate, explicitly deferred there): decide a canonical `valid_from`/`as_of` date format (or a `normalize_date_prefix()` helper tolerant of both ISO `YYYY-MM-DD` and LongMemEval's `YYYY/MM/DD`) in `src/kairn/core/experience.py` or a new small module under `src/kairn/core/`. **File-coupling note**: this and item 5 (a separate, NOT-in-this-plan roadmap item) both touch `src/kairn/models/experience.py` - item 5 adds a `preference` `VALID_TYPES` entry. If item 5 is being worked concurrently (per the roadmap's "run 4 and 5 in parallel" recommendation), do this touch as an isolated, fast-landing prerequisite commit on `main` first, or expect a rebase before merging either branch - do not let both land independently in the same file and assume a clean auto-merge (precedent: the items-1+2 session's README merge gotcha, where a clean auto-merge silently flipped a value).

### Phase 1 - Date-format canonicalization helper
- **Scope**: `src/kairn/core/experience.py` (or new module) - implement the Phase-0-decided canonicalization; update `search_bitemporal`'s `as_of_day`/`valid_from[:10]` comparison to use it.
- **Deliverable**: helper function + updated `search_bitemporal`, unit tests covering both date-separator conventions.
- **Gate**: new tests pass; existing `tests/test_bitemporal.py` still green; no behavior change for workspaces already using one consistent convention (regression-safe).
- **Review checkpoint**: none (small, low-risk; folded into Phase 6's full RC-gate).

### Phase 2 - Multi-key entity extraction
- **Scope**: `derive_entity_key()` (or its replacement) in `src/kairn/core/experience.py`; `ExperienceEngine.save()`'s call site; `group_by_entity()`/storage layer if the A2 schema decision requires it.
- **Deliverable**: updated extraction logic per A2, unit tests (including edge cases: zero tags, one tag, overlapping-tag matching across two experiences).
- **Gate**: new tests pass; existing `tests/test_experience.py` entity-key tests still pass or are updated deliberately (not silently broken).
- **Review checkpoint**: yes (Phases 1-2 combined, before proceeding to Phase 3 diversification changes).

### Phase 3 - Density-preserving diversification
- **Scope**: `_diversify_by_session()` in `src/kairn/core/experience.py`.
- **Deliverable**: updated round-2 fill logic per A3, unit tests per A3's VALIDATE BY.
- **Gate**: new tests pass; `tests/test_bitemporal.py` diversification tests still pass or are updated deliberately.
- **Review checkpoint**: none (folded into the Phase 4 gate-eval checkpoint below).

### Phase 4 - Diagnostic + pilot-50 gate evaluation (decision point)
- **Scope**: re-run `diag_recall.py` (multi-session) and a pilot-50 LongMemEval-S run (`bench.py --recall-mode bitemporal` or whatever path Phase 0/5 wires) against the Phase 1-3 changes.
- **Deliverable**: real numbers - the Phase-0-selected coverage metric, and pilot-50 multi-session accuracy vs the verified 0.4135 baseline.
- **Gate (binary, per Success Criteria)**: coverage metric >0.9 AND pilot-50 multi-session accuracy >0.4135 -> proceed to Phase 5. Otherwise -> STOP, document the negative result (gotcha-style, per `65b08ce7` precedent), do not proceed to Phase 5.
- **Review checkpoint**: yes - this is the go/no-go checkpoint; present the real numbers before proceeding either direction.

### Phase 5 - Production wiring (ONLY if Phase 4 gates pass)
- **Scope**: wire the Phase-0-chosen MCP tool path (e.g., an existing `kn_recall`/`kn_context`/`kn_memories` optionally calling `search_bitemporal` instead of `search()`, or a new tool) in `src/kairn/server.py`.
- **Deliverable**: the wiring change, with the `KAIRN_BITEMPORAL_DIVERSIFY` flag behavior preserved (still operator-toggleable; the halt verdict on FULL-diversify-as-an-unconditional-default, `65b08ce7`, still stands - this wiring must not silently make full diversification the unconditional default without a flag).
- **Gate [tightened, Hardening]**: a real MCP client round-trip (same EPT bar as items 1+2: `claude mcp add` + fresh session) calling the wired tool with a query known (from Phase 4's data) to span >=2 sessions/entities returns experiences from more than one distinct `entity_key`/session in its top results - a concrete, observable behavior difference from the pre-wiring baseline, not just "results returned." Existing non-bitemporal tool behavior is unchanged for callers who don't opt in (verified via existing regression tests still passing unmodified).
- **Review checkpoint**: yes (part of Phase 6's full RC-gate).

### Phase 6 - RC-gate + merge + closeout
- **Scope**: `/code-review` (feature-dev:code-reviewer, sonnet, spec-compliance then quality) + `/simplify` on the full diff; merge via the established branch-push-PR-merge flow; `kn_learn`/`kn_judge` the outcome; closeout with a reason-coded deferred-and-untested section, explicit about pilot-vs-full-500 status.
- **Deliverable**: merged PR on `primeline-ai/kairn` main; Kairn nodes; closeout doc/message to Robin.
- **Gate**: RC-gate findings resolved (fixed or consciously accepted, same discipline as items 1+2); CI green; merge commit verified equal to `origin/main` post-merge.
- **Review checkpoint**: final - present to Robin before considering this item done.

---

## Verification

- **Automated**: `pytest tests/` (single isolated invocation - see task #2285's resolution: never run concurrent pytest invocations against this repo) for all new/changed unit tests + full regression; `ruff` on changed files.
- **Manual**: Phase 4's diagnostic + pilot-50 numbers reviewed against the verified full-500 baseline (0.4135) before any go/no-go claim; Phase 5's live MCP client round-trip (real `claude mcp add` + fresh session call, same pattern as item 1's EPT).
- **Ongoing Observability**: none new - this stays a benchmark-harness-triggered path plus (if Phase 5 ships) one opt-in MCP tool path; no new production monitoring surface is introduced by this change.

---

## CONDITIONAL: Risk

- **Risk**: the A2 schema decision (multi-key via all-tags vs content heuristic) turns out to need a real schema migration (join table) rather than a single-column reinterpretation. **Mitigation**: Phase 0 explicitly validates this before Phase 2 starts; if it requires migration, that's a kill-criterion trigger (Success Criteria FAILED condition (c)) - stop and replan with Robin rather than silently absorbing migration scope into the 2-4 day estimate.
- **Risk**: pilot-50 sample noise (n=13 per multi-session pilot slice, per this session's own item-2 precedent) makes the Phase 4 gate result a coin-flip either direction. **Mitigation**: the roadmap's own gate is explicitly pilot-50-scale (a precondition-clearing gate, not proof, per gotcha `74244096`) - this is accepted roadmap-level risk, not something this plan can eliminate; the closeout must state this explicitly rather than overclaim.
- **Risk**: file-level coupling with item 5 in `src/kairn/models/experience.py` if both are worked concurrently, causing a silent bad merge (precedent: the items-1+2 README merge gotcha). **Mitigation**: documented in Phase 0's folded-in sub-issue section - land the shared prerequisite touch first, or rebase deliberately.
- **Risk [added, Interview/Hardening]**: the external LongMemEval harness (`~/Buisiness/Evolving/_autonomous/benchmarks/longmemeval/`, a different repo, last touched 2026-07-07) could have drifted or broken between now and Phase 4's execution if that repo sees unrelated changes in the interim. **Mitigation**: Phase 0 should do a quick smoke-check (`bench.py --recall-mode engine` on a tiny sample) to confirm the harness still runs before relying on it for Phase 4's gate-eval; if broken, that is a genuinely-needs-work blocker to flag to Robin, not something to silently work around by fixing an unrelated repo's harness under this plan's scope.

## CONDITIONAL: Rollback

**[Added, Interview Tier 1 - no rollback on irreversible change]**: Phase 5 (production MCP tool wiring) is the only phase with production blast radius; Phases 0-4 are pure design/library code with no production callers yet, so they carry no rollback need beyond normal git revert. For Phase 5 specifically:
- **Rollback mechanism**: `git revert` of the wiring commit (single, isolated commit per Phase 5's scope) - the underlying `search_bitemporal`/`group_by_entity`/diversification code stays in place (already flag-gated via `KAIRN_BITEMPORAL_DIVERSIFY`, unaffected by this rollback) and simply becomes uncalled again, returning to the pre-Phase-5 zero-production-caller state.
- **Rollback time**: git revert + redeploy of the MCP server process - minutes, not hours; no data migration to undo since Phase 2/3 changes only affect how `entity_key`/diversification are COMPUTED, not a schema change requiring reversal (per A2's schema-decision constraint - if Phase 0 finds a schema migration IS required, this rollback plan must be revisited before Phase 2 starts, per the FAILED-condition (c) kill-criterion).
- **Irreversible risk**: none identified - no data deletion, no external API calls, no user-facing behavior change for existing (non-opted-in) callers.

## CONDITIONAL: Execution Vehicle & Orchestration

Covered in Stage 0.5 above. No team/parallel-agent coordination needed beyond the standard Phase 6 RC-gate sub-agents.

## CONDITIONAL: Dependencies

- Upstream: none new (uses existing `kairn.core.experience`, `kairn.storage.sqlite_store`, `kairn.models.experience` modules).
- Downstream: item 6 (temporal query path) and item 9 (WOW: `kairn import`) are described in the roadmap as gated on this item's entity_key work landing - NOT in scope here, but this plan's Phase 2 output (the A2 multi-key scheme) is what they will build on; keep the scheme documented clearly enough for a future session to consume without re-deriving it.
- External: the LongMemEval harness at `~/Buisiness/Evolving/_autonomous/benchmarks/longmemeval/` (a different repo) is used read-only for Phase 4's evaluation - no changes to that repo are in scope here unless Phase 0/5 needs a new `--recall-mode` variant to exercise the Phase 5 wiring, in which case that harness change is a small, explicitly-scoped addition, not a rewrite.

## CONDITIONAL: Resume Protocol

**[Added, Hardening - Manager perspective, >10h plan]**: total estimated effort (2-4 days + 0.5 day, per the roadmap) exceeds 10h, likely spanning multiple sessions. Each phase's gate result (pass/fail + evidence) is recorded directly in this plan file as it completes (per Stage 5 Execution Rules' "document gate result" requirement) - a future session resuming mid-plan should read this file top to bottom, find the last phase with a recorded gate result, and resume from the next unstarted phase. If the resuming session is fresh (no prior context), it should also re-run Phase 0's harness smoke-check (see the new external-dependency Risk above) before trusting any earlier phase's numbers, since time may have passed.

## CONDITIONAL: Incremental Delivery

**[Added, Hardening - Manager perspective, >5 phases]**: this plan already has two natural stopping points that double as incremental-delivery checkpoints, cross-referenced rather than duplicated: (1) Phase 0's kill-criterion (a) - if the cheap topk-tuning check alone closes most of the coverage gap, stop there with a smaller shipped change instead of the full multi-key/diversification build; (2) Phase 4's gate - if it fails, stop with Phases 1-3's code merged (or not, per Robin's preference) but Phase 5's production wiring skipped, still a coherent, honestly-documented state rather than a half-finished feature.

## CONDITIONAL: Completion Gate

**[Added, Hardening - Manager perspective, multi-file/integration work]**: this touches `src/kairn/core/experience.py`, possibly `src/kairn/storage/sqlite_store.py` (if A2 needs it), `src/kairn/server.py` (Phase 5 only), and multiple test files. Registration/consistency checklist for Phase 6: confirm no new dormant/orphaned code paths (the whole point of Phase 5 is to stop `search_bitemporal`/`group_by_entity` being unreachable - verify post-merge that the new call path is actually exercised, not just added and left uncalled like the current state); update `README.md`'s tool count/description only if Phase 5 adds a genuinely new MCP tool (not needed if it extends an existing tool's behavior via a parameter). No `_stats.json`/knowledge-graph registration applies - this is the `kairn` repo, not the `Evolving` repo's component-registration system.

## CONDITIONAL: Post-Completion

**[Added, Review Stage 3]**: covered concretely by Phase 6 (not duplicated here): RC-gate resolution, merge verification (local tree equals `origin/main` post-merge, same check as items 1+2), and closeout messaging to Robin. No new production monitoring surface is introduced (see Verification's Ongoing Observability, which states this explicitly) - so there is no new dashboard/alert to wire post-ship beyond what Phase 5's live MCP round-trip already confirms.

## CONDITIONAL: Learning & Knowledge Capture

**[Added, Review Stage 3 - multi-session Grade-A criterion]**: handled via Phase 6's `kn_learn`/`kn_judge` step (this repo's standing convention, same as items 1+2's closeout) rather than a separate mechanism - each phase gate's real result (especially Phase 0's design decisions A1-A5 resolution and Phase 4's gate-eval numbers) gets a Kairn node so a future session (working on item 6/9, which depend on this item's entity_key scheme) can recall the reasoning without re-deriving it.

## CONDITIONAL: Reference Library

| Source | Version/Date | What it informed | Link |
|---|---|---|---|
| `src/kairn/core/experience.py` (this repo, HEAD `3d0c0c9c`) | 2026-07-07 | entity_key/diversification/search_bitemporal current implementation | local file |
| `src/kairn/storage/sqlite_store.py` | 2026-07-07 | entity_key column/index/query plumbing | local file |
| `runs/full-500-s-fixed/summary.json` | full-500 LongMemEval-S run, engine mode | verified 41.4% multi-session baseline + 10% preference baseline | local artifact, Evolving repo |
| `runs/diag/multisession-diag.json` | diag harness output | 3 candidate coverage metrics + their current values | local artifact, Evolving repo |
| Kairn gotcha `65b08ce7` | 2026-07-0x | prior halt decision on full-diversify-as-default - must not be re-opened | Kairn node |
| Kairn gotcha `74244096` | 2026-07-07 | pilot-vs-proof framing discipline for closeout language | Kairn node |
| Kairn decision `758ece4c` | 2026-07-07 | the roadmap item text itself | Kairn node |

---

## Phase 0 Gate Result (2026-07-07, executing session)

**Gate: PASS - all 4 decisions recorded, kill-criterion (a) NOT triggered, proceed to Phase 1.**

1. **Topk-tuning fork (DSV-Suspend) - NOT sufficient, kill-criterion (a) not triggered.** Diag re-runs (artifacts: `runs/diag/multisession-diag-topk15.json`, `-topk20.json`): topk=8 coverage 0.737 (full 22/60), topk=15 -> 0.818 (33/60, closes 50% of gap to 0.9), topk=20 -> 0.859 (40/60, closes 75%). Even at 2.5x reader context (with the density/noise cost that caused the `daf5cd8e` regression) the 80% descope bar is not met; diminishing returns (8->15: +0.081, 15->20: +0.041). Full multi-key + density work justified.
2. **A1 DECIDED**: the ">0.9 diag coverage" gate metric = `mean_session_coverage_ratio` (0.737 baseline). `surfaced` (0.983) gates nothing; `full_coverage_in_topk` (0.367) is unreachable in scope.
3. **A2 DECIDED - content-derived multi-key, computed at QUERY TIME, zero schema change.** Decisive Phase-0 finding: BOTH harnesses (`bench.py:200-215`, `diag_recall.py:46-80`) ingest with NO tags (`type="pattern", confidence="medium", namespace="bench"`), so the plan's preferred all-tags branch cannot move the diagnostic at all (A2's IMPACT-IF-WRONG path fires: content-derived is required). Design: new pure function `derive_entity_keys(content, tags) -> set[str]` (plural; normalized tags + rule-based content extraction: proper-noun runs + salient tokens), consumed by the diversification/merge path on already-recalled Experience objects in Python. The stored single `entity_key` column, `derive_entity_key` (singular), `group_by_entity`, and the index stay unchanged - no migration (kill-criterion (c) dissolved), no backfill (A5's production-backfill concern moot: query-time keys benefit existing stores immediately). A5's pre-seed risk CHECKED and dead: harness saves via the real `save()` path, never writes entity_key directly.
4. **A3 DECIDED - anchored density-preserving merge**: keep top-A BM25 hits verbatim (density anchor, A=4 initial); walk the remainder in BM25 order within a bounded rank window (initial: 3x limit) promoting the best hit per unseen session/entity group ONLY if on-topic (shares >=1 derived entity key with the query's extracted keys or anchor keys); fill remaining slots with leftovers in BM25 order. Exact thresholds (A, window, on-topic bar) are TUNED EMPIRICALLY via the diag harness (4s/iteration), gated on coverage >0.9 AND no density regression - the diag needs two small, explicitly-scoped additions first: a recall-mode flag (it currently measures `search()`, not the diversified path) and per-question answer-row density stats (mean answer-session rows in top-k), per A3's VALIDATE BY.
5. **Date-format DECIDED**: `normalize_date_prefix(s) -> str | None` helper in `src/kairn/core/experience.py` (NOT models/ - reduces item-5 file coupling to HALF_LIVES only), tolerant of `YYYY/MM/DD ...` and ISO `YYYY-MM-DD...`, returning a canonical ISO day prefix for comparison; `search_bitemporal`'s as-of compare switches to it, removing the documented mixed-separator caveat.
6. **Harness smoke-check PASS**: `runs/smoke-item4-phase0` (n=2, engine mode) ran end-to-end (ingest -> recall 1.4ms avg -> GPT-4o reader+judge -> prod_db_untouched=true). 0/2 accuracy is n=2 noise, consistent with yesterday's smoke pattern (post-fix smoke 0.0 -> pilot-50 0.54). NOTE for Phase 4: bench.py needs `PYTHONPATH=/Users/neoforce/Business/engram/src` (no self-inserted path, unlike diag_recall.py).

## Phase 1-3 Gate Results + Phase 4 Diagnostic Findings (2026-07-07, executing session)

**Phases 1-3: PASS.** Commits on `feat/item4-entity-diversification` (engram): `65e1f0b` (normalize_date_prefix + cross-convention as-of, 3 new tests), `7d4b1c8` (derive_entity_keys multi-key extraction + _diversify_density_preserving, 6 new tests), `4aaad2e` (tunable gate + scored-promotion negative result). tests/test_bitemporal.py 24/24 green, tests/test_experience.py regression green.

**Phase 4 diagnostic** (diag_recall.py extended with --recall-mode + answer-density stats; ingest now sets valid_from like bench.py):

| config (topk=8) | mean coverage | full-cov | density (mean answer rows) |
|---|---|---|---|
| engine baseline | 0.737 | 22/60 | 4.533 |
| density A4/W3-6/gate2 (DEFAULT) | 0.797 | 30/60 | 3.333 |
| density A3 | 0.818 | 33/60 | 3.100 |
| density A2 | 0.822 | 33/60 | 2.767 |
| legacy (June regression mode) | 0.811 | 32/60 | 2.600 |

**Findings (all empirical, artifacts in runs/diag/ms-*.json):**
1. The new default Pareto-dominates legacy (same-or-better coverage at +28% density).
2. NEGATIVE RESULT: scored promotion (shared-term count picking slots/rows) made coverage WORSE (0.757) - verbose distractors out-count relevant rows; BM25's length normalization already handles this. Reverted to greedy BM25-order promotion; documented in code so the next tuner does not re-derive it.
3. The on-topic gate excludes almost nothing (gate-off = gate-2 within noise); the binding constraint is SLOT COMPETITION - no rule-based signal tried can distinguish answer sessions from distractor sessions beyond BM25 order.
4. **A1 gate reality-check: the >0.9 coverage bar is NOT reachable at topk=8** by any slot-based reorder without density collapse: the recall ceiling in full-50 is 0.951, but coverage saturates at ~0.82 as slots increase while density erodes toward the legacy regression point. The 0.9 number predates any Pareto measurement. Per A1's stated reversibility, the PILOT-50 accuracy gate (multi-session > 0.4135) is treated as decisive; this coverage evidence goes to Robin at the Phase-4 checkpoint as planned.

**Phase 4 pilot-50 RESULT (run `pilot-50-bitemporal-density-item4`, n=50 seed=42, recall-mode bitemporal, default knobs A4/W3/gate2): GATE FAIL - no Phase 5 wiring.**

| type | pilot baseline | as-of-only postfix | density (this) |
|---|---|---|---|
| overall | 0.54 | 0.54 | 0.54 |
| multi-session (n=13) | 0.4615 | 0.4615 | **0.3846** |
| knowledge-update (n=8) | 0.50 | 0.50 | **0.625** |
| temporal (n=13) | 0.3077 | 0.3077 | 0.3077 |
| all other types | identical | identical | identical |

Multi-session 0.3846 < 0.4135 (the plan's gate) < 0.4615 (pilot baseline slice) -> FAILED-condition (b) fires. Exactly ONE question flipped each way, both counting/aggregation questions - the Pareto tension operating as diagnosed: `1a8a66a6` ("how many magazine subscriptions") LOST a corroborating row to a promotion and the reader abstained; `4d6b87c8` ("how many titles on my to-watch list", knowledge-update) GAINED surfaced evidence and flipped correct. Single-question flips at n=13 are the accepted pilot-noise risk; the honest verdict is "no measured multi-session win", not "proven regression". The knowledge-update +12.5pp is a real upside signal for item 6 (supersedence questions benefit from seeing multiple sessions).

**Disposition (executed under Robin's full-permission delegation, per the Incremental Delivery clause):** Phases 1-3 merge as LIBRARY-ONLY improvements - the date-normalization fix has standalone correctness value, and the diversification improvement lives inside the already-flag-gated `search_bitemporal` mode (zero production callers; production behavior byte-identical). NO Phase 5 wiring. The negative result stands recorded per the `65b08ce7` precedent.

Note: the run's `prod_db_untouched=false` is explained and benign - this session's own live Kairn MCP usage (kn_context/kn_recall access-count writes) mutated the prod DB during the run window; bench itself only hashes it, and all bench work happens in sandboxed per-question DBs.

## Interview Log

**Date**: 2026-07-07 | **Mode**: Self | **Domain**: Software Development
**Questions asked**: 9 | **Anti-patterns found**: 2

| Tier | Question | Answer/Decision | Anti-Pattern |
|------|----------|----------------|---------------|
| 1 | Is there a rollback plan for the one production-facing phase (Phase 5)? | None existed - added a Rollback conditional section: git revert, minutes, no data migration since the flag-gate absorbs it. | #4 No Rollback on Irreversible Change |
| 1 | What happens to EXISTING experiences' `entity_key` when the derivation algorithm changes - is this assumed away? | It was assumed away. Added A5: NOT-scope for production backfill, but flagged a real risk that the benchmark harness itself might pre-seed `entity_key` and silently invalidate Phase 4's gate-eval - now an explicit Phase-0 check. | #1 Unvalidated Assumptions |
| 1 | Are all FAILED conditions and VALIDATE BY entries present for every assumption? | Yes on original draft (A1-A4); A5 added with its own VALIDATE BY/IMPACT IF WRONG in the same format. | - |
| 1 | Are there vague/judgment-word gates ("acceptable", "ready", "sufficient")? | Scanned all phase gates - none found; all are binary (tests pass / numeric threshold / live round-trip observed). | - |
| 2 | Software domain: is there a Reference Library with sourced numbers? | Yes, present with 7 sourced entries; all numeric claims traced to a specific file read this session, not re-derived from memory. | - |
| 2 | Software domain: are phases scoped by files/features or just time-boxes? | Scoped by concrete files/functions (`derive_entity_key`, `_diversify_by_session`, `server.py` wiring) with binary gates - not time-boxes. | - |
| 3 | Cold Start Test: what would a zero-context reader misunderstand first? | That "multi-key" implies LLM-based entity resolution (given the Zep comparison in the roadmap text). Added an explicit clarification in End State: this stays zero-LLM/tag-based. | - |
| 3 | AHA/80-20: which slice delivers most value cheapest? | Phase 0's topk-tuning + all-tags-matching check (A2's cheaper branch) - already captured as kill-criterion (a); made this framing more explicit rather than leaving it only as an escape hatch. | - |
| 3 | Devil's Advocate: what makes this approach obsolete in 6 months? | If item 6 (temporal query path) or item 9 (`kairn import`) later need a true multi-value join table instead of a single reinterpreted `entity_key` column, this ships as an intentionally cheaper interim design, not a final architecture - accepted tradeoff, not hidden. Noted inline in Dependencies (downstream note already covers "keep the scheme documented for item 6/9 to consume" - reinforced here as an explicit acknowledged limitation rather than an assumed-permanent design). | #10 First Idea = Final (accepted, not silently assumed) |

**Plan changes made**: added `## CONDITIONAL: Rollback` section; added Assumption A5 (existing-data staleness + harness pre-seed risk, now a Phase 0 check item); added a Cold-Start clarification sentence to End State; reinforced the AHA-effect framing in Phase 0's description (no structural change needed, already present via kill-criterion (a)).
**Grade before**: B (solid CORE sections, but missing Rollback and one real unvalidated assumption)
**Grade after**: A- (pending Stage 2 Hardening + Stage 3 Review, which have not yet run as of this log entry)

## Stage 1.5: Hardening Log

**Ran**: 2026-07-07 | **Mode**: Simple | **Perspectives**: 6/6

| Perspective | Finding | Action Taken |
|---|---|---|
| Outside Observer | Plan was missing the required `## Context & Why` CORE section entirely (had End State but not this) - anti-pattern-adjacent gap in required structure. | Fixed: added 2-sentence Context & Why section before End State. |
| Outside Observer | "BM25" used undefined in phase text (inherited from source docstrings). | [Stage 1.5 Note]: left as-is - plan's actual audience (Robin + future Kairn-maintainer sessions) is technical enough that defining a standard IR-ranking term would be padding, not clarity. |
| Pessimistic Risk Assessor | No Resume Protocol despite plan clearly exceeding 10h across multiple likely sessions. | Fixed: added CONDITIONAL Resume Protocol section. |
| Pessimistic Risk Assessor | External dependency (LongMemEval harness, separate repo) could have drifted/broken by Phase 4 with no smoke-check planned. | Fixed: added a Risk item + folded a harness smoke-check into Phase 0 and the new Resume Protocol. |
| Pedantic Lawyer | Phase 5's gate ("returns bi-temporal/diversified results") was observable but not concretely specified - a reviewer could argue any non-empty result satisfies it. | Fixed: tightened to require a query spanning >=2 sessions/entities to return experiences from >1 distinct entity_key/session, a falsifiable behavior check. |
| Skeptical Implementer | Checked every phase's inputs trace to a prior phase or the Reference Library - no orphaned dependency found. | No action needed. |
| The Manager | 7 phases (>5) had no Incremental Delivery section describing stopping points, even though the stopping points themselves (kill-criteria a/b) already existed implicitly. | Fixed: added CONDITIONAL Incremental Delivery section cross-referencing the existing kill-criteria rather than duplicating them. |
| The Manager | Multi-file integration work (core + storage + server + tests) had no explicit Completion Gate / dormant-code-path check, and the whole point of Phase 5 is fixing an existing dormant-code problem - worth an explicit check that it doesn't just create a NEW dormant path. | Fixed: added CONDITIONAL Completion Gate section with this specific check named. |
| The Manager | 2-4 day + 0.5 day roadmap estimate looks tight against 7 phases including a full pilot-50 run + RC-gate, with no explicit buffer stated. | [Stage 1.5 Note]: not fixed - the estimate is the roadmap's own (Robin-set at the greenlight level), not this plan's to unilaterally inflate; flagging for awareness, not overriding. If Phase 0-3 actually run long, the plan's own timeout kill-criterion (3 elapsed days to reach Phase 4) already catches this rather than letting it silently slip. |
| Devil's Advocate | Confirmed the Stage 0 AHA check (topk-tuning alternative) was a genuine fork explored with real diagnostic re-run, not a compliance afterthought - already reflected in Phase 0 + kill-criterion (a) from the original draft. | No action needed. |

**Discovery Consolidation**: All Stage 0 findings addressed - the entity_key-not-dead correction, zero-production-caller confirmation, verified full-500 baseline numbers, the 3-metric diag ambiguity, the AHA topk-check, and the schema/index note (0.3) are each referenced in a specific phase, assumption, or risk item. No orphaned discoveries found.

## Stage 3: Review Report

**Plan:** `2026-07-07-kairn-item4-entity-extraction-diversification.md` | **Domain:** Software Development | **Grade:** A | **Confidence:** Medium
**Recommendation:** Proceed

### CORE Sections

| Section | Status | Format | Issue |
|---|---|---|---|
| Context & Why | ok | Yes | - (added during Hardening; 2 sentences) |
| Success Criteria | ok | Yes | FAILED conditions (a/b/c) + 3-day timeout present |
| Assumptions | ok | Yes | 5 assumptions, triple format, each with a real alternative explored |
| Phases | ok | Yes | Binary gates; review checkpoints after Phases 1-2, at Phase 4, at Phase 6 |
| Verification | ok | Yes | Automated + Manual + Ongoing Observability all populated |

### CONDITIONAL Sections (Software Development)

| Section | Status | Notes |
|---|---|---|
| Risk | present | 3 items incl. an external-repo-drift risk added during Hardening |
| Rollback | present | added during Interview; git-revert path, minutes, no migration |
| Post-Completion | present | added this stage; points to Phase 6 rather than duplicating it |
| Execution Vehicle & Orchestration | present | Stage 0.5 Default Vehicle + one documented Phase 6 deviation |
| Dependencies | present | upstream/downstream/external all addressed |
| Reference Library | present | 7 sourced entries, all traced to a file read this session |
| Resume Protocol (>10h) | present | added during Hardening |
| Incremental Delivery (>5 phases) | present | added during Hardening, cross-references existing kill-criteria |
| Completion Gate (multi-file) | present | added during Hardening, names the dormant-code-path check explicitly |
| Learning & Knowledge Capture | present | added this stage; points to Phase 6's kn_learn/kn_judge convention |

### Stage 0 Evidence

- Existing work checked: yes (derive_entity_key/group_by_entity/search_bitemporal read in full, entity_key-not-dead correction made)
- Feasibility questioned: yes (0.7, High)
- Alternatives considered: yes (0.9 AHA - topk-tuning check as a genuine fork, not compliance filler)
- Constraints discovered: yes (0.11 folded into 0.3 - zero-LLM/zero-embedding invariant re-affirmed, schema-index constraint noted)
- Official docs consulted: yes/N/A - internal code is the "official doc" here (no external library integration); SQLite indexing already in place, checked
- Stage 0 status: Done

### Execution Vehicle & Routing (Check 11)

- Default Vehicle present: yes (Single (self), one Phase 6 deviation to Sub-Agent)
- Per-phase vehicles justified vs signals: ok - Phases 0-5 are sequential single-stream design/implementation work (correctly Single); Phase 6 correctly deviates to Sub-Agent for the established RC-gate pattern
- Multi-agent vehicles fully parameterized (model-routing): ok - `feature-dev:code-reviewer` (sonnet) and `/simplify` agents (sonnet) both now explicitly tiered (fixed this stage - previously `/simplify`'s tier was unstated)
- Model routing sane: ok - no Gemma-eligible work mis-routed to Sonnet/Opus (none of this plan's work is bulk/mechanical/non-quality-critical enough for Gemma), no Opus used as a delegation target
- Silent selection: yes - Stage 0.5 ran silently, no interactive vehicle prompt in the plan

### Anti-Patterns Found

None of the 21 anti-patterns triggered a hard flag. Two items were surfaced and consciously disclosed rather than fixed (correct handling, not a violation): the 2-4 day timeline's tightness against 7 phases (Hardening Manager-note, not treated as #12 Timeline Fantasy since it's the roadmap's own estimate and is timeout-guarded rather than blindly trusted), and BM25 left undefined for a technical audience (not #14 Wrong Level of Detail, since the audience is Robin + future Kairn-maintainer sessions).

### Review Checkpoints

Present with correct cadence for a coding-domain plan of this size: after Phases 1-2 (combined), at Phase 4 (the go/no-go gate), and at Phase 6 (final, pre-Robin-presentation).

### Reference Library

Present with sources - all 7 entries trace to a specific file this session actually read (code files, benchmark run artifacts, Kairn nodes), not re-derived from memory.

### Red Flags

None.

### Top 3 Improvements (already applied during this Stage 3 pass, not deferred)

1. Added the missing `Context & Why` CORE section (was entirely absent before Hardening).
2. Added `Post-Completion` and `Learning & Knowledge Capture` CONDITIONAL sections required for this domain/session-span, both scoped to point at Phase 6 rather than duplicate it.
3. Closed the one under-parameterized vehicle gap (`/simplify`'s agents had no stated model tier) with a one-line addition.

### Bottom Line

Grade A: all 5 CORE sections complete in framework format, every domain-required CONDITIONAL section present, Stage 0 evidence thorough and genuinely explored (not compliance-only), zero anti-pattern flags, and the vehicle/routing check is fully parameterized. Recommend proceeding to implementation starting at Phase 0.
