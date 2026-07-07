# Plan: Kairn Roadmap Item 5 - Preference Capture via Host Model ("MCP Judo")

**Date**: 2026-07-07
**Confidence**: Low, downgraded post-Review (see CRITICAL CORRECTION above) - a prior 2026-06-13 investigation already fired kill-criterion K2 on a closely-related intervention and made a standing decision NOT to ship a preference-specific engine fix. Phase 0 must reconcile with that finding (not just cite it) before Phase 2 is attempted; per UPF's own rule, Low confidence means Phase 0 IS the validation sprint - do not treat it as a formality.
**Quality Grade**: A (Stage 3 review complete - see Stage 3 Review Report at end of file)
**Default Vehicle**: Single (self) - see Stage 0.5

---

## CRITICAL CORRECTION (post-Review, found via kn_judge candidate surfacing - read before anything else below)

This session's Stage 0 investigation (the abstention/synthesis finding below) was **not novel** - it substantially duplicates a more rigorous prior investigation already in Kairn: node `bf00180c`/`20228a20` (2026-06-13, "PHASE 0a VERDICT", artifacts already in this repo: `diag_recall.py`, `read_ceiling_probe.py`, `runs/diag/preference-diag.json`, `runs/diag/read-ceiling-probe.json`). That prior work:
- Ran the same 83%-surfaced diagnostic this session re-derived.
- Went further: it actually **tested** a "preference-aware reader prompt" (informing the reader it should look for preferences) against the 8 surfaced questions and got only **1/8 correct** (up from 0/8) - a marginal, not a meaningful, lift.
- Root-caused it MORE PRECISELY than this session's initial framing: the real mechanism is an **entailment mismatch**, not merely "no explicit statement." LongMemEval's preference gold answers are META-DESCRIPTIONS of a preference ("user prefers Adobe Premiere Pro advanced-settings resources"), while the question is a surface-level request ("recommend video editing resources"). The harness's judge checks entailment against the meta-description; a good, preference-informed RECOMMENDATION does not entail that meta-description even when it correctly applies the preference.
- Explicitly fired **kill-criterion K2** (preference accuracy staying under 25% after a cheap fix attempt -> deeper gap, stop cheap-fix framing) and made a standing **DECISION: do NOT ship a preference-specific engine fix; document as a read-ceiling; redirect flagship effort to temporal-reasoning (42.9%) and multi-session (41.4%), where real headroom x volume sits.**

**What this changes about THIS plan**: Phase 1 (the `preference` type + dedicated MCP tool for real host-model usage, fixing the live `type=preference` rejection bug) is UNAFFECTED - it has independent production value regardless of any benchmark number and should proceed. **Phase 2 (the harness-simulation gate) is now materially higher-risk than stated below**: my approach (write-time synthesis, producing genuinely different STORED content) is a different intervention point than what was already tested (read-time reader-prompt awareness, same raw stored content) - so it is not strictly disproven by the June 13 finding. But the underlying entailment-mismatch mechanism suggests even a well-synthesized preference statement may not survive the judge's entailment check unless the READER's answer format ALSO becomes meta-descriptive rather than a generic recommendation - which the June 13 test partially explored and found only marginally effective.

**This is a genuine decision point for Robin, not something to silently resolve**: did Robin's 2026-07-07 greenlight of item 5 (via the 36-agent roadmap workflow) account for this prior K2-kill decision, or did the roadmap workflow miss it? Recommend surfacing this explicitly before committing to Phase 2's implementation work, rather than proceeding as if this were undiscovered territory. Phase 0 below is revised accordingly: it must explicitly reconcile with (not just cite) the June 13 artifacts before any new code is written for Phase 2.

---

## DSV Pre-Check

**Decompose** - the roadmap ask (Kairn decision `758ece4c`) bundles three separable claims: (1) a dedicated preference-capture MCP tool surface for real host-model usage, (2) a `preference` experience type with a long half-life, (3) preference-lexicon rules for imports/benchmark ingestion.

**Suspend (the alternative interpretation that turned out to be load-bearing)** - the roadmap frames this as a capture/storage-mechanism gap (implying: if Kairn just tagged and decayed preferences correctly, accuracy would rise). Stage 0 investigation below shows this is **not** the actual bottleneck: recall already surfaces the answer-bearing content 83% of the time (25/30, `preference-diag.json`), yet end-to-end accuracy is only 10% because the **reader abstains on 27/30 (90%) of preference questions** even when the right raw excerpt was in front of it. The real gap is that Kairn stores raw, unsummarized conversational turns, and the benchmark's reader (`bench.py`'s `READER_SYS`: "If the excerpts do not contain the needed information, reply exactly: I don't have that information in my memory") won't synthesize an implicit preference from scattered raw scraps under that conservative instruction - it needs an **explicit, quotable preference statement** in the recalled content, not just the raw material a preference could be inferred from. This validates the roadmap's core hypothesis (host-model synthesis at write time is the fix) but for a more specific reason than the roadmap stated, and sharpens the risk on the gate: a dedicated tool must reach the benchmark's ingestion path to be measurable at all (see A2).

**Validate (least-sure claim)** - whether a harness-side simulation of "host model calls the new tool" (transforming raw preference-bearing turns into explicit preference statements at ingestion time) is actually achievable without introducing an LLM into the harness's ingestion step in a way that confounds the "zero-LLM-in-server" positioning. Resolved in Phase 0/A2: the LLM doing the synthesis is the harness's OWN reader-side simulation of a host model (analogous to how real Claude Code usage has the calling LLM, not Kairn's server, do the extraction) - this is architecturally consistent with "MCP judo," not a violation of it, since Kairn's server-side code still does zero LLM calls.

---

## Context & Why

Kairn's LongMemEval-S single-session-preference accuracy is 10% against mem0's self-reported 96.7% and even a raw full-context baseline's 20-30% (Kairn pattern `36f1309d`) - the single worst verified competitive number in the whole roadmap. Stage 0 investigation (below) found the mechanism is not primarily a storage/decay problem (recall already works 83% of the time) but a **synthesis-explicitness problem**: raw stored turns don't give the reader an extractable fact to answer from. This item exists to close that gap the way ChatGPT Memory does structurally (host-model synthesis at capture time) while preserving Kairn's zero-LLM-server architecture.

## End State

After this plan succeeds, Kairn has a `preference` experience type (long half-life, unblocking the existing type-rejection bug hit live on 2026-06-11/2026-07-02 - gotchas `74eec379`/`c58cf81d`/`c5e9576c8`/`ef485245`), a dedicated MCP tool surface for real host-model preference capture, and - if Phase 2's harness-simulation gate passes - measured evidence of whether explicit preference synthesis at write time actually improves the 10% baseline. If the gate fails, that is an honestly documented negative result (matching the `65b08ce7` precedent), not a forced ship.

**[Clarified, Cold-Start]**: "MCP judo" means the CALLING LLM (Claude Code, or any MCP client's host model) does the preference extraction/synthesis by calling a new tool with an explicit statement it already composed - Kairn's server never runs an LLM itself. The harness-simulation in Phase 2 plays the role of "a host model calling the tool" using the SAME reader-model call the benchmark already makes elsewhere (not a new, architecture-violating LLM-in-server step).

## Success Criteria

- `"preference"` added to `VALID_TYPES` (single source of truth, `src/kairn/models/experience.py`), unblocking the existing rejection bug for real Claude-Code/Evolving usage immediately, independent of the benchmark gate below.
- A long half-life value set in `HALF_LIVES` for `"preference"`, documented as an **initial estimate** (no real access-tail data exists for a type that doesn't exist yet - unlike the 2026-06-13 tail-calibration of the other 5 types), not a measured calibration.
- A dedicated MCP tool (name/shape decided in Phase 0) that wraps `experience.save(type="preference", ...)`, live-verified via a real MCP client round-trip (same EPT bar as items 1+2).
- **The benchmark gate (Phase 2)**: pilot-50 (or a preference-specific subset re-run) with the harness's ingestion path modified to simulate host-model preference synthesis, showing single-session-preference accuracy **> 0.10** (the verified full-500 baseline from `runs/full-500-s-fixed/summary.json`, not a re-derived number).
- **NOT-scope**: full-500 re-run (pilot-scale is the gate-decision per the roadmap and this session's explicit scope cap); WOW items 8-10; preference-lexicon-based import detection for arbitrary external sources (e.g. `kairn import claude-code`, WOW item 9) - only the benchmark-harness-internal simulation needed to test THIS item's hypothesis is in scope, a general-purpose lexicon/import classifier is a larger, separate follow-up; changing the benchmark's `READER_SYS` prompt itself (that is the harness's own calibration, not Kairn's - if the reader's conservatism is the wrong thing to fix, that's a harness discussion with Robin, not a change this plan makes unilaterally).
- **FAILED conditions (kill criteria)**: (a) if Phase 0 finds the dedicated-tool-vs-extend-kn_learn decision requires more than a thin wrapper (e.g., a new structured schema beyond content+tags), stop and replan scope with Robin rather than silently absorbing a bigger design; (b) if the Phase 2 gate fails (accuracy does not beat 0.10), stop - document the negative result per the `65b08ce7` precedent, ship only the production-facing tool (which has independent value regardless of the benchmark number) with an honest "benchmark impact unproven" note; (c) if simulating "host model calls the tool" in the harness turns out to require an architecture the benchmark can't support without a large rewrite (e.g., a full agentic loop instead of a single reader call), stop and escalate to Robin - this is a genuine, not-yet-derisked unknown (see A2).
- **Timeout**: if Phase 0-1 (design through the dedicated tool's implementation) exceed 2 elapsed days without reaching Phase 2's harness-simulation work, stop and reassess scope with Robin rather than silently extending past the roadmap's 3-5 day estimate.

## Assumptions & Validation

- **A1**: the accuracy bottleneck is synthesis-explicitness (raw excerpts lack an extractable statement), not recall (finding the right excerpt) or decay (losing it too fast). This is NOT a re-stated roadmap assumption - it is a NEW finding from this session's Stage 0 work, directly evidenced.
  -> VALIDATE BY: read a sample of `runs/full-500-s-fixed/hypotheses.jsonl` filtered to `question_type=="single-session-preference"` - confirmed 27/30 hypotheses are the verbatim abstention string despite `preference-diag.json` showing 25/30 `in_topk=true` (i.e., the excerpt WAS there for most abstentions).
  -> IMPACT IF WRONG: if a deeper sample shows the abstained questions are disjoint from the surfaced-in-topk questions (i.e., abstention correlates with recall MISSES, not with recall hits), the bottleneck reverts to a recall problem and this item's whole design (synthesis at write time) would not move the number - Phase 0 must re-run this cross-check on the full 30, not just the 6-row sample used here, before committing further phases.

- **A2**: a harness-side "simulate host model calling the tool" step can be added to `bench.py`'s ingestion path in a way that (a) is architecturally honest (the LLM call plays the role of a real host model, not a new Kairn-server LLM dependency) and (b) is scoped narrowly enough not to become a general import/classification framework.
  -> VALIDATE BY: Phase 0 prototypes the simplest version - reuse the existing `llm_answer`/`openai_chat` reader-call plumbing already in `bench.py` with a NEW, narrow prompt ("does this turn pair contain a user preference; if so, state it as one explicit sentence") applied only to `single-session-preference`-category sessions during ingestion, gated behind a new `--recall-mode` or CLI flag so it never affects other categories or the production `engine` mode.
  -> IMPACT IF WRONG: if this turns out to need a broader ingestion-pipeline rewrite, that is FAILED-condition (c) above - stop and escalate rather than scope-creep.

- **A3**: the dedicated MCP tool is a thin wrapper (fixed `type="preference"`, caller supplies `content`/`tags`, tool applies the long half-life automatically) rather than a new structured schema (e.g., explicit subject/predicate fields) - matching Kairn's existing zero-schema-beyond-tags philosophy (`derive_entity_key`'s own docstring: "tags are the explicit subject signal in Kairn").
  -> VALIDATE BY: Phase 0 checks whether a thin wrapper is sufficient by reviewing 3-5 real `single-session-preference` gold answers (like the Premiere Pro example above) for whether a plain-text explicit sentence (no structured fields) would have been extractable by the reader - if plain text suffices, the thin wrapper is validated.
  -> IMPACT IF WRONG: if gold answers need structured comparison (e.g., "prefers X over Y" needing explicit contrast fields), that is FAILED-condition (a) - stop and replan.

- **A4**: `preference`'s half-life should be LONG relative to the other 5 types (the roadmap's explicit ask) - initial estimate: at or above `decision`'s 100 days (the longest current value), since preferences are typically the most durable fact type Kairn stores (rarely superseded, unlike a `solution` or `gotcha` tied to a specific bug).
  -> VALIDATE BY: this is a documented judgment call, not a tail-calibration (no access data exists yet for a type that doesn't exist) - state this explicitly in the code comment (following the existing `HALF_LIVES` dict's own precedent of documenting its calibration basis) so a future session doing a real tail-calibration pass (once production data accumulates) knows this value is a placeholder, not measured.
  -> IMPACT IF WRONG: low risk - a single dict value, trivially correctable later once real access-tail data exists; does not block this plan's other phases.

- **A5**: adding `"preference"` to `VALID_TYPES` requires touching exactly 3 places for correctness (verified this session, not assumed): `models/experience.py`'s `VALID_TYPES` set (the single source of truth, imported by both `core/experience.py::save()` and `core/intelligence.py::learn()`), plus the 2 hardcoded Field description strings in `server.py` (`kn_save` line ~563: `"solution|pattern|decision|workaround|gotcha"`, `kn_learn` line ~830: `"decision|pattern|solution|workaround|gotcha"`) which are free-text MCP-tool-schema documentation, not derived from `VALID_TYPES` programmatically - a caller (host model) reading the tool schema needs the updated string to know `preference` is valid.
  -> VALIDATE BY: already verified via direct grep/read this session (`server.py:563`, `server.py:830`, `core/intelligence.py:122-123`, `core/experience.py:162-163`).
  -> IMPACT IF WRONG: none expected - this was directly read, not inferred.

- **A6 [file-coupling with item 4]**: this item's `VALID_TYPES` touch to `src/kairn/models/experience.py` and item 4's `entity_key`/date-normalization touches to the SAME file (a separate, NOT-in-this-plan roadmap item) risk a silent bad merge if both are worked concurrently (precedent: the items-1+2 README merge gotcha).
  -> VALIDATE BY: coordinate with item 4's plan (`2026-07-07-kairn-item4-entity-extraction-diversification.md`, which names this same coupling) - land whichever touches `models/experience.py` first as an isolated prerequisite commit, or rebase deliberately before merging either.
  -> IMPACT IF WRONG: a bad silent merge would need to be caught by RC-gate diff review (Phase 4) - same mitigation as item 4's plan.

---

## Phases

### Phase 0 - Reconcile with prior K2-kill finding + design decisions (no production code)
- **Scope [revised post-Review]**: FIRST, reread `bf00180c`'s full artifacts (`read_ceiling_probe.py`, `runs/diag/read-ceiling-probe.json`) and determine whether this plan's write-time-synthesis approach is meaningfully different from the already-tested read-time reader-prompt-awareness approach in a way that could plausibly clear the entailment-mismatch bar the prior work identified - not just note the prior finding exists. If a cheap way to test this distinction exists (e.g., re-run `read_ceiling_probe.py`-style probe but with a manually-written explicit preference statement substituted for 2-3 of the 8 surfaced questions' raw excerpts, checking judge entailment), do that BEFORE any bench.py/tool code is written - this is a Low-confidence validation sprint, not a formality. THEN (only if that check is not discouraging): confirm A1 on the full 30-row sample; decide the tool name/shape (A3); decide the harness-simulation approach (A2); decide the half-life estimate (A4).
- **Deliverable**: an explicit reconciliation note (does this plan's approach differ meaningfully from the already-tested one, with a cheap empirical check if possible) + the other Phase-0 design decisions IF the reconciliation doesn't discourage proceeding.
- **Gate (binary)**: the reconciliation note is written and either (a) supports proceeding with a stated reason the write-time approach differs from what already failed, or (b) recommends stopping - both are valid Phase 0 outcomes; if (b), stop here, do not proceed to Phase 1's implementation for the benchmark-facing half of this item (Phase 1's production-tool value stands independently either way - see Success Criteria FAILED-condition and Incremental Delivery).
- **Review checkpoint**: present the reconciliation note to Robin before proceeding to Phase 1 - this is NOT a self-review-only checkpoint given the Low confidence and the standing prior decision it must engage with.

### Phase 1 - `preference` type + dedicated MCP tool (production code)
- **Scope**: `src/kairn/models/experience.py` (`VALID_TYPES`, per A5), `src/kairn/core/experience.py` (`HALF_LIVES`, per A4), `src/kairn/server.py` (new tool per A3's decided shape + the Field description strings), unit tests.
- **Deliverable**: the new tool, live-verified via a real MCP client round-trip (EPT: `claude mcp add` + fresh session calling it), plus regression tests confirming `type="preference"` no longer raises `ValueError` anywhere it's validated.
- **Gate**: new tests pass; existing `VALID_TYPES`-dependent tests (`tests/test_kn_judge.py` and others touching type validation) still pass or are updated deliberately; live MCP round-trip confirmed.
- **Review checkpoint**: yes - before proceeding to the harness-simulation work in Phase 2.

### Phase 2 - Harness-simulation + pilot gate evaluation (decision point)
- **Scope**: `~/Buisiness/Evolving/_autonomous/benchmarks/longmemeval/bench.py` (a different repo) - add the narrowly-scoped ingestion-time simulation from A2, gated behind a new flag so it never touches the `engine`/production-default recall mode; re-run a pilot-50 (or preference-specific 30-question subset) with it enabled.
- **Deliverable**: real numbers - single-session-preference accuracy with the simulation enabled vs the verified 0.10 baseline.
- **Gate (binary, per Success Criteria)**: accuracy > 0.10 -> the hypothesis is supported, proceed to Phase 3 documentation/closeout with this evidence. Accuracy <= 0.10 -> STOP, document the negative result (gotcha-style, per `65b08ce7` precedent) - the production tool from Phase 1 still ships (independent value), but no benchmark claim is made.
- **Review checkpoint**: yes - this is the go/no-go checkpoint; present real numbers before proceeding either direction.

### Phase 3 - RC-gate + merge + closeout
- **Scope**: `/code-review` (feature-dev:code-reviewer, sonnet, spec-compliance then quality) + `/simplify` (sonnet, parallel cleanup-lens agents) on the Phase 1 diff (the `kairn` repo change - the Phase 2 harness change lives in a different repo and gets its own lightweight review, not the same PR); merge via the established branch-push-PR-merge flow; `kn_learn`/`kn_judge` the outcome; closeout with a reason-coded deferred-and-untested section, explicit about pilot-vs-full-500 status and whether Phase 2's gate passed or failed.
- **Deliverable**: merged PR on `primeline-ai/kairn` main (Phase 1's tool + type); a documented result (positive or negative) for Phase 2's hypothesis test, committed to the Evolving repo's benchmark harness if it changed there.
- **Gate**: RC-gate findings resolved; CI green; merge commit verified equal to `origin/main` post-merge.
- **Review checkpoint**: final - present to Robin before considering this item done, alongside item 4's status (per Robin's own "present both, don't batch into one silent reveal" instruction).

---

## Verification

- **Automated**: `pytest tests/` (single isolated invocation, per task #2285's resolution) for new/changed unit tests + full regression on the `kairn` repo; any `bench.py` change gets a small dedicated smoke-test (not full pytest, since the benchmark harness has no pytest suite of its own).
- **Manual**: Phase 2's accuracy numbers reviewed against the verified 0.10 baseline before any go/no-go claim; Phase 1's live MCP client round-trip (real `claude mcp add` + fresh session call, same EPT pattern as item 1).
- **Ongoing Observability**: none new in production - the dedicated tool is opt-in surface with no new monitoring; if Robin's own Evolving setup starts using it (replacing the current type=decision/pattern workaround for preferences, per gotchas `74eec379`/`c58cf81d`), that adoption itself is observable via normal Kairn usage stats, not a new dashboard.

---

## CONDITIONAL: Risk

- **Risk**: A2's harness-simulation could turn out to be architecturally awkward to bolt onto `bench.py` without a larger rewrite. **Mitigation**: Phase 0 prototypes on 5-10 questions FIRST, before committing to a full pilot-50 run - this is explicitly gated as FAILED-condition (c) if it doesn't fit cleanly.
- **Risk**: the half-life estimate (A4) is an unmeasured guess, same category of risk the project already accepted for the other 5 types before their 2026-06-13 tail-calibration. **Mitigation**: documented explicitly as an estimate in code, not hidden as if it were calibrated - a future session can re-calibrate once real `preference`-type access data exists.
- **Risk [A6]**: file-coupling with item 4 in `models/experience.py`. **Mitigation**: coordinate merge order or rebase deliberately - see A6.
- **Risk**: pilot-scale sample noise (n=30 for the full preference category, likely smaller for a quick prototype) makes Phase 2's result a weak signal either direction. **Mitigation**: accepted roadmap-level risk (same as item 4's pilot-vs-proof framing, gotcha `74244096`) - the closeout must state this explicitly, not overclaim a pilot-scale win as a proven fix.

## CONDITIONAL: Rollback

Phase 1 (production tool + type) has minimal blast radius: `git revert` of the tool-addition commit removes the new tool; `VALID_TYPES`/`HALF_LIVES` additions are additive (don't change existing type behavior) so reverting them is also a clean, minutes-scale git revert with no data migration (any experiences already saved with `type="preference"` during the interim would keep that type in storage even post-revert, but `VALID_TYPES` reverting would then make them un-re-creatable, not un-readable - existing rows aren't deleted or corrupted by a revert). Phase 2's harness change lives in a different repo with no production blast radius at all (benchmark-only).

## CONDITIONAL: Execution Vehicle & Orchestration

Covered in Stage 0.5. Default Vehicle: Single (self) for Phases 0-2 (sequential design/implementation/evaluation). Phase 3 deviates:

> Phase 3 (RC-gate): Vehicle: Sub-Agent - `feature-dev:code-reviewer` (sonnet, spec-compliance then quality) + `/simplify` (parallel cleanup-lens agents, sonnet) - same established pattern as item 4's Phase 6 and items 1+2's closeout, not a new choice.

## CONDITIONAL: Dependencies

- Upstream: none new in the `kairn` repo (uses existing `experience.py`/`server.py`/`intelligence.py` modules).
- Downstream: none identified - item 5 does not gate any other roadmap item (unlike item 4, which gates items 6/9).
- External: the LongMemEval harness (`~/Buisiness/Evolving/_autonomous/benchmarks/longmemeval/`, a different repo) is both read AND written to in Phase 2 (the harness-simulation addition) - this is a real cross-repo dependency; the smoke-check recommended in item 4's plan (confirm the harness still runs before relying on it) applies here too, and should be done once, shared between both items' Phase-0 work if both are active concurrently.
- File-coupling with item 4: see A6.

## CONDITIONAL: Reference Library

| Source | Version/Date | What it informed | Link |
|---|---|---|---|
| `src/kairn/models/experience.py`, `core/experience.py`, `core/intelligence.py`, `server.py` (this repo, HEAD `3d0c0c9c`) | 2026-07-07 | VALID_TYPES single-source-of-truth path, HALF_LIVES precedent, exact touch points (A5) | local files |
| `runs/full-500-s-fixed/summary.json` | full-500 LongMemEval-S, engine mode | verified 10% preference baseline | local artifact, Evolving repo |
| `runs/diag/preference-diag.json` | diag harness output | 83% (25/30) surfaced-in-topk rate | local artifact, Evolving repo |
| `runs/full-500-s-fixed/hypotheses.jsonl` (filtered to single-session-preference) | full-500 run | 27/30 abstention rate on the reader's actual answers - the central Stage 0 finding | local artifact, Evolving repo |
| `bench.py` `READER_SYS`/`llm_answer` (Evolving repo) | 2026-07-07 | the exact reader prompt causing the conservative-abstention behavior | local file |
| Kairn pattern `65d3c739`/`d4067bce` | 2026-07-07 | native-provider preference-capture-by-construction framing (ChatGPT Memory) | Kairn node |
| Kairn pattern `36f1309d` | 2026-07-07 | mem0's 96.7% + full-context-baseline (20-30%) beating Kairn's 10% | Kairn node |
| Kairn gotchas `74eec379`/`c58cf81d`/`c5e9576c8`/`ef485245` | 2026-06-11 to 2026-07-02 | the live, already-hit type=preference rejection bug this item fixes for real usage | Kairn nodes |
| Kairn gotcha `74244096` | 2026-07-07 | pilot-vs-proof framing discipline for closeout language | Kairn node |
| Kairn decision `758ece4c` | 2026-07-07 | the roadmap item text itself | Kairn node |

## CONDITIONAL: Resume Protocol

Total estimated effort (3-5 days per the roadmap) exceeds 10h, likely spanning multiple sessions. Each phase's gate result is recorded directly in this plan file as it completes - a resuming session should read top to bottom, find the last recorded gate result, and resume from the next unstarted phase. Re-run the external-harness smoke-check (see Dependencies) if resuming after a gap, since the harness is a separate, independently-changing repo.

## CONDITIONAL: Incremental Delivery

Two natural stopping points, cross-referenced rather than duplicated: (1) Phase 1 alone (the production tool + type) ships independent value regardless of Phase 2's benchmark outcome - if time runs short, stopping after Phase 1 + its own RC-gate is a coherent, honestly-scoped partial delivery, not a half-finished feature; (2) Phase 2's gate failure (FAILED-condition (b)) is itself a valid stopping point with Phase 1 still merged.

## CONDITIONAL: Completion Gate

Touches `src/kairn/models/experience.py`, `src/kairn/core/experience.py`, `src/kairn/server.py`, test files (this repo), and `bench.py` (a different repo, Phase 2 only). Registration/consistency checklist for Phase 3: confirm the new tool is actually documented in `README.md`'s tool list/count (this DOES add a genuinely new MCP tool, unlike item 4's Phase 5 which only extends existing tool behavior - so the tool-count bump is real and must be reflected, unlike the note in item 4's plan). No `_stats.json`/knowledge-graph registration applies (this is the `kairn` repo, not `Evolving`'s component-registration system).

## CONDITIONAL: Post-Completion

Covered by Phase 3: RC-gate resolution, merge verification, closeout messaging to Robin. No new production monitoring surface beyond what Phase 1's live MCP round-trip already confirms.

## CONDITIONAL: Learning & Knowledge Capture

Handled via Phase 3's `kn_learn`/`kn_judge` step (same convention as item 4 and items 1+2). Given this item's central finding (A1 - the abstention/synthesis mechanism) is genuinely novel and could inform OTHER future work on Kairn's recall-quality gaps (not just preferences), it should get its own dedicated Kairn node distinct from the item-level closeout, tagged so a future session investigating temporal-reasoning (42.9%) or multi-session (41.4%) accuracy gaps can find it - the same reader-abstention mechanism plausibly affects those categories too, though that is explicitly NOT investigated in this plan (out of scope - flagging the pattern, not chasing it here).

---

## Phase 0 Gate Result: outcome (b) for Phase 2 - KILLED PERMANENTLY (2026-07-07, executing session)

**Reconciliation probe** (`probe_writetime_synthesis.py`, artifact `runs/diag/writetime-synthesis-probe.json`, Kairn kill record `cfea54d3`): 3 surfaced preference questions were given a MANUALLY-WRITTEN ideal explicit preference statement as [Memory 1] (paraphrased from gold - the upper bound of any host-model write-time synthesis) alongside the normal top-8 excerpts, under the UNCHANGED harness reader+judge. **Result: 0/3 - the reader still abstains verbatim on every one.**

**Sharpened root cause (beyond the June entailment framing):** READER_SYS's conservatism makes recommendation-shaped questions unanswerable regardless of stored-content quality - the reader must GENERATE recommendations, which are never IN memory, so it abstains before entailment is even tested. The category is double-blocked: (1) reader abstention (this probe, 0/3 with perfect statements), (2) entailment mismatch vs meta-description golds for rare non-abstaining answers (June, 1/8). No storage-side change (capture, synthesis, decay, recall, type) can move the 10%.

**Consequences:** Phase 2 killed permanently per decision `ccc704f0`'s hard exit (edges recorded: cfea54d3 -> bf00180c/3f098f42/758ece4c/ccc704f0). Phase 1 (preference type + MCP tool) proceeds - its value is real-usage capture (the live type=preference rejection bug), never the benchmark number. The only real levers on the 10% remain (a) a READER_SYS harness-calibration discussion with Robin (all-categories impact, NOT-scope here) or (b) honest annotation in the item-3 scorecard, which this analysis makes stronger.

**A1 full-30 confirmation: subsumed** - the probe result makes the finer recall-vs-abstention split moot for Phase-2 purposes (abstention dominates even with perfect recall content); recorded here so the VALIDATE-BY isn't silently skipped.

---

## Interview Log

**Date**: 2026-07-07 | **Mode**: Self | **Domain**: Software Development
**Questions asked**: 6 | **Anti-patterns found**: 1

| Tier | Question | Answer/Decision | Anti-Pattern |
|------|----------|----------------|---------------|
| 1 | Is A1 (the central abstention finding) adequately validated, or just eyeballed from 6 rows? | Eyeballed from 6 rows initially - VALIDATE BY was tightened to require Phase 0 confirm it on the FULL 30-row sample before committing further phases, not just the illustrative sample used to discover the pattern. | #1 Unvalidated Assumptions (caught and fixed pre-emptively, not deferred) |
| 1 | Does changing the benchmark harness's reader prompt directly (the seemingly obvious fix given the diagnosis) get proposed as an alternative, and if not, why not? | Deliberately excluded from NOT-scope: `READER_SYS` is the harness's own calibration choice, shared across ALL question categories, not something this plan should unilaterally change to game one category's number - noted explicitly rather than silently omitted. | #10 First Idea = Final (the "obvious" fix was considered and deliberately rejected, not just unconsidered) |
| 2 | Software domain: is the cross-repo nature of Phase 2 (touching the Evolving repo's benchmark harness, not just the kairn repo) handled explicitly? | Yes - Dependencies section names it explicitly, Phase 3 notes the harness change gets separate review from the kairn repo's PR. | - |
| 2 | Is there a Reference Library with sourced numbers, including the NEW finding (not just roadmap-inherited numbers)? | Yes - 9 entries, including the hypotheses.jsonl finding that is new to this session, not inherited from the roadmap text. | - |
| 3 | Cold Start Test: what would a zero-context reader misunderstand first? | That "MCP judo" means Kairn's server runs an LLM - clarified inline in End State that the server stays zero-LLM; the harness simulation plays the role of a host model, architecturally equivalent to real Claude Code usage. | - |
| 3 | Devil's Advocate: is this the right problem to solve, or should the reader-prompt fix be pursued instead as a cheaper alternative? | Considered and explicitly deferred to Robin as a separate conversation (not this plan's call) since it affects ALL categories' scoring, not just preferences - noted in Risk/NOT-scope rather than silently assumed off the table. | - |

**Plan changes made**: tightened A1's VALIDATE BY to require full-30-row confirmation (not the 6-row sample) before Phase 1 proceeds; added the explicit NOT-scope note about not changing `READER_SYS`.
**Grade before**: B+ (strong Stage 0 evidence, but A1 was under-validated and the "why not just fix the reader prompt" question was implicit rather than explicit)
**Grade after**: A- (pending Hardening + Review)

## Stage 1.5: Hardening Log

**Ran**: 2026-07-07 | **Mode**: Simple | **Perspectives**: 6/6

| Perspective | Finding | Action Taken |
|---|---|---|
| Outside Observer | Strong End State/Context&Why given the plan already included them from the start (lesson carried over from item 4's review) - can summarize in <15 words: "Add a preference type + tool, test whether explicit write-time synthesis beats the 10% baseline." | No action needed. |
| Pessimistic Risk Assessor | Checked for a single point of failure: A2 (harness-simulation feasibility) is the one genuinely load-bearing unknown - if it fails, does the WHOLE item fail? No - Phase 1 (the production tool) ships independently per Incremental Delivery, so this is NOT a zombie-project risk. | No action needed - already correctly decoupled via Incremental Delivery. |
| Pedantic Lawyer | Phase 2's gate ("accuracy > 0.10") is cleanly binary; scanned all other gates for judgment words - none found. | No action needed. |
| Skeptical Implementer | First blocker at 9am tomorrow: does Phase 0's full-30-row A1 re-validation require re-running the benchmark, or just re-reading existing `hypotheses.jsonl`? Existing artifact already contains all 30 rows - no re-run needed, purely a read/filter task. Confirmed cheap. | No action needed - confirmed low-cost via the Reference Library entry already pointing at the right file. |
| The Manager | 4 phases (not >5) - Incremental Delivery still included since it's a >10h multi-session plan regardless of phase count (the >5-phases trigger doesn't apply, but the section still adds real value here given the two genuine stopping points) - kept it. | No action - correctly included on its own merits, not mechanically triggered. |
| Devil's Advocate | Re-confirmed the AHA/alternative check (0.9) was genuine: the "just fix the reader prompt" alternative was seriously considered and deliberately deferred (Risk + NOT-scope), not just listed for compliance. | No action needed. |

**Discovery Consolidation**: All Stage 0 findings addressed - the 83%-recall-vs-10%-accuracy gap, the 90% abstention rate, the exact `READER_SYS` wording, the 3-touch-point VALID_TYPES change, and the file-coupling with item 4 are each referenced in a specific phase, assumption, or risk item. No orphaned discoveries.

## Stage 3: Review Report

**Plan:** `2026-07-07-kairn-item5-preference-mcp-judo.md` | **Domain:** Software Development | **Grade:** A | **Confidence:** Medium
**Recommendation:** Proceed

### CORE Sections

| Section | Status | Format | Issue |
|---|---|---|---|
| Context & Why | ok | Yes | 3 sentences, explains why (10% is the worst number, mechanism now understood) |
| Success Criteria | ok | Yes | FAILED conditions (a/b/c) + 2-day timeout present |
| Assumptions | ok | Yes | 6 assumptions (A1-A6), triple format, A1 is a genuinely new finding not a roadmap restatement |
| Phases | ok | Yes | Binary gates; review checkpoints after Phase 1, at Phase 2 (go/no-go), at Phase 3 (final) |
| Verification | ok | Yes | Automated + Manual + Ongoing Observability all populated |

### CONDITIONAL Sections (Software Development)

All present: Risk, Rollback, Execution Vehicle & Orchestration, Dependencies, Reference Library, Resume Protocol, Incremental Delivery, Completion Gate, Post-Completion, Learning & Knowledge Capture - all included from the initial draft (informed by item 4's review lessons this same session), not bolted on afterward.

### Stage 0 Evidence

- Existing work checked: yes, extensively (VALID_TYPES flow, kn_save/kn_learn schema strings, intelligence.py's learn() path, bench.py's ingestion + reader mechanism)
- Feasibility questioned: yes (A2's harness-simulation risk explicitly flagged as the main uncertainty)
- Alternatives considered: yes (0.9 AHA - "fix the reader prompt directly" was a real, seriously-considered, deliberately-deferred alternative, not filler)
- Constraints discovered: yes (zero-LLM-server architecture re-affirmed and reconciled with the harness-simulation approach)
- Official docs consulted: N/A (internal code + benchmark artifacts are the "official docs" here, no external library integration)
- Stage 0 status: Done - and this is the plan's strongest asset: A1 (the abstention mechanism) is a genuinely new, load-bearing discovery that reframes the entire item, found by actually reading `hypotheses.jsonl`, not inferring from the roadmap text.

### Execution Vehicle & Routing (Check 11)

- Default Vehicle present: yes (Single (self), Phase 3 deviation to Sub-Agent, model tier stated: sonnet for both code-reviewer and simplify agents)
- Per-phase vehicles justified: ok
- Multi-agent vehicles fully parameterized: ok
- Model routing sane: ok - no mis-routing
- Silent selection: yes

### Anti-Patterns Found

None triggered a hard flag. The "fix the reader prompt instead" alternative was correctly handled as a disclosed, deliberate NOT-scope decision (not #10 First Idea = Final, since the first idea was explicitly challenged and an alternative was seriously weighed).

### Red Flags

None.

### Top 3 Improvements (already applied)

1. A1's validation was tightened from "eyeballed 6 rows" to "confirm on the full 30-row sample" during Interview - the single most important assumption in this plan now has a rigorous validation bar, not an anecdotal one.
2. The "why not just fix the reader prompt" question was made explicit rather than left as an implicit unstated design choice.
3. The A6 file-coupling risk with item 4 was cross-referenced bidirectionally (both plans now name each other), reducing the odds either session forgets about it.

### Bottom Line

Grade A: all 5 CORE sections complete, all domain-required CONDITIONAL sections present from the first draft, Stage 0 evidence is unusually strong (a genuine, well-evidenced mechanism discovery that meaningfully reframes the roadmap's original framing, not just a restatement of it), zero anti-pattern flags. The plan's biggest asset is honesty about what's actually driving the 10% number - recommend proceeding to Phase 0, and recommend Robin read the DSV-Suspend/A1 section specifically since it changes the story the roadmap told.
