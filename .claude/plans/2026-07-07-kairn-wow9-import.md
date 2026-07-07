# Kairn WOW-9: `kairn import` (Claude Code history + git log)

**Status**: Ratified draft - Stages 0 through 4 complete (Discovery, Interview, Hardening, independent Review, Finalization). Awaiting Robin's go/no-go gate before Stage 5 execution begins in a follow-up session.
**Quality Grade**: A (self-graded A after Stage 1.5; independent Stage 3 planner-agent review found 2 Red Flags + 3 gate/scope mismatches, all fixed; grade re-confirmed A post-fix, not re-verified by a third pass)
**Confidence**: Medium (2 unknowns: transcript rule-based extraction precision at scale; final redaction ruleset scope)
**Domain**: Software Development (library/CLI feature, Python)
**Default Vehicle**: Single (self, Sonnet, effort high) - all 5 phases are sequential, single-codebase, no independent-stream fan-out; RC-gate review after every phase per project rule, not a separate vehicle
**Date**: 2026-07-07
**Repo**: `/Users/neoforce/Business/engram` (main @ 818df24 at plan time; 9f48719 after v0.2.1 rendering fix, confirmed via `git log` during Stage 0)

---

## DSV Pre-Check

1. **Decompose** - key claims in this request:
   - (a) A deterministic, zero-LLM, rule-based pipeline can extract meaningful "experiences" from Claude Code JSONL transcripts and git log history at $0 marginal cost - the structural differentiator vs. competitors who require an LLM call per ingested episode.
   - (b) This import can be done safely (privacy/redaction) and idempotently (no duplicate bloat, no runaway store growth) with acceptable precision.
   - (c) The right vehicle for this session is a full UPF plan now; code lands in later, separate build sessions.

2. **Suspend** - alternative interpretation considered: the handoff bundles "claude-code transcripts" and "git log" as one homogeneous "import" feature, but they have very different risk profiles. Git log is structured, low-risk (no secrets beyond what's already in commit history, which the user already committed), and - per the Stage 0 sampling below - carries strong signal density in this user's repos. Claude Code transcripts are unstructured, carry real exposure (tool outputs, pasted credentials, client data), and their rule-based extraction precision is unvalidated. Treating them as one feature risks gating the safe, high-confidence half (git) behind the unresolved, high-risk half (transcripts). **Resolution applied in this plan**: sequence git import first as a standalone-shippable phase, transcript import second and explicitly privacy-gated - see Phases 1-3. This is a scope/sequencing decision, not a re-ask to Robin (Autonomous mode selected); flagged here for transparency.

3. **Validate** - least-sure claim: (a)'s precision assumption for transcripts. Addressed empirically in Stage 0 Discovery below (schema probe on a real transcript file) rather than left as pure speculation. Result: partially validates, but surfaces a new distinction (see Discovery finding 4) that reshapes Phase 3's design.

---

## Context & Why

Kairn's structural edge over LLM-per-episode competitors is that its deterministic pipeline can backfill months of a user's real history instantly for $0, while competitors must call an LLM per ingested episode. `kairn import claude-code` and `kairn import git` are the concrete product for that edge: turning a new (or existing) Kairn user's pre-existing Claude Code transcripts and git commit history into real, immediately-recallable experiences with zero setup cost. Robin selected this as the next big Kairn build (Kairn node `758ece4c`, decision `19831c9f`) after the honesty-scorecard ship, explicitly commissioning this session to produce a ratified UPF plan before any build session starts.

## End State

After this plan is fully executed (across however many build sessions Phases 1-5 take), a Kairn user can run `kairn import git <repo-path>...` against any local git repository and `kairn import claude-code [--root PATH] [--since DATE]` against their `~/.claude/projects/` (and equivalent secondary-account) transcript trees, and see real historical decisions, solutions, and gotchas appear in their Kairn store under dedicated `imported-git` / `imported-claude-code` namespaces - each tagged with a stable source reference for idempotent re-runs, each produced with zero network calls and zero LLM tokens spent, and each previewable via `--dry-run` before anything is written. A user who has never touched Kairn before can adopt it and immediately recall months of prior decisions on day one. The whole import path is documented in the README, covered by tests, and has been run once for real (sandboxed) against Robin's own multi-repo, multi-account history as its own proof.

**[Clarified, Interview Tier 3 Q9 - Cold Start Test]**: two likely misreadings to head off in the README (Phase 4): (1) "zero-LLM" describes the *import path* only - no LLM call happens while extracting/writing experiences - it does not mean the resulting data is never touched by an LLM (a Claude Code agent still reads it back at recall time); (2) `kairn import` is not a one-time migration tool - it's designed to be re-run as history accumulates, safely, because of the idempotency guarantee in the Success Criteria below.

## Success Criteria

- `kairn import git <repo>...` ships as a working CLI command supporting `--since DATE` and `--dry-run`; running it twice on an unchanged repo produces **zero** new experiences on the second run (idempotency, measured via before/after `count_nodes`/experience count in the target namespace).
- `kairn import claude-code [--root PATH]` ships as a working CLI command with the same `--since`/`--dry-run` flags plus a mandatory redaction pass; `--dry-run` output is reviewable before any write occurs and is the default confirmation gate (no silent first-run writes).
- A manual precision audit (Phase 3, N>=50 sampled extracted experiences from real transcripts) shows a junk/false-positive rate low enough to be worth shipping fine-grained (per-signal) extraction - concrete bar: **<20% clearly-noise entries** in the audited sample. Above that bar, ship the coarse fallback (Success Criterion below) instead, not silently ship the noisy version.
- Coarse fallback exists and is documented: if fine-grained extraction doesn't clear the precision bar, `kairn import claude-code` still ships in "one experience per session, session-summary only" mode - lower detail, same privacy/redaction guarantees, still zero-LLM.
  - **[Low confidence, Interview Tier 3 Q10]**: the 20% junk-rate bar is a reasonable starting heuristic, not an empirically-derived number - there is no prior precision benchmark for this exact extraction task to calibrate against. Treat it as Robin-adjustable at the actual Phase 3 audit review, not a fixed scientific threshold.
- Redaction layer is validated against a known secret-pattern test corpus (common API-key prefixes, `Bearer `/`Authorization:` headers, `password=`/`token=` assignments, AWS/OpenAI/Anthropic key shapes) with **zero missed matches** on that corpus before the first real (non-dry-run) import is run against Robin's actual history.
- Full real-data EPT (Empirical Proof Test - the standing 3-leg trigger/effect/consumer proof requirement, Phase 5): a sandboxed store (never the production `~/.kairn` workspace) receives a real import of Robin's actual `~/.claude/projects/` + `~/.claude-secondary/projects/` trees and >=3 real git repos, completes without unhandled exceptions, and at least 10 real historical recall queries against that sandbox surface an imported experience that predates this feature.
- Zero new runtime dependencies beyond Python stdlib (`subprocess` for git, `json`/`re` for parsing) - no new third-party package added to `pyproject.toml` for this feature, preserving the "$0, zero-LLM, minimal-dependency" story.

**NOT-scope** (deliberately excluded):
- Zero LLM calls anywhere in the import path, for any reason - this is the entire product promise. Any design that needs an LLM call is out of scope for this feature, full stop.
- No new MCP tool - CLI-only surface unless Stage 3 `/plan-review` finds a concrete reason to add one.
- No changes to recall, relevance scoring, or decay logic - import only writes `Experience`/`Node` rows through the existing storage layer; it does not touch `core/experience.py`'s `relevance()`/decay math.
- No import of other people's data from shared/multi-author repos beyond what the user already has committed to their own local clone - git import operates only on repos the user explicitly names as CLI arguments, never auto-discovered.
- WOW 8 (`uvx kairn try`), WOW 10 (`kn_why` receipts), roadmap items 6 (temporal query path, permanently parked) and 7 (salience replay eval) - untouched.
- Publishing releases (Robin-run per standing process; task #2297 is already closed/completed as of this session's Kairn preflight, contrary to the handoff's "may still be open" caveat).
- Moving the LongMemEval accuracy harness into the OSS repo - stays in the private Evolving research tree.
- Any change to the recall path - if Phase 3/5 uncovers a reason one is needed, the README radical-honesty scorecard commitment (node `26d17a00`) fires and must be flagged to Robin explicitly, not silently bundled into this feature.

**FAILED conditions** (kill criteria + timeout):
- If Phase 2's privacy/redaction design cannot converge on an approach that passes its own test corpus (see Success Criteria) within its allotted phase - kill transcript import (`kairn import claude-code`) for this cycle. Ship `kairn import git` alone. This is a permanent-for-this-cycle kill, not a silent defer - document it in the plan and in a Kairn decision node.
- If the Phase 3 precision audit (N>=50) shows >=20% junk-rate AND the coarse session-summary fallback is also judged not worth shipping (e.g., because even session-summaries leak un-redactable content) - kill `kairn import claude-code` entirely for this cycle; ship git-only.
- Timeout: if Phase 1 (`kairn import git`, the low-risk MVP) has not shipped to `main` within 3 build sessions of starting Stage 5 execution, stop and bring the scope back to Robin rather than continuing to iterate unbounded - this is explicitly a 1-2 week item per the handoff, not open-ended.
- **[Clarified, Interview Tier 1 Q1]** Overall plan timeout: if Phases 1-5 collectively have not shipped within 2 calendar weeks of Stage 5 execution starting, stop at whatever phase is in flight, ship what has cleared its gate, and bring the remainder back to Robin as a scope/timeline decision rather than silently running long - mirrors the handoff's explicit "1-2 week item" framing.

---

## Assumptions & Validation

- Rule-based markers (conventional-commit-style prefixes, decision/gotcha/fix keyword regexes, markdown-header conventions) can extract high-precision signal from Claude Code transcripts without an LLM.
  -> VALIDATE BY: Phase 3 spike - run the extractor against a labeled sample of 30-50 real transcript files spanning both Kairn-integrated sessions (which already call `mcp__kairn__kn_learn` - see Discovery finding 4, these must be excluded from extraction to avoid pure duplication) and pre-Kairn sessions; manually audit precision.
  -> IMPACT IF WRONG: ship the coarse session-summary fallback (one low-detail experience per session) instead of fine-grained extraction, or kill transcript import per the FAILED conditions above.

- A dedicated import namespace (`imported-git`, `imported-claude-code`) plus a content-hash `source_ref` stored in `Experience.properties` is sufficient for idempotency and rollback, with no schema migration needed.
  -> VALIDATE BY: Stage 0 confirmed `namespace` is already a first-class indexed column on both `nodes` and `experiences` tables (`storage/sqlite_store.py`); Phase 1 must additionally confirm (or add) a namespace-scoped bulk-delete/list path in the storage layer, since one wasn't found by name during Stage 0 grep (only `namespace=` filter params on read paths were confirmed, not a dedicated delete-by-namespace method).
  -> IMPACT IF WRONG: needs an actual schema migration (new indexed column, e.g. `source_ref`) instead of reusing the existing free-form `properties` dict - larger scope, replan Phase 1.

- Regex/keyword redaction (API-key shape patterns, `Bearer `/auth-header patterns, `password=`/`token=`/`secret=` assignment patterns, common vendor key prefixes) plus a mandatory `--dry-run` preview is an acceptable privacy bar for a $0 CLI tool aimed at a technical, security-aware user base.
  -> VALIDATE BY: cross-check the redaction ruleset against a well-known public secret-pattern rule set (e.g. gitleaks' default rules, conceptually - used as a reference checklist, not a new dependency) plus a manual review of real `--dry-run` output on Robin's own history before the first non-dry-run import.
  -> IMPACT IF WRONG: privacy design needs a stronger approach (e.g. an opt-in allowlist-only mode, or a stricter default that redacts entire tool-result blocks rather than pattern-matching within them) - could force a scope renegotiation with Robin if it meaningfully increases complexity beyond the "$0, zero-dependency" goal.

- Git commit messages in Robin's own repos carry decision-shaped signal often enough to be worth importing as typed experiences (not just "wip"/"fix typo" noise).
  -> VALIDATE BY: Stage 0 already sampled `git log --oneline -20` on `engram` and `Evolving` - both show strong conventional-commit signal (`fix:`, `feat:`, `docs:`, `chore:`, `data:` prefixes with substantive bodies). **Validated during Stage 0**, high confidence.
  -> IMPACT IF WRONG (residual risk on repos not sampled): git import scope narrows to metadata-only timeline entries (commit sha/date/message-as-content, no type inference) rather than typed `decision`/`solution` experiences - still useful, smaller Phase 1 scope.

- Claude Code transcript volume (2,463 files / 2.1GB measured on the primary account alone during Stage 0, plus a secondary account) is processable by a pure-Python streaming parser in a reasonable wall-clock budget without needing batching infrastructure.
  -> VALIDATE BY: Phase 4 - time a full real run across the measured corpus; if it exceeds a few minutes, add a simple `--since` date-window default (not a queue/batch system - stays proportionate to a $0 CLI tool).
  -> IMPACT IF WRONG: add `--since` as a required-by-default flag (e.g. defaults to last 90 days) rather than "all history" being the default invocation, to keep first-run UX fast.
  -> [Clarified, Interview Tier 1 Q3]: "a reasonable wall-clock budget" tightened to a concrete number - **10 minutes** for a full run against the measured corpus (2,463 files / 2.1GB). Above that, `--since` windowing becomes the default-on behavior, not just an available flag.
  -> **[Low confidence, Stage 3 fix - independent planner-agent review]**: like the 20% junk-rate bar, this 10-minute number is a heuristic ("a $0 CLI tool should feel roughly instant"), not derived from a formal benchmark or user-research target. Treat it with the same Robin-adjustable status as the 20% bar, not as a harder-sourced number just because it looks more precise.

---

## Phases

### Phase 1: `kairn import git` - deterministic commit-metadata importer (low-risk MVP)

- **Scope**: New `src/kairn/importers/` package; `git.py` importer using `subprocess` (`git log --format=...`) against user-named local repo paths; maps conventional-commit prefixes (`fix:`->gotcha/solution, `feat:`->pattern/decision, `docs:`/`chore:`->lower-priority context, merge commits filtered/deprioritized) to `Experience` type; stable `id`/`properties.source_ref` derived from commit SHA for idempotency; new CLI command `kairn import git <repo>... [--since DATE] [--dry-run]` in `cli.py`; confirm/add namespace-scoped delete path in `storage/sqlite_store.py` per the Assumptions section above.
- **Deliverable**: working CLI command, unit tests (idempotency, conventional-commit type mapping, `--since` filtering, `--dry-run` produces no writes), README section.
- **Gate** (binary): `kairn import git <2+ real repos> --dry-run` then for-real produces experiences in the `imported-git` namespace; running it a second time produces zero new rows; the namespace-scoped delete path is empirically tested (write N rows, delete by namespace, confirm zero remain) - not just present in scope but proven at the gate; `pytest` suite green; RC-gate (`/code-review`) passed.
- **Review Checkpoint**: yes (start of the every-2-phases cadence).

### Phase 2: Privacy & Redaction Design (the load-bearing gate - spec + preview mechanism, no transcript parsing yet)

- **Scope**: Design and implement the redaction ruleset as a standalone, testable module (`importers/redact.py`) independent of the transcript parser: regex rules for API-key shapes, auth headers, `password=`/`token=`/`secret=` assignments, common vendor key prefixes; a `--dry-run` preview renderer that shows exactly what would be stored (post-redaction) before any write; an explicit confirmation gate for the first non-dry-run invocation. Built and unit-tested against a test corpus (not real transcripts yet - this phase is deliberately parser-independent so it can be validated in isolation). **[Stage 3 fix, independent planner-agent review]**: the test corpus must NOT be purely self-authored (writing the redaction rules and their own test cases together makes "zero missed matches" a low bar by construction) - it must literally incorporate a sample of gitleaks' public default-rule test fixtures as data (not a new dependency, just reference fixture patterns), alongside any synthetic cases, so the zero-miss bar is checked against an externally-sourced pattern set.
- **Deliverable**: `redact.py` module + test corpus + passing tests; a written redaction policy (what gets redacted, what gets excluded entirely vs. masked) documented in the plan/README.
- **Gate** (binary): redaction module scores zero missed matches on its test corpus (Success Criteria bar); `pytest` green; RC-gate passed.
- **Review Checkpoint**: yes.

### Phase 3: `kairn import claude-code` - transcript extraction engine (privacy-gated on Phase 2)

- **Scope**: Streaming JSONL parser (`importers/claude_code.py`) over `~/.claude/projects/**/*.jsonl` (and a configurable `--root` for secondary accounts / non-default install locations); rule-based signal extraction per the Assumptions section (regex/keyword markers on user+assistant text and `thinking` blocks); **must explicitly skip** any `tool_use` block that is itself an `mcp__kairn__kn_learn`/`kn_save`/`kn_add` call, to avoid re-importing what's already in the store (Stage 0 finding: this ecosystem's own sessions already call these tools directly); routes all extracted text through Phase 2's redaction module before storage; stable `source_ref` = hash of (file path, line/uuid range); new CLI command `kairn import claude-code [--root PATH]... [--since DATE] [--dry-run]`.
- **Deliverable**: working CLI command; N>=50-sample precision audit executed and documented (see Success Criteria / Assumptions); coarse session-summary fallback implemented if the audit misses the precision bar.
- **Gate** (binary): precision audit result documented with a pass/fail verdict against the 20% junk-rate bar; whichever mode ships (fine-grained or coarse) passes its own tests; the schema-shape smoke test (from the Discovered Risk mitigation below) is present and passing against a fresh real transcript sample - not just described in scope but demonstrated at the gate; `pytest` green; RC-gate passed.
- **Review Checkpoint**: yes.
- **[Clarified, Interview Tier 1 Q2]**: run the N>=50 precision spike against a *small* early sample FIRST (before building out the full parser/extraction engine), not after. Assumption 1 (extraction precision) is the single highest-leverage unknown in this plan - if it fails, it should fail cheap and early, not after a full parser is built.

**[Stage 5 - Phase 1 GATE: PASS, 2026-07-08]**
- Built via TDD: `src/kairn/importers/git.py` (`classify_commit_type`, `import_git_repo`, `_iter_commits`), `delete_experiences_by_namespace` added to `storage/base.py`+`sqlite_store.py`, `kairn import git` CLI command wired in `cli.py`. 21 new tests (storage + importer + CLI), all green; full suite 519 -> 526 passed, 0 regressions; `ruff check`/`ruff format` clean.
- EPT (real, sandboxed - never `~/.kairn`): `kairn import git` against 2 real repos (`engram`, `Evolving`) wrote 1783 experiences into `imported-git`; idempotent re-run confirmed `imported=0`/`skipped_duplicate` exactly matching first-run counts; namespace-scoped delete removed exactly 1783 rows, confirmed via `kairn status` (`experiences: 0`).
- Independent RC-gate (`feature-dev:code-reviewer`, sonnet/high, dispatched as a background agent) found 5 real issues, all fixed same-session: (1) unhandled `CalledProcessError` on empty/non-git repos now raises a clean `GitLogError`; (2) commit timestamps now UTC-normalized (`_to_utc_iso`) instead of preserving the original commit offset, which broke `query_experiences_since`'s string-comparison invariant; (3) experience id widened 8->16 hex chars + `source_ref` verified before treating an `IntegrityError` as a duplicate (a real collision now surfaces as a new `collisions` counter, never silently drops data); (4) merge detection switched from a subject-line regex to git's structural parent count (`%P`), fixing both a false-positive ("Merge duplicate customer records" was being dropped) and a false-negative ("Merged pull request..." phrasing); (5) explicit `encoding="utf-8", errors="replace"` added to the `git log` subprocess call. 8 new tests added covering each finding; re-verified against real `engram` repo history post-fix (0 non-UTC timestamps, 0 collisions, correct classification).
- Commits: `ac113d8` (feature), `2f1fc8b` (docs), `3e5f95c` (RC-gate fixes) on `feat/wow9-phase1-git-import`.
- Rollback path (namespace-scoped delete) empirically tested per the Stage 3 fix requirement - see EPT above.
- **[Clarified, Interview Tier 3 Q8]**: `kairn import claude-code` with no `--root` given defaults to auto-scanning BOTH `~/.claude/projects/` and `~/.claude-secondary/projects/` if they exist (Stage 0 found a real dual-account asymmetry: primary has 195+ project dirs, secondary has 5) - `--root` becomes an override/addition, not a mandatory flag for multi-account setups.
- **[Discovered Risk, Interview Tier 3 Q7 - Devil's Advocate]**: the Claude Code JSONL transcript schema is internal/undocumented and not a versioned public API - it could change between CC releases and silently break the parser (crash, or worse, silently miss data without erroring). Mitigation folded into this phase's scope: parse defensively (skip/log unknown record types and unexpected shapes rather than crash), and add a schema-shape smoke test that fails loudly if core fields (`type`, `message.role`, `timestamp`) go missing from a fresh real sample - this is the canary for schema drift, not a one-time check.

### Phase 4: Volume control, idempotency hardening, CLI/docs polish

- **Scope**: `--since` default-window behavior if Phase 4's timing spike (Assumptions) shows it's needed; consolidated `kairn import` CLI help text (minimum bar: `--help` documents every flag from both subcommands with a one-line description each, verified by reading the rendered `--help` output, not just that a docstring exists); README "Importing your history" section covering both subcommands, the privacy model, and the coarse-vs-fine-grained transcript mode; full idempotency re-check across both importers together (run both twice, confirm zero duplicate growth).
- **Deliverable**: docs merged; timing numbers for the full measured corpus documented.
- **Gate** (binary): full double-run produces zero duplicates across both importers combined; the Phase 4 timing spike's measured wall-clock time is checked against the 10-minute threshold (Assumptions) - if it exceeds 10 minutes, `--since` default-on windowing is implemented and demonstrated; if it does not, an explicit note is recorded that no change was needed (not silently skipped either way); README section renders with no broken internal links and no unclosed code fences (verified by a markdown lint pass or manual render); `pytest` green; RC-gate passed.
- **Review Checkpoint**: yes.

### Phase 5: EPT - real sandboxed import + recall spot-checks

- **Scope**: Stand up a throwaway sandbox Kairn workspace (never `~/.kairn` production); run `kairn import git` against >=3 of Robin's real repos and `kairn import claude-code` against the real `~/.claude/projects/` + `~/.claude-secondary/projects/` trees; run >=10 real historical recall queries against the sandbox and confirm imported experiences surface; write the Verify Report companion file per UPF Stage 4 guidance.
- **Deliverable**: `2026-07-07-kairn-wow9-import-verify.md` sibling file with pasted evidence for all 3 EPT legs (trigger/effect/consumer) per phase gate above.
- **Gate** (binary): all Success Criteria items checked off with pasted evidence; deferred-and-untested section written for anything not provable now. **[Stage 3 fix, independent planner-agent review]**: the >=10-recall-surfaces-an-imported-experience criterion has a known confound - Kairn's reader-ceiling limitation on temporal/multi-session recall (Kairn `7018bbd5`, referenced in Reference Library) is a pre-existing, out-of-scope limitation independent of this feature. If fewer than 10 queries surface an imported experience, first diagnose whether the shortfall traces to that known reader-ceiling limitation (unchanged by this feature, does NOT block shipping) versus an actual import/write defect (in-scope, DOES block shipping) before treating the gate as failed.
- **Review Checkpoint**: final (plan completion).

---

## Verification

- **Automated**: `pytest` suite (existing + new importer/redaction tests) green on every phase; CI (`.github/workflows/ci.yml`, live since PR #15) runs the same suite on Python 3.11/3.12/3.13 on every PR.
- **Manual**: RC-gate (`/code-review`, `feature-dev:code-reviewer` sonnet/high) after every phase per project rule; manual redaction-output review on real `--dry-run` output before the first real (non-dry-run) invocation against Robin's actual history; manual precision audit in Phase 3.
- **Ongoing Observability**: none new required - this is an offline, on-demand CLI feature with no running service component; the existing CI suite is the ongoing regression guard.

---

## Rollback

Both importers write exclusively into dedicated, disposable namespaces (`imported-git`, `imported-claude-code`) - never into the default `knowledge` namespace. Rollback of a bad import is a namespace-scoped delete (Phase 1 confirms/builds this path), not a data-recovery problem. Source data (git repos, transcript files) is read-only to the importer at every phase - there is no path by which this feature can corrupt or lose the user's original history.

## Risk

- **Privacy/redaction miss** (highest risk): a secret or private-client string survives redaction and lands in the Kairn store. Mitigated by: mandatory `--dry-run` before first real run, a dedicated test corpus with a zero-miss bar (Success Criteria), and namespace-scoped rollback if something does slip through.
- **Precision/noise**: rule-based extraction produces low-value junk experiences that pollute recall. Mitigated by: the N>=50 audit gate in Phase 3, and the coarse-fallback kill-switch.
- **Volume/store bloat**: importing years of history could dwarf a user's organic Kairn usage. Mitigated by: `--since` windowing (Phase 4), dedicated namespaces (so imported vs. organic experiences are always distinguishable and separately prunable).
- **False sense of completeness**: a user might assume `--dry-run` output represents everything that could theoretically be extracted, when it's actually bounded by the (unvalidated-until-Phase-3) rule set's recall, not just its precision. Documented explicitly in the README so this isn't oversold.
- **[Discovered Risk, Interview Tier 3 Q7 - Devil's Advocate] Schema drift**: the Claude Code JSONL transcript schema (`type`, `message.role`, `timestamp`, `cwd`, `gitBranch`, content-block kinds) is internal and undocumented, not a versioned public API. A future CC release could change it and silently break the parser. Mitigated by defensive parsing (skip/log unrecognized shapes rather than crash) and a schema-shape smoke test built into Phase 3 that fails loudly on drift, rather than assuming the Stage-0-probed shape holds forever.

## Resume Protocol

**[Added, Interview Tier 1 Q4 - plan spans 1-2 weeks / >10h cumulative, a Resume Protocol was missing from the Stage 1 draft.]** Each phase is self-contained with a binary gate, an RC-gate review, and a git commit boundary. To resume after a session break: read this plan file, find the last phase whose gate is marked PASS with pasted evidence (in-plan or in the Phase 5 Verify Report companion once it exists), and start at the next unstarted phase. No phase depends on in-memory state from a prior session - only on the committed code state of the repo, which `git log`/`git status` on `/Users/neoforce/Business/engram` makes directly inspectable.

## Execution Vehicle & Orchestration

Default Vehicle: Single (self, Sonnet, effort high) for all 5 phases - this is sequential, single-codebase library work with no independent parallel streams; each phase's design decisions depend on the prior phase (redaction module before transcript parser, git-import namespace pattern reused by transcript importer). Standard Stage-5 execution rules apply regardless (RC-gate + EPT every phase, sub-agents only for genuinely parallel sub-tasks like the Phase 3 precision-audit sampling if it's large enough to warrant fan-out - if that sub-agent path triggers, it needs an explicit input/output contract defined at the time, not assumed).

**[Stage 3 fix, independent planner-agent review]**: deviation from the uniform default - Phase 2 (the plan's own self-declared highest-risk, load-bearing gate) and the redaction-integration portion of Phase 3 run at **effort xhigh**, not the uniform `high`, given the plan's own risk ranking. This is a targeted upgrade on the two riskiest phases, not a full escalation to Opus (self/orchestrator tier stays reserved, per the standing rule that Opus is never a delegation target).

## Security

- No network calls anywhere in the import path (reinforces zero-LLM/zero-network privacy story - the entire pipeline runs against local files only).
- Redaction is defense-in-depth, not the only control: `--dry-run` as a mandatory human-in-the-loop gate before any real write is the primary control; regex redaction is the secondary control.
- Git import is scoped to explicitly user-named repo paths only - never auto-discovery/auto-scan of arbitrary directories, to avoid accidentally pulling in repos with other people's private history.
- No secrets/credentials/private-client data may be routed to any external asset per the standing data-boundary rule - trivially satisfied here since nothing leaves the local machine.

## Dependencies

- No new third-party Python packages (Success Criteria). Git access via `subprocess` + the system `git` binary (already a hard requirement of working in this repo at all). JSONL parsing via stdlib `json`. Regex via stdlib `re`.
- Depends on the existing `Experience` model (`models/experience.py`) and `namespace`-aware storage layer (`storage/sqlite_store.py`) - both already present, no upstream blocking work needed before Phase 1 starts.

## Post-Completion

- Update README "Importing your history" section (Phase 4).
- `kn_learn` a decision node for the ratified plan + `kn_judge` edges to `758ece4c` (roadmap) and `7018bbd5` (reader-ceiling evidence, relevant context for why this doesn't touch recall) - done at Stage 4 finalization of this session, not deferred.
- No `_stats.json`/knowledge-graph/detection-index registration needed - this is a Kairn-repo-internal CLI feature, not an Evolving-repo system component (the Evolving artifact-registration rule does not apply to the engram repo).
- **[Stage 1.5 Note - The Manager perspective]**: no dedicated "Completion Gate" section (5-component Evolving template: Registration/Connections/Documentation/Orphan Detection/Consistency) is added, by deliberate dismissal rather than omission - this is a single-repo OSS CLI feature with no cross-system Evolving integration surface (confirmed above), so Phase 5's Success-Criteria-based gate + Verify Report serves the equivalent function for this context.

## Reference Library

| Source | Version/Date | What it informed | Link |
|---|---|---|---|
| Kairn `Experience` model | `src/kairn/models/experience.py`, main@818df24 | Confirmed `namespace`, `properties` (dict), overridable `id` fields exist already - no schema migration needed for the core design | (local file) |
| Kairn storage layer | `src/kairn/storage/sqlite_store.py`, main@818df24 | Confirmed `namespace` is an indexed, filterable column on `nodes`/`experiences`; flagged that a dedicated bulk-delete-by-namespace method needs to be added/confirmed in Phase 1 | (local file) |
| Claude Code transcript schema (empirical probe) | Sampled 2026-07-07, redacted-value inspection of a real 24-line JSONL session | Confirmed top-level fields (`type`, `message.role`, `timestamp`, `cwd`, `gitBranch`, `sessionId`) and that `assistant` content splits into `thinking`/`tool_use`/`text` blocks - directly shaped the Phase 3 parser design and the finding that `tool_use` blocks (not just message text) need redaction coverage | (local files under `~/.claude/projects/`) |
| Conventional Commits spec | v1.0.0 | Basis for the `fix:`/`feat:`/`docs:`/`chore:` prefix-to-experience-type mapping in Phase 1 | https://www.conventionalcommits.org/en/v1.0.0/ |
| Kairn roadmap decision node | Kairn `758ece4c` / `19831c9f` | Confirms WOW-9 selection and the "UPF plan first, Robin gates it" session shape | (Kairn store) |
| Kairn reader-ceiling diagnostic | Kairn `7018bbd5` (embedded in `19831c9f`/`3c57892c`) | Confirms this feature adds recall *supply* (more experiences to surface), not a recall *reading* fix - relevant to the NOT-scope boundary around not touching the recall path | (Kairn store) |

---

## Stage 0 Discovery Log (for the record)

1. **Existing work**: no `import` CLI command exists in `cli.py` today (28 existing commands enumerated, none named `import`) - greenfield feature, no dead/half-built prior attempt found.
2. **Feasibility**: `namespace` already indexed on both tables; `Experience.properties: dict` already exists for arbitrary metadata (source_ref) without a migration; git commit signal density sampled as strong in 2 real repos.
3. **Better alternatives (AHA check)**: considered treating claude-code + git import as one undifferentiated feature per the handoff's phrasing - rejected in favor of git-first/transcript-second sequencing (see DSV Suspend above) since it decouples a low-risk shippable MVP from the higher-risk, privacy-gated half.
4. **Key discovery reshaping Phase 3**: this ecosystem's own Claude Code sessions already call `mcp__kairn__kn_learn`/`kn_save` directly as `tool_use` blocks with structured `type`/`content` params - meaning naive full-transcript extraction would massively duplicate data already in the store for any session where Kairn was actively used. The extractor must specifically detect and skip these tool-call blocks, and instead target decision/gotcha-shaped prose that was *never* explicitly captured - which is also exactly the scenario (pre-Kairn-adoption history, or sessions where capture was missed) that gives this feature its real value.
5. **Volume**: 2,463 `.jsonl` files / 2.1GB measured in `~/.claude/projects/` alone (primary account), plus a distinct secondary-account tree (`~/.claude-secondary/projects/`, 5 project dirs) - confirms volume control (Phase 4) is a real, not hypothetical, concern.
6. **ROI**: high - this is Robin's explicitly-selected "next big Kairn build," framed as the structural differentiator competitors cannot copy (Kairn node `758ece4c`).

---

## Interview Log

**Date**: 2026-07-07 | **Mode**: Self | **Domain**: Software Development
**Questions asked**: 10 | **Anti-patterns found**: 3 (#11 Zombie Project - partial/overall-timeout gap; #18 Unverifiable Gates - partial/vague timing word; #20 Discovery Amnesia - dual-account asymmetry not yet reflected)

| Tier | Question | Answer/Decision | Anti-Pattern |
|------|----------|----------------|-------------|
| 1 | What kills this project overall, not just Phase 1? | Added an overall 2-calendar-week plan timeout (mirrors handoff's "1-2 week item"), ship-what-cleared-gate-so-far on timeout | #11 Zombie Project |
| 1 | Which assumption, if wrong, invalidates the most others? Validate that first. | Assumption 1 (transcript extraction precision) - moved its validation spike to run FIRST inside Phase 3, before the full parser is built | #1 Unvalidated Assumptions |
| 1 | "A few minutes" wall-clock budget - is that a real gate or a judgment word? | Tightened to a concrete 10-minute threshold against the measured 2,463-file/2.1GB corpus | #18 Unverifiable Gates |
| 1 | Plan spans 1-2 weeks (>10h) - where's the Resume Protocol? | Added: resume = find last PASSED gate in this file, start next phase; no in-memory session state, only committed repo state | (missing CONDITIONAL section) |
| 2 | Review checkpoint cadence - every phase is denser than UPF's every-2-phases guideline, is that a gap? | Not a gap - it matches this project's own stricter binding rule (RC-gate + EPT after every phase, never batched); left as-is with reasoning noted | - |
| 2 | Are phases scoped by files/features or time estimates? | Confirmed files/features (named modules per phase: `importers/git.py`, `redact.py`, `claude_code.py`) | - |
| 3 | Devil's Advocate: what makes this obsolete in 6 months? | Found real gap: Claude Code's JSONL schema is internal/undocumented and could drift - added as a Discovered Risk with a defensive-parsing + schema-shape-smoke-test mitigation | #20 Discovery Amnesia (risk not carried from Stage 0 probe into Risk section) |
| 3 | Discovery Consolidation: what did Stage 0 find that isn't reflected yet? | Dual-account asymmetry (195+ vs. 5 project dirs) - added default auto-scan-both-locations behavior to Phase 3 scope | #20 Discovery Amnesia |
| 3 | Cold Start Test: what would a zero-context reader misunderstand first? | "zero-LLM" could be misread as "no AI involvement ever" (it means no LLM call during import only); "import" could be misread as one-time migration rather than a repeatable, idempotent command - both flagged for the Phase 4 README | - |
| 3 | Is the 20% junk-rate precision bar an empirical number or a heuristic? | Heuristic, no prior benchmark exists for this exact task - flagged [Low confidence], explicitly Robin-adjustable at the real audit review, not fixed | #15 Numbers Without Sources (partial - now sourced as "heuristic, not empirical") |

**Plan changes made**: added overall plan timeout; reordered Phase 3 to spike-precision-first; tightened the timing assumption to a concrete number; added a Resume Protocol section; added a Discovered Risk (schema drift) with mitigation; added default dual-root auto-scan behavior to Phase 3; added two Cold-Start-Test clarifications for the eventual README; flagged the 20% bar as a heuristic, not an empirical constant. No original content removed or rewritten - all changes are additive per Plan Integrity Rules.
**Grade before**: B (solid 5-CORE + most CONDITIONAL sections, but missing Resume Protocol, one vague timing word, one un-carried Stage-0 risk)
**Grade after**: A (gaps closed; remaining flagged items are explicitly-acknowledged low-confidence heuristics, not silent gaps)

---

## Stage 1.5: Hardening Log

**Ran**: 2026-07-08 | **Mode**: Simple | **Perspectives**: 6/6

| Perspective | Finding | Action Taken |
|---|---|---|
| Outside Observer | "EPT" used in Success Criteria and Phase 5 without being spelled out for a zero-context reader | Fixed: expanded to "EPT (Empirical Proof Test - the standing 3-leg trigger/effect/consumer proof requirement)" at first use |
| Pessimistic Risk Assessor | The 2-week overall timeout sits at the top of the handoff's own "1-2 week" estimate with no extra buffer | [Stage 1.5 Note]: not mutated - this mirrors Robin's own stated timeframe, not this plan's independent estimate; the consequence is graceful (ship-what's-cleared), not a cliff, so the tightness is an acceptable tradeoff, not a defect |
| Pedantic Lawyer | Phase 4's gate used the soft phrase "README section renders correctly" | Fixed: tightened to "renders with no broken internal links and no unclosed code fences (verified by a markdown lint pass or manual render)" |
| Skeptical Implementer | Checked whether Phase 1 is cold-startable from the plan alone | No fix needed - confirmed startable (exact files, CLI signature, and gate all named) |
| The Manager | No dedicated "Completion Gate" section (5-component Evolving template) present | Fixed: added an explicit one-line dismissal-with-reason in Post-Completion (single-repo OSS feature, no cross-system Evolving integration surface) rather than leaving it silently absent (anti-pattern #20 avoidance) |
| The Manager | 5 phases sits exactly at the ">5 phases -> Incremental Delivery" boundary, not clearly over it | [Stage 1.5 Note]: no fix - the plan is already functionally incremental via the git-first/transcript-second sequencing decision (Phase 1 is independently shippable), which serves the same purpose as a dedicated section |
| Devil's Advocate | Confirmed the Stage 0 AHA check (item 3) was a real, applied decision (git-first sequencing), not compliance theater | No fix needed - verified genuine |
| Devil's Advocate | Speculative external risk not yet mentioned: a future native Claude-Code memory-import feature could erode this feature's differentiation | [Stage 1.5 Note]: awareness only, no concrete mitigation exists today; not added as a Risk-section entry since it has no actionable response, unlike the schema-drift risk which does |

**Discovery Consolidation**: All 6 Stage 0 Discovery Log findings are addressed in the plan body (existing-work check -> Phase 1 scope; feasibility -> Assumptions; AHA/sequencing -> DSV Suspend + phase order; kn_learn-duplication -> Phase 3 scope; volume/dual-account -> Assumptions + Phase 3/4; ROI -> Context & Why). 0 orphaned.

**Structural fixes**: 3 | **Notes for user**: 3

---

## Stage 3: Review Log

**Ran**: 2026-07-08 | **Vehicle**: dispatched to the independent `planner` subagent (per Kairn pattern `ef0bba38`/`f7fe0e53`: independent second-pass planner-agent review catches Grade-A blockers a self-review misses, even on an already-self-graded-A plan) | **Grade before**: A (self-graded, Stage 1.5) | **Grade after independent review**: B, one fix-cycle from A

The independent reviewer confirmed Stage 0 discovery, the DSV/AHA sequencing decision, FAILED conditions, Resume Protocol, and Discovery Consolidation were all genuinely solid - but found 5 issues the self-review's structural checklist wasn't positioned to catch because they require cross-referencing *different* sections against each other (Scope vs. Gate; Reference Library vs. Success Criteria) rather than re-scanning one section in isolation:

| # | Finding | Fix Applied |
|---|---|---|
| 1 | Phase 1's scope includes building/confirming the namespace-scoped delete (rollback) path, but its gate never tested it - could pass without the rollback mechanism actually existing | Added an explicit gate line: delete path must be empirically tested (write N rows, delete by namespace, confirm zero remain) |
| 2 | Phase 3's scope includes a schema-shape smoke test (mitigating the Discovered Risk), but neither the Deliverable nor Gate required it to exist | Added an explicit gate line requiring the smoke test present and passing against a fresh real sample |
| 3 | Phase 4's scope is conditional on the Phase 4 timing spike result, but its gate never checked the measured time against the 10-minute threshold or that `--since` default-on was actually added if required | Added an explicit gate branch: measured time checked against threshold, `--since` default-on demonstrated if over, or an explicit "no change needed" note if under |
| 4 (Red Flag) | Phase 2's redaction gate ("zero missed matches") is scored against a test corpus that phase itself authors - self-validating by construction, a real weakness on the plan's own self-declared highest-risk item | Strengthened Phase 2 scope: the test corpus must incorporate a sample of gitleaks' public default-rule fixtures as data (not a new dependency), not just a self-authored synthetic corpus |
| 5 (Red Flag) | Phase 5's EPT gate (>=10 recall queries surface an imported experience) has no defined behavior when Kairn's known pre-existing reader-ceiling limitation (Kairn `7018bbd5`) caps what surfaces - conflates "import broken" with "recall pre-existing ceiling, out of scope" | Added an explicit gate branch: diagnose reader-ceiling-caused shortfall (does not block ship) vs. actual import/write defect (blocks ship) before treating the gate as failed |

**Additional lower-priority fixes applied** (Anti-Pattern #15 consistency + Execution Vehicle parameterization, both flagged by the same review):
- The 10-minute timing threshold (Assumptions) is now flagged `[Low confidence]` with the same "heuristic, not a formal benchmark" disclaimer already applied to the 20% junk-rate bar, for epistemic consistency.
- Execution Vehicle & Orchestration: Phase 2 and the redaction-integration portion of Phase 3 bumped from uniform `effort high` to `effort xhigh`, matching the plan's own risk ranking (targeted upgrade, not an Opus escalation).
- Phase 3's conditional sub-agent fan-out path now explicitly requires an input/output contract at the time it triggers, closing a latent #17-adjacent gap (Delegation Without Context Transfer) the reviewer flagged as low-priority but real.
- Phase 4's CLI help-text deliverable now has a concrete acceptance bar (every flag documented, verified by reading rendered `--help` output).

**Grade after fixes applied**: A (all 2 Red Flags and 3 gate/scope mismatches closed; both Anti-Pattern #15 and Execution Vehicle findings addressed). Not re-verified by a third independent pass - per UPF guidance this iterates until Red Flags clear, which they now are; Stage 4 finalization proceeds on this basis.
