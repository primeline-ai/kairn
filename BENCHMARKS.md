# Kairn Benchmarks

Memory systems are easy to describe and hard to measure. This page reports
Kairn's recall quality on a published, third-party benchmark so the "context
aware" claim is a number, not an adjective.

## LongMemEval

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) is a 500-question
benchmark for long-term conversational memory. Each question ships with a long
multi-session chat "haystack"; a system must ingest it, recall the relevant
turns, answer, and a judge scores the answer against gold.

**Result: 56.2% overall on LongMemEval-S (500 questions).**

| Run | Date | Reader | Judge | Sample | Score |
|-----|------|--------|-------|--------|-------|
| `full-500-s` | 2026-06-12 | GPT-4o | GPT-4o | 500 / 500 | **0.562** |

500 of 500 questions scored, 0 errors. Mean recall latency: **1.4 ms** per
query (FTS5, in-process, no network).

### Per-category breakdown

The headline number hides where the work is. Kairn is strong on single-session
recall and abstention, and weak on cross-session synthesis, temporal reasoning
and one-shot preferences. This is the roadmap, in data:

| Category | n | Accuracy |
|----------|----|----------|
| single-session-user | 70 | 91.4% |
| single-session-assistant | 56 | 83.9% |
| knowledge-update | 78 | 70.5% |
| temporal-reasoning | 133 | 42.9% |
| multi-session | 133 | 41.4% |
| single-session-preference | 30 | 10.0% |
| abstention | 30 | 96.7% |

Read it as: Kairn reliably finds a fact stated once in a session (91%) and
reliably declines to answer when the fact was never stated (97%). It loses
ground when an answer must be assembled across many sessions or resolved on a
timeline. Those weak categories are the largest, so they set the ceiling.
Each red cell has a diagnosis below; the numbers stay published while the
diagnoses are worked.

### Temporal reasoning: 42.9%

Kairn surfaces the right sessions but does not reason over a timeline. A
day-granularity bug in the flag-gated bi-temporal path (as-of filtering
compared full ISO timestamps where the dataset speaks in days) was found and
fixed (PR #8); that alone does not move this cell, because the shipped recall
path does not yet use validity windows at query time. The open lever is a
timeline-aware query path; until that ships, 42.9% is the honest number.

### Multi-session: 41.4%

Assembling one answer from evidence spread across many sessions. A
density-preserving entity diversification for the flag-gated bi-temporal path
was built and measured head-to-head (PR #9). On retrieval metrics it
Pareto-dominates the legacy diversifier (coverage 0.797 / match-density 3.333
vs 0.811 / 2.600 legacy, against the 0.737 / 4.533 engine baseline), but it
FAILED the accuracy gate on a 50-question pilot: multi-session 0.3846 vs
0.4615 baseline (n=13, exactly one question worse). It therefore shipped
library-only - the production recall path is byte-identical - and this cell
is published unchanged. Two negative results worth keeping: any answer-blind
promoter saturated near 0.82 coverage at top-k 8 (recall ceiling 0.951 in
that configuration), and scored promotion did worse than plain BM25-greedy
selection. The remaining gap looks like a reading/synthesis problem, not pure
retrieval.

### Single-session preference: 10.0%

The worst cell on the board, published rather than gamed, because its
mechanism is fully diagnosed and sits outside storage. Two independent
blocks:

1. **Reader abstention.** Preference questions are recommendation-shaped
   ("suggest resources for..."). The benchmark reader answers only from
   recalled excerpts and abstains otherwise; a recommendation must be
   generated and is never verbatim in memory, so the reader abstains before
   memory quality even matters. Probe: three preference questions were given
   a manually written, ideal explicit preference statement as the top memory,
   the upper bound of anything a storage system could produce, and the reader
   still abstained on 3 of 3 (`runs/diag/writetime-synthesis-probe.json` in
   the harness tree).
2. **Entailment mismatch.** The gold answers for this category are
   meta-descriptions of the preference. On the rare non-abstaining path, a
   good recommendation that merely applies the preference does not entail the
   meta-description text, so the judge scores it wrong: a preference-aware
   reader variant scored 1/8 vs 0/8 for the standard reader
   (`runs/diag/read-ceiling-probe.json`).

Recall itself is not the bottleneck - the right evidence lands in the top-8
excerpts for 25 of 30 preference questions. Consequence: no storage-side
change (capture, synthesis, decay, recall) can move this cell under the
current protocol. The only real lever is re-calibrating the benchmark reader,
which would touch every category, so the 10% stays on the board until the
protocol changes openly rather than being quietly tuned for one cell.

## How it compares

LongMemEval is run by several memory products. Treat cross-vendor numbers as
directional, not a controlled head-to-head: readers, ingestion and judge
prompts differ between published runs.

| System | LongMemEval-S | Source |
|--------|---------------|--------|
| Zep / Graphiti | 63.8% | vendor-published |
| **Kairn** | **56.2%** | this repo, protocol below |
| mem0 | 49.0% | vendor-published |

## Protocol and honesty notes

- **Reader + judge:** GPT-4o for both, matching the published LongMemEval
  protocol (binary judge of hypothesis vs gold).
- **Recall path:** the live `kn_memories` path - free text is shaped into an
  FTS5 keyword query (`core.fts.to_fts_query`), matched with BM25 ranking, and
  re-ordered with decay as a coarse tiebreak.
- **Determinism:** fixed sample seed (42), greedy decoding where the API allows.
- **Conservative lower bound:** each ingested turn-pair is truncated to ~1800
  characters before storage, so long turns lose tail context. Real scores with
  full-length ingestion would be equal or higher.
- **Single run:** one full pass, not an average of several. No cherry-picking
  across runs.
- **Sandbox:** every question runs against a fresh, throwaway SQLite database;
  no production workspace is read or written during a run.

## Reproducing

The accuracy harness is not bundled in this repo: it lives in the
[Evolving](https://github.com/primeline-ai) research tree and drives Kairn's
public library directly (`ExperienceEngine.search`, the same call
`kn_memories` makes). To reproduce:

1. Download the LongMemEval-S dataset from the upstream repo.
2. For each question: ingest the haystack sessions as experiences, recall the
   top-k with `ExperienceEngine.search(text=...)`, answer with a reader LLM
   from the recalled excerpts, and judge the answer against gold.
3. Keep the sandbox contract: one fresh DB per question, production workspace
   untouched (verify by hashing the DB file before and after).

The performance half ships with Kairn itself: `kairn benchmark <workspace>`
measures insert, FTS5 query, and graph traversal latency at scale on your own
machine. It does not run LongMemEval.

## Bi-Temporal Level-Up: findings (2026-06-13)

A bi-temporal recall mode (`search_bitemporal`: as-of validity filtering +
session/valid-time diversification) was built behind a flag and measured
head-to-head. It **regressed** on LongMemEval-S (pilot-50: overall 0.38 vs 0.54
baseline, temporal-reasoning 0.0, multi-session 0.31). Diagnosis: returning
session-diverse excerpts trades match-strength density for breadth, so the
reader gets the right *sessions* but not the right *content* and abstains. An
ablation (diversification off, as-of only) recovered to 0.50 but stayed below
baseline. **Conclusion: plain BM25 + decay (`ExperienceEngine.search`) remains
the recall path; the bitemporal mode is flag-gated and off by default.** The
single-session-preference 10% category was diagnosed as a reader/task-framing
ceiling, not an engine recall defect (the right evidence already lands in the
top-8 for 25/30 questions).

The bi-temporal **schema** (`valid_from`/`valid_to` valid-time window,
orthogonal to decay) and a rule-based cross-session `entity_key` ship as
additive infrastructure. Decay `HALF_LIVES` were recalibrated against the real
access tail (`scripts/calibrate_halflives.py`; prior values were 3-21x longer
than observed p95 re-access).
