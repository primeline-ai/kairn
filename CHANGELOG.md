# Changelog

## 0.3.1 - 2026-10-11

### Changed
- `kairn serve` opens `~/.kairn` when no path is given, like `kairn init`. The plugin starts `kairn serve --init` without a path.

### Fixed
- A failed database open waits up to five seconds for aiosqlite's worker thread to stop before returning the error, avoiding "Event loop is closed" after the caller's loop closes. It logs a warning if the thread is still running. aiosqlite 0.21 or newer is required.
- `KAIRN_LOG_LEVEL` is read by `kairn serve` only, so `kairn init` no longer writes it into config.yaml.

### Added
- The Kairn plugin for Claude Code and Codex (`plugin/`, installable with `/plugin marketplace add primeline-ai/kairn`).

### All changes since 0.3.0
- 7e1caef Tests: kairn serve defaults to ~/.kairn, and the plugin command names no path
- 77ed755 Merge pull request #42 from primeline-ai/docs/plugin-desktop-path
- 2658f32 Merge pull request #41 from primeline-ai/ci/plugin-launch-windows
- 46b8707 Merge pull request #40 from primeline-ai/fix/store-open-failure-thread
- 3606a5a Add the Kairn plugin for Claude Code and Codex (#39)
- 8134d23 Merge pull request #38 from primeline-ai/fix/followups-after-0.3.0
- e7ced92 Merge pull request #37 from primeline-ai/chore/server-json-sha-0.3.0
- d549a1d Merge pull request #36 from primeline-ai/feat/serve-init

## 0.3.0 - 2026-10-10

### Added
- `kairn serve --init` creates a missing workspace before starting the MCP server.

### All changes since 0.2.1
- 5891ff5 Add a 30 second panel video of how the repo works to the README header
- ae27170 fix: read the user config and packaging test inputs as UTF-8
- abcd4ea chore(deps): update fastmcp requirement from <4.0,>=2.14 to >=2.14,<5.0 (#32)
- 5bf95f8 chore(deps): bump pyjwt from 2.13.0 to 2.15.0 in /packaging/mcpb (#33)
- 3a51cf7 fix(tests): compare census paths with forward slashes on every OS
- 086b851 fix(tests): read and write source files as UTF-8 so the suite passes on Windows
- 80ece29 fix(recall): the model reported pure decay, and doctor answered with a traceback
- ec913d1 fix(config): say what the match floor measures, instead of calling it a percent
- a4669e3 fix(recall): one match scale across every surface that reports it
- eabdbb5 fix(recall): a question with nothing searchable in it is not a browse
- 2a52a6d docs(fts): name the caller that was actually exposed, measured not assumed
- 766a801 fix(fts): a query costs what its vocabulary costs, not the prompt's length
- 72074d3 fix(fts): a query costs what its vocabulary costs, not what the prompt's length costs
- 48a09c7 feat(recall): one compensatory score for experiences, instead of a decay bucket
- f3d2a51 docs: drop the Alpha label, and make the performance table reproducible (#31)
- 6ae4387 Rank context() nodes instead of returning an arbitrary handful (#30)
- 5f4008b Say what a relevance number means, and stop one source eating the whole result budget (#29)
- 06ba901 fix(recall): weight node relevance by query-term coverage, and make the tokenizer unicode-aware (#26)
- 3466c1c List Kairn in the official MCP registry (#28)
- 1de2bbe Ship an MCP Bundle so Kairn is installable outside pip (#27)
- 655b9b8 fix(recall): make experience relevance match-aware, and let it abstain
- 83dddee fix(recall): scale node relevance by query-term coverage
- f496be7 ci: run the suite on macOS and Windows, not only Linux (#24)
- 7212635 feat(recall): opt-in local-embedding semantic rerank (semantic_recall flag) (#23)
- e30f0a5 fix(recall): honest BM25 relevance for nodes + live min_relevance gate (#22)
- f2b8e6c fix(security): weakness-audit engine hardening - atomic route merge, git-import redaction, fail-closed JWT, uniform kn_* namespace (#21)
- b2095c2 docs(import): README 'Importing your history' - claude-code + privacy model (WOW-9 Phase 4) (#20)
- 72f5486 feat(import): kairn import claude-code - coarse transcript importer (WOW-9 Phase 3) (#19)
- 9e689b5 chore: stop tracking local .claude/plans working notes
- 4199297 feat(import): secret-redaction module for transcript import (WOW-9 Phase 2) (#18)
- a839bd4 fix: normalize bare-date --since to midnight (CI failure root cause)
- d7d25f5 chore: track item4/item5 UPF plan files (were untracked since creation)
- 098e4e6 docs(plan): mark WOW-9 Phase 1 gate PASS with EPT + RC-gate evidence
- 3e5f95c fix: address RC-gate findings in kairn import git
- 2f1fc8b docs: add benchmark scorecard visual + kairn import git section
- ac113d8 feat(import): add kairn import git - deterministic commit-metadata importer
