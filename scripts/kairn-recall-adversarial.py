#!/usr/bin/env python3
"""Adversarial abstention harness for the semantic_recall / abstention path.

Runs the externally-authored SHOULD-ABSTAIN / SHOULD-RETURN queries
(scripts/data/kairn-recall-adversarial.json) through the kairn CLI against a
store you point it at, and reports abstain vs return accuracy. The decision is
taken on NODE results (the surface semantic_recall controls); an alien query
should surface no nodes, a paraphrased in-domain query should surface some.

Point it at a store with the flag ON (config.yaml semantic_recall: true, then a
backfill) to measure the flag-ON lift over the keyword baseline. Run BEFORE and
AFTER, the honest before/after acceptance pattern - the 30 queries are external
(written blind by grok-build + Fugu), so a high score is not self-graded.

Usage:
  python3 scripts/kairn-recall-adversarial.py --db /path/to/workspace-dir
  python3 scripts/kairn-recall-adversarial.py --db ./ws --kairn-bin ~/.local/bin/kairn
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = REPO_ROOT / "scripts" / "data" / "kairn-recall-adversarial.json"


def run_recall(kairn_bin: str, db_dir: str, query: str, k: int) -> list[dict]:
    out = subprocess.run(
        [kairn_bin, "recall", str(db_dir), "--topic", query, "--limit", str(k)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if out.returncode != 0:
        raise RuntimeError(f"kairn recall failed: {out.stderr.strip()[:200]}")
    return json.loads(out.stdout).get("results", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="workspace dir containing kairn.db")
    ap.add_argument("--kairn-bin", default="kairn", help="kairn CLI path")
    ap.add_argument("--label", default=None, help="free-text run label for stdout")
    args = ap.parse_args()

    spec = json.loads(QUESTIONS.read_text())
    k = int(spec.get("default_k", 10))
    queries = spec["queries"]

    hits = 0
    false_abstentions: list[str] = []  # SHOULD-RETURN but got nothing (cardinal sin)
    false_passes: list[str] = []  # SHOULD-ABSTAIN but returned (polysemy leak)
    for q in queries:
        results = run_recall(args.kairn_bin, args.db, q["query"], k)
        nodes = [r for r in results if r.get("source") == "node"]
        decision = "RETURN" if nodes else "ABSTAIN"
        if decision == q["expected"]:
            hits += 1
        elif q["expected"] == "RETURN":
            false_abstentions.append(q["query"][:60])
        else:
            false_passes.append(q["query"][:60])
        mark = "PASS" if decision == q["expected"] else "FAIL"
        print(f"  {q['id']:7} want={q['expected']:7} got={decision:7} {mark}  {q['query'][:52]}")

    total = len(queries)
    print(f"\nadversarial: {hits}/{total} ({hits / total * 100:.0f}%)"
          + (f'  [{args.label}]' if args.label else ''))
    print(f"  false-abstentions (cardinal sin, {len(false_abstentions)}): {false_abstentions}")
    print(f"  false-passes (polysemy leak, {len(false_passes)}): {false_passes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
