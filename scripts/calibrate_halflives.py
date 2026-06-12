#!/usr/bin/env python3
"""Re-runnable derivation for core/experience.py:HALF_LIVES decay calibration.

Profiles a Kairn experience store's REAL access tail and prints, per type, the
observed re-access interval percentiles (the empirical basis for tail-anchored
half-lives). Read-only: never writes the DB (copy it first if it is live).

Rationale: a half-life should keep a still-valid fact recallable at its p99
re-access interval (relevance ~0.6-0.7), not decay it to near-zero. Anchor the
half-life at ~1.5-2x the observed p99 - "calibrate on the tail, never the mean".

Usage:
    python scripts/calibrate_halflives.py /path/to/kairn.db
    # (against a live DB, copy first: cp live.db /tmp/c.db; ... /tmp/c.db)
"""
from __future__ import annotations

import math
import sqlite3
import sys

TYPES = ["solution", "pattern", "decision", "workaround", "gotcha"]


def percentile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def main(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    print(f"# Decay half-life derivation over {db_path}\n")
    total = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
    never = conn.execute(
        "SELECT COUNT(*) FROM experiences WHERE access_count = 0").fetchone()[0]
    print(f"experiences={total}  never_re_accessed={never} ({100*never/total:.1f}%)\n")
    print(f"{'type':<11} {'n_reacc':>7} {'p50':>6} {'p95':>6} {'p99':>6} {'max':>6} "
          f"{'-> half_life':>12}")
    for t in TYPES:
        rows = conn.execute(
            """SELECT julianday(last_accessed) - julianday(created_at) AS d
               FROM experiences
               WHERE type=? AND last_accessed IS NOT NULL
                 AND created_at IS NOT NULL AND last_accessed > created_at
               ORDER BY d""", (t,)).fetchall()
        vals = [r[0] for r in rows if r[0] is not None]
        p50, p95, p99 = (percentile(vals, q) for q in (0.50, 0.95, 0.99))
        mx = vals[-1] if vals else None
        # Tail-anchored recommendation: ~2x p99 (>=30d floor for sparse types).
        hl = max(30.0, round(2 * p99)) if p99 else 40.0
        fmt = lambda x: f"{x:.1f}" if x is not None else "  -"  # noqa: E731
        print(f"{t:<11} {len(vals):>7} {fmt(p50):>6} {fmt(p95):>6} {fmt(p99):>6} "
              f"{fmt(mx):>6} {hl:>12.0f}")
    print("\n# Shipped HALF_LIVES are rounded, slightly conservative vs ~2x p99,")
    print("# keeping structural types (pattern/decision) a little longer.")
    conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
