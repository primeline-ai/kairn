"""Round 2 of the P5 review: two findings the first round's own tests could
not see, and both are the same shape as the defect they were guarding.

F1 - THE MODEL KEPT REPORTING PURE DECAY. `Experience.to_response()` still
computed `round(self.relevance(), 3)` after every wire surface had been moved
onto the match-aware value, and the census test written to catch exactly that
was hard-scoped to a two-file list, so it could not enumerate the site it was
missing. The rule now has ONE implementation, on the model, and the census
sweeps the whole package with the definition exempted BY NAME.

F2 - `doctor` BYPASSED THE ERROR ENVELOPE. 18 of the 19 `_build_intel_stack`
callers in `cli.py` run through `_run_json`, which turns a ValueError into
`{"_v", "error"}` on stderr with exit 1. `doctor` is the one that cannot: it
needs the report before it can compute its exit code. So a broken config gave
it a raw traceback while every sibling command gave JSON - and `doctor --json`
documents "the same shape as the MCP tool kn_doctor". The exposure was NEW:
before the config validator existed, `_build_intel_stack` could not raise here,
which is why no earlier test covered it.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from kairn.cli import main as cli_main
from kairn.models.experience import Experience
from kairn.storage.sqlite_store import SQLiteStore


def _ws(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def _seed(db: Path) -> None:
    """Synchronous on purpose. `CliRunner.invoke` runs `asyncio.run` inside the
    command, which raises if a loop is already running - so every CLI test in
    this file is a plain `def`, never an async one."""

    async def _go() -> None:
        store = SQLiteStore(db)
        await store.initialize()
        await store.close()

    asyncio.run(_go())


def _cli(*args: str):
    return CliRunner().invoke(cli_main, list(args))


def _cli_text(result) -> str:
    """Everything the command printed. `result.output` and `result.stderr` are
    the SAME buffer in this click version, so they are not concatenated - doing
    that duplicated every line and made a line-oriented parse ambiguous."""
    return result.output or ""


# ── F1: the model reports the same quantity as every other surface ───


def _experience(**kw) -> Experience:
    kw.setdefault("decay_rate", 0.01)
    return Experience(type="solution", content="alpha beta gamma", **kw)


def test_f1_to_response_reports_the_match_aware_value_when_a_recall_set_one():
    """The defect: this method returned pure decay while kn_recall,
    kn_memories, kn_context and the CLI all returned the composite."""
    exp = _experience()
    exp.recall_relevance = 0.1234
    assert exp.to_response()["relevance"] == 0.1234
    assert exp.to_response(detail="full")["relevance"] == 0.1234


def test_f1_to_response_falls_back_to_decay_and_keeps_its_3_decimal_contract():
    """Negative control. Without a recall value nothing changes, including the
    number of decimals - this method's wire contract is 3, not the helper's 4.
    """
    exp = _experience()
    # It is a declared field with a None default, not an absent attribute, so
    # the rule keys on the VALUE. `hasattr` is True either way and would make
    # this precondition trivially satisfied.
    assert exp.recall_relevance is None
    reported = exp.to_response()["relevance"]
    assert reported == pytest.approx(round(exp.relevance(), 3))
    assert len(str(reported).split(".")[-1]) <= 3, reported


def test_f1_there_is_exactly_one_implementation_of_the_reporting_rule():
    """`intelligence._reported_relevance` must DELEGATE, not re-implement.

    Two copies of this rule are what let the model drift away from the wire
    surfaces in the first place, so the delegation is pinned rather than left
    to convention.
    """
    from kairn.core import intelligence

    calls: list[object] = []

    class _Probe:
        def reported_relevance(self, *, at=None):
            calls.append(at)
            return 0.5

    assert intelligence._reported_relevance(_Probe(), None) == 0.5
    assert calls == [None], "the helper did not route through the model's method"


# ── F2: doctor answers in the documented shape, even on a broken config ──


def test_f2_doctor_emits_the_error_envelope_on_an_out_of_range_floor(tmp_path):
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    _seed(db)
    (ws / "config.yaml").write_text("experience_min_match: 5.0\n")

    result = _cli("doctor", str(ws))
    assert result.exit_code == 1, _cli_text(result)
    text = _cli_text(result)
    assert "Traceback" not in text, text
    payload = json.loads([ln for ln in text.splitlines() if ln.strip().startswith("{")][0])
    assert payload["_v"] == "1.0"
    assert "experience_min_match" in payload["error"]


def test_f2_doctor_still_reports_normally_on_a_good_config(tmp_path):
    """Positive control: the new except must not swallow the ordinary path."""
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    _seed(db)
    (ws / "config.yaml").write_text("experience_min_match: 0.0\n")

    result = _cli("doctor", str(ws))
    text = _cli_text(result)
    assert "Traceback" not in text, text
    payload = json.loads(text[text.index("{") :])
    assert "summary" in payload, payload
    assert "error" not in payload, payload


def test_f2_a_sibling_command_answers_in_the_same_shape(tmp_path):
    """The comparison that made the gap visible: `recall` already did this."""
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    _seed(db)
    (ws / "config.yaml").write_text("experience_min_match: 5.0\n")

    result = _cli("recall", str(ws), "--topic", "anything")
    text = _cli_text(result)
    assert result.exit_code == 1, text
    assert "Traceback" not in text, text
    payload = json.loads([ln for ln in text.splitlines() if ln.strip().startswith("{")][0])
    assert payload["_v"] == "1.0"
    assert "experience_min_match" in payload["error"]
