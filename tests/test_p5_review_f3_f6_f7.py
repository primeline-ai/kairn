"""P5 review findings F3, F6, F7 - regression tests.

F3  One search must report ONE scale. `kn_memories`, the `kn://memories`
    resource and the CLI `memories` command emitted `round(e.relevance(), 4)`
    (pure time decay) while `recall` / `crossref` / `context` reported the
    match-aware composite. A caller comparing two numbers from the same store
    was comparing different quantities.

F6  `experience_min_match` is a fraction in [0.0, 1.0]. An out-of-range value
    (the review reproduced `65`) was accepted silently and became a floor no
    score can clear, so recall came back EMPTY. An empty result is an answer;
    "your config is wrong" is not the same statement. The invalid value must
    be LOUD, never fail closed into a plausible-looking zero.

F7  The floor must reach EVERY IntelligenceLayer construction. It reached two
    of three: the CLI demo path built a layer with no floor at all.

Instrument notes (deliberate, do not "simplify" these away):
  * The CLI is exercised IN PROCESS via click's CliRunner. The suite's older
    CLI tests shell out to `sys.executable -m kairn.cli`, which resolves
    `kairn` through the editable-install .pth and therefore does NOT exercise
    this working tree.  test_positive_control_* pins that down.
  * Every census assertion carries a non-vacuity guard. An empty collection
    makes an "all of them" assertion trivially true.
"""

from __future__ import annotations

import ast
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from fastmcp import Client

import kairn
from kairn.cli import main as cli_main
from kairn.core.experience import ExperienceEngine
from kairn.core.graph import GraphEngine
from kairn.core.ideas import IdeaEngine
from kairn.core.intelligence import IntelligenceLayer
from kairn.core.memory import ProjectMemory
from kairn.core.router import ContextRouter
from kairn.events.bus import EventBus
from kairn.server import create_server
from kairn.storage.sqlite_store import SQLiteStore

SRC_KAIRN = Path(kairn.__file__).resolve().parent

QUERY = "kubernetes helm rollout"
SEED: list[tuple[str, str]] = [
    ("solution", "Kubernetes helm rollout failed because the chart pinned an old image tag"),
    ("gotcha", "Helm rollback leaves orphaned kubernetes configmaps behind"),
    ("pattern", "Blue green deployment on kubernetes with helm charts and a canary gate"),
    ("decision", "Chose postgres over mysql for the billing service"),
    ("workaround", "Restart the docker daemon when the socket goes stale"),
]
EXP_TYPES = {t for t, _ in SEED}

# Same-scale tolerance. The two paths compute decay at slightly different
# wall-clock instants, so exact equality would be flaky for the wrong reason.
# The bug this guards produces a gap of ~0.8, five orders of magnitude larger.
SAME_SCALE_TOL = 1e-5


# ── fixtures / helpers ───────────────────────────────────────────────

def _ws(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def _write_config(ws: Path, value: object) -> None:
    (ws / "config.yaml").write_text(f"experience_min_match: {value}\n")


async def _seed(db: Path) -> None:
    store = SQLiteStore(db)
    await store.initialize()
    bus = EventBus()
    engine = ExperienceEngine(store, bus)
    for type_, content in SEED:
        await engine.save(type=type_, content=content)
    await store.close()


async def _build_intel(db: Path, **kwargs):
    """The engine stack, wired exactly as the CLI/server wire it."""
    store = SQLiteStore(db)
    await store.initialize()
    bus = EventBus()
    graph = GraphEngine(store, bus)
    router = ContextRouter(store, bus)
    memory = ProjectMemory(store, bus)
    experience = ExperienceEngine(store, bus)
    ideas = IdeaEngine(store, bus)
    intel = IntelligenceLayer(
        store=store,
        event_bus=bus,
        graph=graph,
        router=router,
        memory=memory,
        experience=experience,
        ideas=ideas,
        **kwargs,
    )
    return store, intel


async def _recall_experience_relevances(db: Path) -> dict[str, float]:
    store, intel = await _build_intel(db)
    try:
        results = await intel.recall(topic=QUERY, limit=20)
    finally:
        await store.close()
    return {r["id"]: r["relevance"] for r in results if r.get("source") == "experience"}


async def _pure_decay(db: Path) -> dict[str, float]:
    """What the buggy sites reported: round(e.relevance(), 4), no match term."""
    store = SQLiteStore(db)
    await store.initialize()
    try:
        engine = ExperienceEngine(store, EventBus())
        rows = await engine.search(limit=100)
        now = datetime.now(UTC)
        return {e.id: round(e.relevance(at=now), 4) for e in rows}
    finally:
        await store.close()


def _cli(*args: str):
    return CliRunner().invoke(cli_main, list(args))


def _cli_text(result) -> str:
    out = result.output or ""
    try:
        out += result.stderr or ""
    except (ValueError, AttributeError):  # stderr not captured separately
        pass
    if result.exception is not None:
        out += repr(result.exception)
    return out


# ── positive control on the instrument itself ────────────────────────

def test_positive_control_imports_come_from_this_worktree():
    """If `kairn` resolves to the editable install, every test below is a
    check that cannot fail: it would grade the LIVE tree, not this branch."""
    this_repo = Path(__file__).resolve().parents[1]
    expected = this_repo / "src" / "kairn"
    assert expected == SRC_KAIRN, f"tests import kairn from {SRC_KAIRN}, not from {expected}"


# ── F3: one search, one scale ────────────────────────────────────────

async def test_f3_server_kn_memories_uses_the_same_scale_as_recall(tmp_path: Path):
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    await _seed(db)

    recalled = await _recall_experience_relevances(db)
    assert recalled, "fixture produced no experiences through recall"

    server = create_server(str(db))
    async with Client(server) as client:
        payload = json.loads(
            (await client.call_tool("kn_memories", {"text": QUERY, "limit": 20})).content[0].text
        )
    reported = {e["id"]: e["relevance"] for e in payload["experiences"]}

    shared = set(recalled) & set(reported)
    assert shared, "no experience id appeared on both paths"
    diverged = {
        i: (recalled[i], reported[i])
        for i in shared
        if abs(recalled[i] - reported[i]) > SAME_SCALE_TOL
    }
    assert not diverged, f"same store, same query, two scales: {diverged}"


async def test_f3_server_kn_memories_is_not_pure_decay_for_a_text_query(tmp_path: Path):
    """The discriminator. Same-scale alone would also pass if BOTH paths
    regressed to pure decay, so pin the reported number off that scale."""
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    await _seed(db)
    decay = await _pure_decay(db)

    server = create_server(str(db))
    async with Client(server) as client:
        payload = json.loads(
            (await client.call_tool("kn_memories", {"text": QUERY, "limit": 20})).content[0].text
        )
    reported = {e["id"]: e["relevance"] for e in payload["experiences"]}
    assert reported, "kn_memories returned nothing to grade"

    off_decay = [i for i in reported if abs(reported[i] - decay[i]) > 1e-3]
    assert off_decay, (
        "every kn_memories relevance equals pure decay for a TEXT query - "
        f"reported={reported} decay={ {k: decay[k] for k in reported} }"
    )


async def test_f3_server_browse_query_still_reports_pure_decay(tmp_path: Path):
    """No-regression control: with no text there is no match strength, so the
    reported number must stay pure decay. Guards against the fix turning the
    browse path into something else."""
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    await _seed(db)
    decay = await _pure_decay(db)

    server = create_server(str(db))
    async with Client(server) as client:
        payload = json.loads((await client.call_tool("kn_memories", {"limit": 20})).content[0].text)
    reported = {e["id"]: e["relevance"] for e in payload["experiences"]}
    assert reported
    for i, value in reported.items():
        assert abs(value - decay[i]) < 1e-3, f"browse relevance moved off decay for {i}"


async def test_f3_resource_memories_reports_pure_decay_for_browse(tmp_path: Path):
    """Second server site (kn://memories). It is a browse read, so the value
    must equal decay - but it must reach that value through the SAME helper,
    which the source census below pins."""
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    await _seed(db)
    decay = await _pure_decay(db)

    server = create_server(str(db))
    async with Client(server) as client:
        contents = await client.read_resource("kn://memories")
    payload = json.loads(contents[0].text)
    assert payload["count"] >= 1
    for item in payload["experiences"]:
        assert abs(item["relevance"] - decay[item["id"]]) < 1e-3


def test_f3_cli_memories_uses_the_same_scale_as_cli_recall(tmp_path: Path):
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    asyncio.run(_seed(db))

    r_recall = _cli("recall", str(ws), "--topic", QUERY, "--limit", "20")
    assert r_recall.exit_code == 0, _cli_text(r_recall)
    recalled = {
        r["id"]: r["relevance"]
        for r in json.loads(r_recall.output)["results"]
        if r.get("source") == "experience"
    }

    r_mem = _cli("memories", str(ws), "--text", QUERY, "--limit", "20")
    assert r_mem.exit_code == 0, _cli_text(r_mem)
    reported = {e["id"]: e["relevance"] for e in json.loads(r_mem.output)["experiences"]}

    shared = set(recalled) & set(reported)
    assert shared, "no experience id appeared on both CLI paths"
    diverged = {
        i: (recalled[i], reported[i])
        for i in shared
        if abs(recalled[i] - reported[i]) > SAME_SCALE_TOL
    }
    assert not diverged, f"same store, same query, two scales in the CLI: {diverged}"


def test_f3_cli_memories_is_not_pure_decay_for_a_text_query(tmp_path: Path):
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    asyncio.run(_seed(db))
    decay = asyncio.run(_pure_decay(db))

    r_mem = _cli("memories", str(ws), "--text", QUERY, "--limit", "20")
    assert r_mem.exit_code == 0, _cli_text(r_mem)
    reported = {e["id"]: e["relevance"] for e in json.loads(r_mem.output)["experiences"]}
    assert reported
    off_decay = [i for i in reported if abs(reported[i] - decay[i]) > 1e-3]
    assert off_decay, f"CLI memories reports pure decay for a TEXT query: {reported}"


def _decay_scale_reports(source: str) -> list[int]:
    """Line numbers of `round(<something>.relevance(...), n)` CALLS.

    Parsed from the AST, not grepped: a regex over raw lines also grades
    COMMENTS that quote the old pattern, which is a check failing for a reason
    that has nothing to do with the defect. The AST sees code only.
    """
    tree = ast.parse(source)
    hits: list[int] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "round" or not node.args:
            continue
        inner = node.args[0]
        if (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "relevance"
        ):
            hits.append(node.lineno)
    return hits


def _called_names(source: str) -> set[str]:
    """Names that are actually CALLED in the source (not merely mentioned)."""
    tree = ast.parse(source)
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


# The ONE place the fallback round() is allowed to live. Everything else in the
# package that reports an experience relevance has to go through it.
_REPORTING_RULE_DEFINITION = ("models/experience.py", "reported_relevance")


def _rounds_outside(source: str, allowed_function: str | None) -> list[int]:
    """`_decay_scale_reports`, minus the hits inside one named function.

    An exemption by NAME, not by file, because the previous census was scoped
    to a file LIST and therefore could not see the site it missed.
    """
    hits = set(_decay_scale_reports(source))
    if allowed_function is None:
        return sorted(hits)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == allowed_function:
            hits -= {
                n.lineno
                for n in ast.walk(node)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "round"
                and n.args
                and isinstance(n.args[0], ast.Call)
                and isinstance(n.args[0].func, ast.Attribute)
                and n.args[0].func.attr == "relevance"
            }
    return sorted(hits)


def test_f3_census_no_decay_scale_report_left_anywhere():
    """Every experience-relevance report in the WHOLE package must go through
    the one shared rule, not a local `round(e.relevance(), 4)`.

    Widened from a two-file list after a review found `Experience.to_response`
    still emitting `round(self.relevance(), 3)`: a census scoped to a file list
    cannot enumerate the site it is missing, which is the N-1-of-N shape one
    layer up from the defect this file exists to close.
    """
    # Positive control on the detector itself: a detector that finds nothing
    # would pass the assertion below for free.
    control = (
        "def f(e, exp, now):\n"
        "    a = round(e.relevance(), 4)\n"
        "    b = round(exp.relevance(at=now), 4)\n"
    )
    assert len(_decay_scale_reports(control)) == 2, "the detector does not detect"
    assert "_reported_relevance" not in _called_names(control)

    exempt_file, exempt_fn = _REPORTING_RULE_DEFINITION
    offenders: list[str] = []
    scanned = 0
    for path in sorted(SRC_KAIRN.rglob("*.py")):
        rel = path.relative_to(SRC_KAIRN).as_posix()
        scanned += 1
        allowed = exempt_fn if rel == exempt_file else None
        offenders += [f"{rel}:{ln}" for ln in _rounds_outside(path.read_text(), allowed)]
    # Non-vacuity: a broken glob would scan nothing and pass.
    assert scanned > 10, f"only {scanned} files scanned - the sweep is broken"
    assert not offenders, f"decay-scale relevance reports still present at: {offenders}"

    # And the exemption is REAL, not a hole: the definition site does contain
    # the pattern, so a rename of that function would surface it as an offender.
    definition = (SRC_KAIRN / exempt_file).read_text()
    assert _rounds_outside(definition, None), (
        f"{exempt_file} no longer contains the fallback round() - "
        "the exemption is now a hole that hides nothing and would hide a new copy"
    )

    for name in ("server.py", "cli.py"):
        source = (SRC_KAIRN / name).read_text()
        assert "_reported_relevance" in _called_names(source), (
            f"{name} never CALLS the shared reported-relevance helper"
        )


# ── F6: an out-of-range floor must be LOUD ───────────────────────────

OUT_OF_RANGE = ["65", "-0.1", "1.5", "42"]
IN_RANGE = ["0.0", "0.5", "1.0"]


@pytest.mark.parametrize("bad", OUT_OF_RANGE)
async def test_f6_server_rejects_out_of_range(tmp_path: Path, bad: str):
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    await _seed(db)
    _write_config(ws, bad)

    server = create_server(str(db))
    # Broad on purpose: the MCP transport wraps the ValueError in a ToolError.
    with pytest.raises(Exception) as excinfo:
        async with Client(server) as client:
            await client.call_tool("kn_recall", {"topic": QUERY, "limit": 10})
    message = str(excinfo.value)
    assert "experience_min_match" in message, message
    # Assert on the ECHOED value, not a digit. A digit assertion is satisfied by
    # the static hint text ("[0.0, 1.0]", "0.65 for 65%"), which made this check
    # vacuous for 3 of the 4 parameters - a check that cannot fail.
    echo = f"got {float(bad)!r}"
    assert echo in message, f"the message does not echo the offending value: {message}"
    # Negative control: a DIFFERENT value must not appear as the echo.
    assert f"got {float(bad) + 7.0!r}" not in message


@pytest.mark.parametrize("good", IN_RANGE)
async def test_f6_server_accepts_in_range(tmp_path: Path, good: str):
    """Positive control: the guard must NOT be a catch-all."""
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    await _seed(db)
    _write_config(ws, good)

    server = create_server(str(db))
    async with Client(server) as client:
        payload = json.loads(
            (await client.call_tool("kn_recall", {"topic": QUERY, "limit": 10})).content[0].text
        )
    assert payload["_v"] == "1.0"


async def test_f6_no_config_file_still_works(tmp_path: Path):
    """Positive control: the default path is untouched."""
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    await _seed(db)
    server = create_server(str(db))
    async with Client(server) as client:
        payload = json.loads(
            (await client.call_tool("kn_recall", {"topic": QUERY, "limit": 10})).content[0].text
        )
    assert payload["count"] >= 1


async def test_f6_invalid_value_does_not_fail_closed_into_empty_results(tmp_path: Path):
    """The UNKNOWN-wearing-an-answer's-clothes case, stated directly: an
    invalid floor must NOT come back as a well-formed envelope with count 0."""
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    await _seed(db)
    _write_config(ws, "65")

    server = create_server(str(db))
    silent_empty = None
    try:
        async with Client(server) as client:
            payload = json.loads(
                (await client.call_tool("kn_recall", {"topic": QUERY, "limit": 10})).content[0].text
            )
        silent_empty = payload
    except Exception:
        silent_empty = None
    assert silent_empty is None, (
        "an invalid config produced a normal-looking answer instead of an error: "
        f"{silent_empty}"
    )


@pytest.mark.parametrize("bad", OUT_OF_RANGE)
def test_f6_cli_recall_rejects_out_of_range(tmp_path: Path, bad: str):
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    asyncio.run(_seed(db))
    _write_config(ws, bad)

    result = _cli("recall", str(ws), "--topic", QUERY)
    assert result.exit_code != 0, result.output
    assert "experience_min_match" in _cli_text(result)


@pytest.mark.parametrize("bad", OUT_OF_RANGE)
def test_f6_cli_memories_rejects_out_of_range(tmp_path: Path, bad: str):
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    asyncio.run(_seed(db))
    _write_config(ws, bad)

    result = _cli("memories", str(ws), "--text", QUERY)
    assert result.exit_code != 0, result.output
    assert "experience_min_match" in _cli_text(result)


def test_f6_cli_demo_rejects_out_of_range(tmp_path: Path):
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    asyncio.run(_seed(db))
    _write_config(ws, "65")

    result = _cli("demo", str(ws))
    assert result.exit_code != 0, result.output
    assert "experience_min_match" in _cli_text(result)


@pytest.mark.parametrize("good", IN_RANGE)
def test_f6_cli_accepts_in_range(tmp_path: Path, good: str):
    """Positive control for the CLI guard."""
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    asyncio.run(_seed(db))
    _write_config(ws, good)

    result = _cli("recall", str(ws), "--topic", QUERY)
    assert result.exit_code == 0, _cli_text(result)
    assert json.loads(result.output)["_v"] == "1.0"


def _function_body_dump(path: Path, name: str) -> str:
    """AST dump of a function's body with the docstring dropped.

    The two `_validate_experience_min_match` copies carry different docstrings
    on purpose (each explains its own site); everything they DO must stay
    identical, and a duplicated guard that silently drifts is the exact
    N-1-of-N shape this whole review is about.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = list(node.body)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            return "\n".join(ast.dump(stmt) for stmt in body)
    raise AssertionError(f"{name} not found in {path}")


def test_f6_the_two_validator_copies_have_not_drifted():
    """`experience_min_match` is range-checked in server.py AND cli.py, because
    importing one from the other costs a measured +358 ms on every CLI call.
    Duplication is only safe while the copies agree, so pin them."""
    server_body = _function_body_dump(SRC_KAIRN / "server.py", "_validate_experience_min_match")
    cli_body = _function_body_dump(SRC_KAIRN / "cli.py", "_validate_experience_min_match")
    # Non-vacuity: an empty body would make the equality trivially true.
    assert len(server_body) > 200, "validator body looks empty - the extractor is broken"
    assert server_body == cli_body, (
        "the two _validate_experience_min_match copies have drifted apart"
    )


# ── F3b: the abstention floor is a workspace policy, so it binds every surface

async def test_floor_binds_kn_memories_not_just_recall(tmp_path: Path):
    """Found by the code review on this change. `experience_min_match` reached
    recall / crossref / context and NOT kn_memories, so one store and one query
    gave "abstain" on one surface and a full result set on another - the same
    N-1-of-N shape as the missing constructor."""
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    await _seed(db)
    _write_config(ws, "0.99")

    store, intel = await _build_intel(db, experience_min_match=0.99)
    try:
        rows = await intel.recall(topic=QUERY, limit=20)
        recall_n = len([r for r in rows if r.get("source") == "experience"])
    finally:
        await store.close()

    server = create_server(str(db))
    async with Client(server) as client:
        payload = json.loads(
            (await client.call_tool("kn_memories", {"text": QUERY, "limit": 20})).content[0].text
        )
    assert payload["count"] == recall_n, (
        f"floor 0.99: recall kept {recall_n} experiences, kn_memories kept {payload['count']}"
    )
    assert recall_n == 0, "fixture no longer exercises abstention"


def test_floor_binds_cli_memories_not_just_cli_recall(tmp_path: Path):
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    asyncio.run(_seed(db))
    _write_config(ws, "0.99")

    r_recall = _cli("recall", str(ws), "--topic", QUERY, "--limit", "20")
    assert r_recall.exit_code == 0, _cli_text(r_recall)
    recall_n = len(
        [r for r in json.loads(r_recall.output)["results"] if r.get("source") == "experience"]
    )
    r_mem = _cli("memories", str(ws), "--text", QUERY, "--limit", "20")
    assert r_mem.exit_code == 0, _cli_text(r_mem)
    assert json.loads(r_mem.output)["count"] == recall_n
    assert recall_n == 0, "fixture no longer exercises abstention"


async def test_zero_floor_leaves_kn_memories_unchanged(tmp_path: Path):
    """Positive control: the default floor must not filter anything, or the
    test above would pass for the wrong reason."""
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    await _seed(db)
    _write_config(ws, "0.0")

    server = create_server(str(db))
    async with Client(server) as client:
        payload = json.loads(
            (await client.call_tool("kn_memories", {"text": QUERY, "limit": 20})).content[0].text
        )
    assert payload["count"] >= 3, f"the default floor filtered rows: {payload['count']}"


# ── F7: every construction site carries the floor ────────────────────

def _intelligence_layer_calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else None
        )
        if name == "IntelligenceLayer":
            found.append(node)
    return found


def _kwarg_names(call: ast.Call) -> set[str]:
    return {kw.arg for kw in call.keywords if kw.arg}


def _kwarg_value(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _assigned_from_validator(path: Path, varname: str) -> bool:
    """Is `varname` assigned from _validate_experience_min_match(...) here?"""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if varname not in targets:
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "_validate_experience_min_match"
        ):
            return True
    return False


REPO_ROOT = Path(__file__).resolve().parents[1]

# Construction sites that deliberately carry NO floor, each with its reason.
# An entry here is a disclosure, not a pass: the test below fails if an
# allowlisted path stops existing or stops being unwired, so the exemption
# cannot quietly rot into a lie.
KNOWN_UNWIRED = {
    # Standalone documentation script: reads no config at all, ships no
    # workspace, and is outside the file scope of this change. Reported
    # upward rather than edited here.
    "examples/demo.py",
}


def test_f7_every_construction_wires_experience_min_match():
    """N of N across the WHOLE repo, not just src/. Scoping the census to the
    directory being edited is how a fourth site stays invisible."""
    census: list[tuple[str, int, ast.Call]] = []
    for root in (REPO_ROOT / "src", REPO_ROOT / "examples"):
        if not root.exists():
            continue
        for py in sorted(root.rglob("*.py")):
            for call in _intelligence_layer_calls(py):
                census.append((str(py.relative_to(REPO_ROOT)), call.lineno, call))

    # Non-vacuity: four known sites (server.py:_init, cli.py:demo,
    # cli.py:_build_intel_stack, examples/demo.py). If the detector stops
    # finding them, every assertion below becomes trivially true.
    assert len(census) >= 4, f"census found only {len(census)} construction sites: {census}"

    unwired = {f for f, _ln, call in census if "experience_min_match" not in _kwarg_names(call)}
    unexpected = sorted(unwired - KNOWN_UNWIRED)
    assert not unexpected, f"IntelligenceLayer built without the floor at: {unexpected}"

    stale = sorted(KNOWN_UNWIRED - unwired)
    assert not stale, f"allowlisted as unwired but no longer is (drop the entry): {stale}"

    # PROVENANCE, not just presence. `experience_min_match=config.experience_min_match`
    # carries the right kwarg NAME and reintroduces F6, because the value never
    # passed the validator. Require a plain local whose assignment in the same
    # module comes from _validate_experience_min_match(...).
    bad_provenance: list[str] = []
    for path, lineno, call in census:
        if path in KNOWN_UNWIRED:
            continue
        value = _kwarg_value(call, "experience_min_match")
        if not isinstance(value, ast.Name):
            bad_provenance.append(f"{path}:{lineno} (not a local name)")
            continue
        if not _assigned_from_validator(REPO_ROOT / path, value.id):
            bad_provenance.append(f"{path}:{lineno} ({value.id} never validated)")
    assert not bad_provenance, (
        "floor forwarded without passing the range check at: " + str(bad_provenance)
    )


def test_f7_demo_does_not_silently_proceed_on_an_unreadable_config(tmp_path: Path):
    """Consequence of wiring the demo to the config, stated as a contract.

    Before this change the demo ignored config.yaml entirely, so a broken one
    was invisible there. It now reads the same config as every other command,
    and must therefore FAIL rather than run the tutorial with a configuration
    it could not read - the same principle as F6, one level up. Only the exit
    code is pinned; the message shape is the CLI's shared concern.
    """
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    asyncio.run(_seed(db))
    (ws / "config.yaml").write_text("experience_min_match: [unclosed\n")

    result = _cli("demo", str(ws))
    assert result.exit_code != 0, "demo ran the tutorial on a config it could not read"
    # And it must SAY so, rather than dumping a traceback.
    text = _cli_text(result)
    assert "Error:" in text, f"no clean error line, only: {text[:400]}"
    assert "Traceback" not in text, f"demo tracebacked instead of reporting: {text[:400]}"


def _recording_layer(sink: list[dict]):
    def factory(**kwargs):
        sink.append(kwargs)
        return IntelligenceLayer(**kwargs)

    return factory


async def test_f7_server_construction_receives_the_config_floor(tmp_path: Path):
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    await _seed(db)
    _write_config(ws, "0.65")

    seen: list[dict] = []
    server = create_server(str(db))
    with patch("kairn.server.IntelligenceLayer", _recording_layer(seen)):
        async with Client(server) as client:
            await client.call_tool("kn_recall", {"topic": QUERY, "limit": 5})
    assert seen, "no IntelligenceLayer was constructed"
    assert seen[0].get("experience_min_match") == 0.65


def test_f7_cli_stack_construction_receives_the_config_floor(tmp_path: Path):
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    asyncio.run(_seed(db))
    _write_config(ws, "0.65")

    seen: list[dict] = []
    with patch("kairn.core.intelligence.IntelligenceLayer", _recording_layer(seen)):
        result = _cli("recall", str(ws), "--topic", QUERY)
    assert result.exit_code == 0, _cli_text(result)
    assert seen, "no IntelligenceLayer was constructed by the CLI"
    assert seen[0].get("experience_min_match") == 0.65


def test_f7_cli_demo_construction_receives_the_config_floor(tmp_path: Path):
    ws = _ws(tmp_path)
    db = ws / "kairn.db"
    asyncio.run(_seed(db))
    _write_config(ws, "0.65")

    seen: list[dict] = []
    with patch("kairn.core.intelligence.IntelligenceLayer", _recording_layer(seen)):
        result = _cli("demo", str(ws))
    assert result.exit_code == 0, _cli_text(result)
    assert seen, "the demo constructed no IntelligenceLayer"
    assert seen[0].get("experience_min_match") == 0.65
