"""The README and the benchmark command do what a new reader is told they do.

Each test runs the real CLI or reads the real command tree, so the docs cannot
drift away from either one without a failure here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import click
import pytest

from kairn.cli import main
from kairn.relevance import RELEVANCE_KINDS

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
BENCHMARKS = (ROOT / "BENCHMARKS.md").read_text(encoding="utf-8")


def _run(
    *args: str, env: dict[str, str] | None = None, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    full_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", **(env or {})}
    return subprocess.run(
        [sys.executable, "-m", "kairn.cli", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=full_env,
        cwd=cwd,
        check=False,
    )


def _ok(*args: str, **kwargs) -> str:
    proc = _run(*args, **kwargs)
    assert proc.returncode == 0, f"{args}\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    return proc.stdout


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "brain"
    _ok("init", str(workspace))
    return workspace


def _flat(text: str) -> str:
    """Rich wraps long lines at the terminal width; compare on single spaces."""
    return re.sub(r"\s+", " ", text)


# --- the CLI section lists every command ------------------------------------


def _leaf_commands(group: click.Group, prefix: str = "kairn") -> dict[str, click.Command]:
    leaves: dict[str, click.Command] = {}
    for name, command in group.commands.items():
        path = f"{prefix} {name}"
        if isinstance(command, click.Group):
            leaves.update(_leaf_commands(command, path))
        else:
            leaves[path] = command
    return leaves


LEAVES = _leaf_commands(main)


def _cli_lines() -> list[str]:
    section = README.split("\n## CLI\n", 1)[1].split("\n## ", 1)[0]
    block = re.search(r"```bash\n(.*?)```", section, re.DOTALL)
    assert block, "the README's CLI section has no bash block"
    return [line for line in block.group(1).splitlines() if line.startswith("kairn ")]


def _lines_for(command_path: str) -> list[str]:
    return [
        line
        for line in _cli_lines()
        if line == command_path or line.startswith(command_path + " ")
    ]


def test_the_command_tree_is_not_empty() -> None:
    """An empty tree would make every test below pass while checking nothing."""
    assert "kairn learn" in LEAVES
    assert "kairn import git" in LEAVES
    assert len(LEAVES) >= 27


@pytest.mark.parametrize("command_path", sorted(LEAVES))
def test_every_command_is_listed(command_path: str) -> None:
    assert _lines_for(command_path), f"{command_path} is missing from the README's CLI section"


@pytest.mark.parametrize("command_path", sorted(LEAVES))
def test_required_flags_and_arguments_are_shown(command_path: str) -> None:
    shown = " ".join(_lines_for(command_path))
    for param in LEAVES[command_path].params:
        if not param.required:
            continue
        if isinstance(param, click.Option):
            flag = max(param.opts, key=len)
            assert flag in shown, f"{command_path} needs {flag}, the README does not show it"
        else:
            name = f"<{param.name}"
            assert name in shown and f"[{name}" not in shown, (
                f"{command_path} needs {name}>, the README does not show it as required"
            )


def test_every_listed_command_exists() -> None:
    for line in _cli_lines():
        words = line.split()
        assert any(
            words[: len(path.split())] == path.split() for path in LEAVES
        ), f"the README lists a command that does not exist: {line}"


def test_the_lines_a_new_reader_needs_are_exact() -> None:
    """Optional in click, but the command does nothing useful without them."""
    lines = _cli_lines()
    assert any(line.startswith("kairn serve --init") for line in lines)
    assert any(line.startswith("kairn recall <path> --topic") for line in lines)
    learn = next(line for line in lines if line.startswith("kairn learn "))
    for kind in ("decision", "pattern", "solution", "workaround", "gotcha", "preference"):
        assert kind in learn, f"the learn line does not name the type {kind}"


# --- the recall example is a real run ----------------------------------------


def _readme_json_blocks() -> list[dict]:
    return [json.loads(raw) for raw in re.findall(r"```json\n(\{.*?\})\n```", README, re.DOTALL)
            if '"_v"' in raw]


def _recall_example() -> dict:
    examples = [block for block in _readme_json_blocks() if "results" in block]
    assert len(examples) == 1, "expected exactly one recall example in the README"
    return examples[0]


def test_the_recall_example_matches_a_real_run(tmp_path: Path) -> None:
    example = _recall_example()["results"]
    content = example[0]["description"]
    workspace = _workspace(tmp_path)
    _ok("learn", str(workspace), "--content", content, "--type", "decision")
    real = json.loads(
        _ok("recall", str(workspace), "--topic", "postgres analytics concurrent")
    )["results"]

    assert [row["source"] for row in example] == [row["source"] for row in real]
    for shown, measured in zip(example, real, strict=True):
        assert set(shown) == set(measured), f"fields differ for {shown['source']}"
        assert shown["relevance_kind"] == measured["relevance_kind"]
        # One stored fact: the score sits at or near SQLite's floor. Not an exact
        # number, because the experience row can round to 0.0 or to 1e-06.
        assert shown["relevance"] < 0.01
        assert measured["relevance"] < 0.01


def test_every_relevance_kind_is_explained() -> None:
    paragraph = README.split("**Read `relevance_kind`", 1)[1].split("\n\n", 1)[0]
    for kind in RELEVANCE_KINDS:
        assert f"`{kind}`" in paragraph, f"the README does not explain relevance_kind {kind}"


def test_the_small_store_score_is_explained() -> None:
    """A new reader's first recall shows a score near zero. Say why, and what changes it."""
    paragraph = README.split("**Read `relevance_kind`", 1)[1].split("\n\n", 1)[0]
    assert "1e-06" in paragraph
    assert "SQLite" in paragraph


# --- benchmark and demo stay out of the reader's real workspace --------------


def _digest(workspace: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(workspace.iterdir())
        if path.is_file()
    }


def test_benchmark_leaves_a_named_workspace_untouched(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    before_status = _ok("status", str(workspace))
    before = _digest(workspace)
    assert "kairn.db" in before, "the check would pass on an empty folder"

    out = _ok("benchmark", str(workspace), "--nodes", "20")

    assert _digest(workspace) == before
    assert _ok("status", str(workspace)) == before_status
    assert "Insert" in out
    assert "temporary" in _flat(out).lower()


def test_benchmark_needs_no_workspace_and_removes_its_own(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    env = {"TMPDIR": str(scratch), "TEMP": str(scratch), "TMP": str(scratch)}

    out = _ok("benchmark", "--nodes", "20", env=env, cwd=tmp_path)

    assert "Insert" in out
    assert list(scratch.iterdir()) == [], "the temporary workspace was left behind"
    assert not (tmp_path / "kairn.db").exists()


def test_benchmark_does_not_create_a_named_folder_that_is_missing(tmp_path: Path) -> None:
    missing = tmp_path / "not-a-workspace"
    _ok("benchmark", str(missing), "--nodes", "20")
    assert not missing.exists()


def test_docs_do_not_point_benchmark_or_demo_at_the_real_workspace() -> None:
    assert "kairn benchmark ~/brain" not in README
    assert "kairn demo ~/brain" not in README
    assert "kairn benchmark <workspace>" not in BENCHMARKS


def test_the_demo_warning_names_what_it_writes() -> None:
    lines = [line for line in README.splitlines() if "kairn demo" in line]
    assert lines, "the README no longer mentions kairn demo"
    assert all("sample data" in line for line in lines), lines
    assert "sample data" in _flat(_ok("demo", "--help"))


def test_the_demo_is_not_called_interactive() -> None:
    """It runs start to finish without asking anything."""
    assert "Interactive" not in " ".join(_lines_for("kairn demo"))
    assert "Interactive" not in _flat(_ok("demo", "--help"))
