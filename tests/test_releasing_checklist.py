"""The release checklist keeps its quickstart check.

The check in RELEASING.md is the one step that walks the README the way a new
reader does. These tests fail if the section is removed, if it loses the two
kinds of defect it reports, its filter or its length bound, or if the README
stops pointing at the checklist.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RELEASING = ROOT / "RELEASING.md"
README = ROOT / "README.md"


def _text(path: Path) -> str:
    assert path.is_file(), f"{path.name} is missing at the repo root"
    return path.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    assert heading in lines, f"RELEASING.md has no heading line '{heading}'"
    start = lines.index(heading) + 1
    end = next(
        (i for i in range(start, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    return "\n".join(lines[start:end])


def test_the_quickstart_check_names_what_it_reports() -> None:
    section = _section(_text(RELEASING), "## Quickstart check")
    for phrase in ("success moment", "doc vs code", "output vs effect"):
        assert phrase in section, f"the quickstart check no longer mentions '{phrase}'"


def test_the_quickstart_check_stays_on_task_and_short() -> None:
    section = _section(_text(RELEASING), "## Quickstart check")
    assert "Leave out everything else" in section, (
        "the quickstart check lost the line that drops off-task findings"
    )
    assert "one and a half times" in section, "the quickstart check lost its length bound"


def test_the_tag_waits_for_the_quickstart_check() -> None:
    section = " ".join(_section(_text(RELEASING), "## Before the tag").split())
    assert "Run the quickstart check below on the commit you are about to tag" in section, (
        "'Before the tag' no longer tells the releaser to run the quickstart check"
    )
    assert "before the tag is pushed" in section, (
        "'Before the tag' no longer says the defects are handled before the tag"
    )


def test_the_readme_links_the_release_checklist() -> None:
    assert re.search(r"\]\((?:\./)?RELEASING\.md\)", _text(README)), (
        "README.md has no link to RELEASING.md"
    )
