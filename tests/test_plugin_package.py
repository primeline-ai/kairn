"""The Claude Code / Codex plugin in plugin/ and the root marketplace file.

These run in the normal suite. `claude plugin validate --strict` runs in its own
CI job because it needs Node.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "plugin"
MARKETPLACE = ROOT / ".claude-plugin" / "marketplace.json"
MAX_FILE_BYTES = 256 * 1024
# The two ids in the root README's example output come from a throwaway demo
# store. Any other 8-hex id in the README or the plugin is treated as a leak.
README_DEMO_IDS = {"002d9c22", "d0710c2f"}
NODE_ID = re.compile(r"(?<![0-9a-fA-F-])[0-9a-f]{8}(?![0-9a-fA-F-])")
# Patterns, not names: a test that spelled out a private host would publish it.
# Any subdomain of the project domain (the public site itself is the bare
# domain), and home folders on macOS, Linux and Windows.
PRIVATE_PATTERNS = (
    re.compile(r"\b[a-z0-9-]+\.primeline\.cc\b", re.IGNORECASE),
    re.compile(r"/Users/|/home/|[A-Za-z]:\\+Users\\", re.IGNORECASE),
)


def _plugin_files() -> list[Path]:
    return sorted(p for p in PLUGIN.rglob("*") if p.is_file() or p.is_symlink())


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _words_outside_code(markdown: str) -> int:
    prose = re.sub(r"```.*?```", " ", markdown, flags=re.DOTALL)
    prose = re.sub(r"`[^`]*`", " ", prose)
    return len(re.findall(r"[A-Za-z][A-Za-z'-]*", prose))


def test_plugin_layout() -> None:
    for rel in (
        ".claude-plugin/plugin.json",
        ".codex-plugin/plugin.json",
        ".mcp.json",
        ".mcp.codex.json",
        "skills/kairn-memory/SKILL.md",
        "README.md",
        "LICENSE",
    ):
        assert (PLUGIN / rel).is_file(), f"plugin/{rel} is missing"
    market = _json(MARKETPLACE)
    assert market["name"] == "kairn"
    entries = [p for p in market["plugins"] if p["name"] == "kairn"]
    assert len(entries) == 1 and entries[0]["source"] == "./plugin", market
    codex = _json(PLUGIN / ".codex-plugin" / "plugin.json")
    assert (PLUGIN / codex["mcpServers"]).resolve() == (PLUGIN / ".mcp.codex.json").resolve()


@pytest.mark.parametrize("mcp_file", [".mcp.json", ".mcp.codex.json"])
def test_server_starts_with_init_and_names_no_path(mcp_file: str) -> None:
    # The pinned version itself is checked with every other version site in
    # packaging/mcpb/tests/test_version_parity.py.
    server = _json(PLUGIN / mcp_file)["mcpServers"]["kairn"]
    assert server["command"] == "uvx"
    args = server["args"]
    assert args[0] == "--from" and args[1].startswith("kairn-ai=="), args
    # `kairn serve` defaults to ~/.kairn. The directory checklist blocks any
    # path in an MCP server command that does not start from
    # ${CLAUDE_PLUGIN_ROOT} when the plugin sits in a subfolder of its repo.
    assert args[2:] == ["kairn", "serve", "--init"], args
    for arg in [server["command"], *args]:
        assert not any(mark in arg for mark in ("/", "\\", "~", "$")), arg


def test_plugin_folder_hygiene() -> None:
    files = _plugin_files()
    assert files
    for path in files:
        rel = path.relative_to(ROOT)
        assert not path.is_symlink(), f"{rel} is a symlink"
        assert path.name != ".DS_Store", f"{rel} must not be shipped"
        assert path.stat().st_size < MAX_FILE_BYTES, f"{rel} is over 256 KiB"
    words = _words_outside_code((PLUGIN / "README.md").read_text(encoding="utf-8"))
    assert words >= 40, f"plugin/README.md has {words} words outside code, needs 40"


def _privacy_targets() -> list[Path]:
    return [*_plugin_files(), MARKETPLACE, ROOT / "README.md"]


def test_no_private_data_in_plugin_or_readme() -> None:
    for path in _privacy_targets():
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(ROOT)
        for pattern in PRIVATE_PATTERNS:
            found = pattern.search(text)
            assert not found, f"{rel} contains {found.group(0)!r}"
        allowed = README_DEMO_IDS if path == ROOT / "README.md" else set()
        ids = set(NODE_ID.findall(text)) - allowed
        assert not ids, f"{rel} contains what look like Kairn node ids: {sorted(ids)}"
