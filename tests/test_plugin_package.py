"""The Claude Code / Codex plugin in plugin/ and the root marketplace file.

These run in the normal suite. `claude plugin validate --strict` runs in its own
CI job because it needs Node.
"""

from __future__ import annotations

import json
import re
import struct
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "plugin"
MARKETPLACE = ROOT / ".claude-plugin" / "marketplace.json"
MAX_FILE_BYTES = 256 * 1024
# The directory listing icon, named by `icon` in plugin.json. The portal's
# rules: a square PNG or JPEG, 512 to 2048 px per side, under 2 MB. The
# binding size limit is MAX_FILE_BYTES above (256 KiB, every plugin file).
# It is the plugin's only binary file, so it is checked here instead of by
# the text-based privacy scan, and it may carry no metadata chunks.
ICON = PLUGIN / ".claude-plugin" / "icon.png"
ICON_MIN_PX, ICON_MAX_PX, ICON_MAX_BYTES = 512, 2048, 2 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_ALLOWED_CHUNKS = {b"IHDR", b"PLTE", b"IDAT", b"IEND"}
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
        ".claude-plugin/icon.png",
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



def test_listing_links_are_https_and_agree_across_manifests() -> None:
    # The directory reads documentationUrl and supportUrl from plugin.json;
    # without documentationUrl it takes the README's first link instead.
    claude = _json(PLUGIN / ".claude-plugin" / "plugin.json")
    codex = _json(PLUGIN / ".codex-plugin" / "plugin.json")
    for field in ("homepage", "documentationUrl", "supportUrl", "repository"):
        assert claude[field].startswith("https://"), (field, claude[field])
    assert claude["homepage"].rstrip("/") != claude["repository"].rstrip("/"), "homepage is the repo"
    assert claude["supportUrl"].startswith(claude["repository"].rstrip("/") + "/"), claude["supportUrl"]
    # Codex lists the website from interface.websiteURL; keep both manifests on one homepage.
    assert codex["homepage"] == claude["homepage"]
    assert codex["interface"]["websiteURL"] == claude["homepage"]
    assert codex["repository"] == claude["repository"]

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
    # The icon is the one binary file; every other plugin file is read as text.
    return [*(p for p in _plugin_files() if p != ICON), MARKETPLACE, ROOT / "README.md"]


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



def _png_chunks(data: bytes) -> list[bytes] | None:
    """The chunk types of a PNG, or None if the chunk structure is broken."""
    chunks, pos = [], len(PNG_SIGNATURE)
    while pos + 12 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        end = pos + 12 + length
        if end > len(data):
            return None
        if zlib.crc32(data[pos + 4 : end - 4]) != struct.unpack(">I", data[end - 4 : end])[0]:
            return None
        chunks.append(kind)
        pos = end
        if kind == b"IEND":
            return chunks if pos == len(data) else None
    return None


def _icon_problems(data: bytes) -> list[str]:
    """What is wrong with a PNG for the directory icon."""
    if len(data) >= ICON_MAX_BYTES:
        return [f"{len(data)} bytes is not under 2 MB"]
    if not data.startswith(PNG_SIGNATURE) or len(data) < 33 or data[8:16] != b"\x00\x00\x00\rIHDR":
        return ["not a PNG"]
    chunks = _png_chunks(data)
    if chunks is None or chunks[0] != b"IHDR":
        return ["not a PNG"]
    width, height = struct.unpack(">II", data[16:24])
    problems = []
    if width != height:
        problems.append(f"not square: {width}x{height}")
    if not ICON_MIN_PX <= min(width, height) <= max(width, height) <= ICON_MAX_PX:
        problems.append(f"{width}x{height} is outside {ICON_MIN_PX}-{ICON_MAX_PX} px")
    extra = sorted({c.decode("latin-1") for c in chunks} - {c.decode() for c in PNG_ALLOWED_CHUNKS})
    if extra:
        problems.append(f"metadata chunks {extra}")
    return problems


def _chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def _png(width: int, height: int, *, extra: bytes = b"") -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return PNG_SIGNATURE + _chunk(b"IHDR", ihdr) + extra + _chunk(b"IDAT", b"") + _chunk(b"IEND", b"")


def test_listing_icon_is_a_square_png_in_range() -> None:
    assert _json(PLUGIN / ".claude-plugin" / "plugin.json")["icon"] == "./.claude-plugin/icon.png"
    assert _icon_problems(ICON.read_bytes()) == []


@pytest.mark.parametrize("px", [ICON_MIN_PX, 1024, ICON_MAX_PX])
def test_icon_check_accepts_a_valid_png(px: int) -> None:
    assert _icon_problems(_png(px, px)) == []


# Short ids: pytest puts the id into an environment variable, and Windows
# refuses one over 32,767 characters (pytest-dev/pytest#6881).
@pytest.mark.parametrize(
    ("make", "expected"),
    [
        (lambda: b"GIF89a" + bytes(32), "not a PNG"),
        (lambda: _png(1024, 1024)[:20], "not a PNG"),
        (lambda: _png(1024, 1024)[:-12], "not a PNG"),
        (lambda: _png(1024, 768), "not square"),
        (lambda: _png(ICON_MIN_PX - 1, ICON_MIN_PX - 1), "outside"),
        (lambda: _png(ICON_MAX_PX + 1, ICON_MAX_PX + 1), "outside"),
        (lambda: _png(1024, 1024, extra=_chunk(b"tEXt", b"Comment\x00/home/x")), "metadata"),
        (lambda: _png(1024, 1024) + bytes(ICON_MAX_BYTES - len(_png(1024, 1024))), "not under 2 MB"),
    ],
    ids=["gif", "truncated-header", "no-iend", "not-square", "too-small", "too-large", "text-chunk", "exactly-2mb"],
)
def test_icon_check_rejects_a_wrong_file(make, expected: str) -> None:
    problems = _icon_problems(make())
    assert any(expected in problem for problem in problems), problems
