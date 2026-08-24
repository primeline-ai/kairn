"""The version of Kairn is written in seven hand-maintained places.

A release bumps the root pyproject and publishes to PyPI. If the bundle is not
bumped with it, `mcpb pack` still succeeds and still validates: the bundle just
quietly advertises the old version and pins the old release, so whoever installs
"the current bundle" gets the previous one. Nothing about that failure is loud.

This test is that guard. It is deliberately data-driven off the real files, so
adding a fifth site means adding a line here rather than remembering to.
"""

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
BUNDLE = REPO / "packaging" / "mcpb"


def _toml_version(path: Path) -> str:
    for line in path.read_text().splitlines():
        m = re.match(r'^version\s*=\s*"([^"]+)"', line.strip())
        if m:
            return m.group(1)
    raise AssertionError(f"no version found in {path}")


def _sites() -> dict[str, str]:
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    bundle_toml = (BUNDLE / "pyproject.toml").read_text()
    pin = re.search(r'kairn-ai==([0-9][^"\s]*)', bundle_toml)
    assert pin, "bundle pyproject.toml must pin kairn-ai to an exact version"

    server_py = (REPO / "src" / "kairn" / "server.py").read_text()
    fastmcp_version = re.search(r'FastMCP\(\s*"kairn"\s*,\s*version="([^"]+)"', server_py)
    assert fastmcp_version, "could not read the version passed to FastMCP"

    init_py = (REPO / "src" / "kairn" / "__init__.py").read_text()
    dunder = re.search(r'__version__\s*=\s*"([^"]+)"', init_py)
    assert dunder, "could not read __version__ from src/kairn/__init__.py"

    # uv.lock pins kairn-ai independently. After a bump the shipped lock is
    # stale, and `uv run` then silently re-resolves at the user's first launch -
    # needing network and write access inside the extension directory - instead
    # of using the locked hashes.
    lock = (BUNDLE / "uv.lock").read_text()
    locked = re.search(r'name = "kairn-ai"\nversion = "([^"]+)"', lock)
    assert locked, "could not read the kairn-ai pin from uv.lock"

    server_json = json.loads((REPO / "server.json").read_text())
    reg_pkg = server_json["packages"][0]

    return {
        "root pyproject.toml": _toml_version(REPO / "pyproject.toml"),
        "server.json": server_json["version"],
        "server.json package": reg_pkg["version"],
        "src/kairn/__init__.py": dunder.group(1),
        "bundle uv.lock pin": locked.group(1),
        "bundle pyproject.toml": _toml_version(BUNDLE / "pyproject.toml"),
        "bundle kairn-ai pin": pin.group(1),
        "bundle manifest.json": manifest["version"],
        "src/kairn/server.py FastMCP": fastmcp_version.group(1),
    }


def test_all_version_sites_agree():
    sites = _sites()
    distinct = set(sites.values())
    assert len(distinct) == 1, (
        "version drift across release sites:\n"
        + "\n".join(f"  {k:32s} {v}" for k, v in sites.items())
    )


def test_the_guard_notices_a_broken_version_line(tmp_path):
    """A regex that silently stops matching would make the guard vacuous.

    The previous version of this test asserted `_sites()[site]` is truthy, which
    can never fail: `_sites()` already raises on an unreadable site, and it is
    called at collection time. This one feeds a deliberately broken file to the
    same parser and asserts it complains.
    """
    broken = tmp_path / "pyproject.toml"
    broken.write_text("[project]\nname = 'x'\n# version line deleted\n")
    with pytest.raises(AssertionError, match="no version found"):
        _toml_version(broken)


def test_every_known_site_is_actually_found():
    """Names the sites explicitly, so silently dropping one fails here."""
    expected = {
        "root pyproject.toml",
        "src/kairn/__init__.py",
        "bundle uv.lock pin",
        "bundle pyproject.toml",
        "bundle kairn-ai pin",
        "bundle manifest.json",
        "src/kairn/server.py FastMCP",
        "server.json",
        "server.json package",
    }
    assert set(_sites()) == expected


# --- MCP registry contract ------------------------------------------------


def test_registry_asset_url_names_the_same_version():
    """server.json points at one exact release asset.

    After a version bump the URL still resolves - to the OLD bundle - so the
    entry would advertise a new version while shipping the previous artifact.
    Nothing else notices, because the file downloads fine.
    """
    pkg = json.loads((REPO / "server.json").read_text())["packages"][0]
    version = json.loads((REPO / "server.json").read_text())["version"]
    assert f"/v{version}/" in pkg["identifier"], pkg["identifier"]
    assert pkg["identifier"].endswith(f"kairn-{version}.mcpb"), pkg["identifier"]


def test_readme_carries_the_registry_ownership_token():
    """The PyPI validator reads this token out of the PUBLISHED README.

    It has to survive into the package description, sit on its own line, and
    name exactly the server in server.json. If someone rewrites the README
    intro, the next PyPI release silently stops qualifying and the failure only
    shows up at publish time.
    """
    name = json.loads((REPO / "server.json").read_text())["name"]
    lines = (REPO / "README.md").read_text().splitlines()
    assert f"mcp-name: {name}" in lines, (
        f"README must contain a line exactly 'mcp-name: {name}'"
    )
