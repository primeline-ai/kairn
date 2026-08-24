"""The version of Kairn is written in four hand-maintained places.

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

    return {
        "root pyproject.toml": _toml_version(REPO / "pyproject.toml"),
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


@pytest.mark.parametrize("site", list(_sites()))
def test_each_site_is_readable(site):
    """A regex that stops matching would make the test above vacuously pass."""
    assert _sites()[site], f"{site} resolved to an empty version"
