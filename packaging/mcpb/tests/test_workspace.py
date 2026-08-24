"""Tests for the bundle entry point's workspace resolution.

The case that matters is an unsubstituted MCPB placeholder: it is silent, it
puts the user's database in the wrong place, and no existing test covered it.
"""

import importlib.util
import os
from pathlib import Path, PureWindowsPath

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "bundle_server", Path(__file__).resolve().parents[1] / "src" / "server.py"
)
bundle_server = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bundle_server)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("KAIRN_WORKSPACE", raising=False)


def test_unset_uses_default():
    assert bundle_server._workspace() == bundle_server.DEFAULT_WORKSPACE


def test_blank_and_whitespace_use_default(monkeypatch):
    for value in ("", "   ", "\t\n"):
        monkeypatch.setenv("KAIRN_WORKSPACE", value)
        assert bundle_server._workspace() == bundle_server.DEFAULT_WORKSPACE


@pytest.mark.parametrize(
    "literal",
    ["${HOME}/.kairn", "${user_config.workspace}", "/data/${HOME}/x"],
)
def test_unsubstituted_placeholder_falls_back(monkeypatch, literal):
    """The regression: these must NOT become a directory named '${HOME}'."""
    monkeypatch.setenv("KAIRN_WORKSPACE", literal)
    resolved = bundle_server._workspace()
    assert resolved == bundle_server.DEFAULT_WORKSPACE
    assert "${" not in str(resolved)


def test_tilde_is_expanded(monkeypatch):
    monkeypatch.setenv("KAIRN_WORKSPACE", "~/somewhere")
    assert bundle_server._workspace() == Path.home() / "somewhere"


def test_real_env_var_is_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("KAIRN_TEST_ROOT", str(tmp_path))
    monkeypatch.setenv("KAIRN_WORKSPACE", "$KAIRN_TEST_ROOT/ws")
    assert bundle_server._workspace() == tmp_path / "ws"


def test_absolute_path_passes_through(monkeypatch, tmp_path):
    monkeypatch.setenv("KAIRN_WORKSPACE", str(tmp_path / "explicit"))
    assert bundle_server._workspace() == tmp_path / "explicit"


# --- cases the external review found uncovered -------------------------------


@pytest.mark.parametrize("relative", [".kairn", "kairn/db", "sub/dir"])
def test_relative_value_does_not_land_in_the_bundle(monkeypatch, relative, capsys):
    """cwd is the unpacked bundle, so a relative value must NOT be joined to it.

    The manifest runs `uv run --directory ${__dirname}`. Resolving a relative
    value against cwd puts the user's database inside the extension directory,
    where they cannot find it and an upgrade discards it.
    """
    monkeypatch.setenv("KAIRN_WORKSPACE", relative)
    resolved = bundle_server._workspace()
    assert resolved == bundle_server.DEFAULT_WORKSPACE
    assert Path.cwd() not in resolved.parents
    assert "ignoring non-absolute" in capsys.readouterr().err


def test_empty_home_substitution_does_not_reach_the_drive_root(monkeypatch, capsys):
    """`${HOME}/.kairn` with HOME unset becomes `/.kairn`.

    Windows has no HOME by default, it has USERPROFILE. A host doing a naive
    replace produces `/.kairn`, which on Windows carries no drive: joining it to
    cwd yields C:\\.kairn, the drive root, not the user profile. The `${` guard
    cannot see this - substitution already happened.
    """
    monkeypatch.setenv("KAIRN_WORKSPACE", "/.kairn")
    resolved = bundle_server._workspace()
    if PureWindowsPath("/.kairn").is_absolute():  # pragma: no cover - POSIX host
        pytest.skip("platform treats a driveless root as absolute")
    if os.name == "nt":
        assert resolved == bundle_server.DEFAULT_WORKSPACE
        assert "ignoring non-absolute" in capsys.readouterr().err
    else:
        # On POSIX "/.kairn" is a genuine absolute path and is honoured.
        assert resolved == Path("/.kairn")


def test_windows_driveless_root_is_not_absolute():
    """The platform fact the test above rests on, asserted rather than assumed."""
    assert PureWindowsPath("/.kairn").is_absolute() is False
    assert PureWindowsPath("C:/Users/Alice/.kairn").is_absolute() is True
