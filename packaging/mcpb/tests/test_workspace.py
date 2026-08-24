"""Tests for the bundle entry point's workspace resolution.

The case that matters is an unsubstituted MCPB placeholder: it is silent, it
puts the user's database in the wrong place, and no existing test covered it.
"""

import importlib.util
import os
from pathlib import Path

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
