"""Start the plugin's server exactly as a client does, and ask it one question.

`tests/test_plugin_package.py` checks what the manifests say; `claude plugin
details` in CI checks that the server is declared. Neither starts it, so a
launch line that exits at once, or a published release whose server does not
come up, would stay green. This test runs the command from `plugin/.mcp.json` with no shell,
an empty home and an empty uv cache, and calls `kn_status`.

It downloads the pinned release from PyPI, so it only runs when
KAIRN_PLUGIN_LAUNCH=1 (the `plugin` CI job sets it) and `uvx` is on PATH.
Right after a version bump the release is not on PyPI yet; that state skips
loudly, the same way the bundle's runtime smoke does. The release is published
from the release PR's head before that PR merges, so the run on the merged
commit starts the pinned release for real.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MCP_JSON = ROOT / "plugin" / ".mcp.json"
TIMEOUT_S = 120

pytestmark = pytest.mark.skipif(
    os.environ.get("KAIRN_PLUGIN_LAUNCH") != "1" or shutil.which("uvx") is None,
    reason="set KAIRN_PLUGIN_LAUNCH=1 with uvx on PATH to run (downloads from PyPI)",
)


def _pinned_version(args: list[str]) -> str:
    return args[1].split("==", 1)[1]


def _on_pypi(version: str) -> bool:
    # curl, not urllib: some Python installs ship without a CA bundle, and a
    # certificate error must not read as "not published".
    code = subprocess.run(
        ["curl", "-s", "-o", os.devnull, "-w", "%{http_code}",
         f"https://pypi.org/pypi/kairn-ai/{version}/json"],
        capture_output=True, text=True, timeout=60, check=False,
    ).stdout.strip()
    if code not in ("200", "404"):
        raise AssertionError(f"could not ask PyPI about kairn-ai {version} (HTTP {code!r})")
    return code == "200"


def test_the_plugin_launch_line_starts_a_server_that_answers(tmp_path: Path) -> None:
    server = json.loads(MCP_JSON.read_text(encoding="utf-8"))["mcpServers"]["kairn"]
    version = _pinned_version(server["args"])
    if not _on_pypi(version):
        pytest.skip(f"kairn-ai {version} is not on PyPI yet - skipped until published")

    home = tmp_path / "home"
    home.mkdir()
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "USERPROFILE": str(home),  # Windows expands ~ from this, never from HOME
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
    }
    # Windows needs these in a child process environment.
    for name in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATHEXT", "COMSPEC"):
        if name in os.environ:
            env[name] = os.environ[name]
    proc = subprocess.Popen(
        [shutil.which(server["command"]), *server["args"]],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        cwd=home,
    )
    lines: queue.Queue[str] = queue.Queue()
    stderr: list[str] = []
    for stream, sink in ((proc.stdout, lines.put), (proc.stderr, stderr.append)):
        threading.Thread(
            target=lambda s=stream, f=sink: [f(x) for x in iter(s.readline, "")], daemon=True
        ).start()

    def request(msg_id: int, method: str, params: dict | None = None) -> dict:
        payload = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params is not None:
            payload["params"] = params
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        deadline = time.time() + TIMEOUT_S
        while time.time() < deadline:
            try:
                line = lines.get(timeout=1).strip()
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            if line:
                reply = json.loads(line)  # a non-JSON stdout line fails here
                if reply.get("id") == msg_id:
                    return reply
        tail = "".join(stderr[-20:])
        raise AssertionError(f"no reply to {method}; exit={proc.poll()}; stderr={tail}")

    try:
        init = request(1, "initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "plugin-launch-test", "version": "0"},
        })
        assert init["result"]["serverInfo"]["version"] == version
        initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        proc.stdin.write(json.dumps(initialized) + "\n")
        proc.stdin.flush()
        reply = request(2, "tools/call", {"name": "kn_status", "arguments": {}})
        assert "error" not in reply, reply
        status = json.loads(reply["result"]["content"][0]["text"])
        if isinstance(status, dict) and isinstance(status.get("result"), str):
            status = json.loads(status["result"])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    assert Path(status["db_path"]).resolve() == (home / ".kairn" / "kairn.db").resolve()
