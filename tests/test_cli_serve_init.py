"""Tests for `kairn serve --init`.

A client that launches `kairn serve <path>` on a machine where nobody ran
`kairn init` used to get a server that exits at once. `--init` creates the
workspace on first start instead. The plain `serve` keeps refusing an empty
workspace, and says how to fix it.

stdout is the MCP transport, so the server is driven over real stdio here and
every stdout line must be JSON: a stray status line from the init path would
break the client's handshake.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

TIMEOUT_S = 60


def _run_kairn(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "kairn.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=TIMEOUT_S,
    )


def _pump(stream, sink) -> None:
    for line in iter(stream.readline, ""):
        sink(line)
    stream.close()


class _StdioServer:
    """Drive `kairn serve` over stdio with a real deadline (readline blocks)."""

    def __init__(self, *args: str) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "kairn.cli", "serve", *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.stdout_lines: list[str] = []
        self.stderr_lines: list[str] = []
        self._q: queue.Queue[str] = queue.Queue()

        def to_stdout(line: str) -> None:
            self.stdout_lines.append(line)
            self._q.put(line)

        for stream, sink in (
            (self.proc.stdout, to_stdout),
            (self.proc.stderr, self.stderr_lines.append),
        ):
            threading.Thread(target=_pump, args=(stream, sink), daemon=True).start()
        self._next_id = 0

    def request(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        payload = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload)
        deadline = time.time() + TIMEOUT_S
        while time.time() < deadline:
            try:
                line = self._q.get(timeout=1.0).strip()
            except queue.Empty:
                if self.proc.poll() is not None and self._q.empty():
                    break
                continue
            if not line:
                continue
            message = json.loads(line)  # a non-JSON stdout line fails the test here
            if message.get("id") == self._next_id:
                return message
        raise AssertionError(
            f"no reply to {method!r}; exit={self.proc.poll()}; "
            f"stderr={''.join(self.stderr_lines[-50:])}"
        )

    def notify(self, method: str) -> None:
        self._send({"jsonrpc": "2.0", "method": method})

    def handshake(self) -> dict:
        reply = self.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "kairn-test", "version": "0"},
            },
        )
        self.notify("notifications/initialized")
        return reply

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)

    def _send(self, payload: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()


def _status(server: _StdioServer) -> dict:
    reply = server.request("tools/call", {"name": "kn_status", "arguments": {}})
    assert "error" not in reply, reply
    text = reply["result"]["content"][0]["text"]
    payload = json.loads(text)
    # FastMCP may wrap a str return value as {"result": "<json>"}.
    if isinstance(payload, dict) and isinstance(payload.get("result"), str):
        payload = json.loads(payload["result"])
    return payload


def test_serve_init_creates_missing_workspace_and_answers(tmp_path: Path) -> None:
    workspace = tmp_path / "fresh" / "nested"  # does not exist yet, like ~/.kairn on a new machine
    server = _StdioServer("--init", str(workspace))
    try:
        reply = server.handshake()
        assert reply["result"]["serverInfo"]["name"] == "kairn"

        tools = server.request("tools/list")["result"]["tools"]
        assert "kn_status" in {tool["name"] for tool in tools}

        status = _status(server)
        assert Path(status["db_path"]) == (workspace / "kairn.db").resolve()
        assert status["nodes"] == 0
    finally:
        server.close()

    assert (workspace / "kairn.db").is_file(), "serve --init answered but left no kairn.db"
    assert (workspace / "config.yaml").is_file(), "serve --init did not write config.yaml like init"
    for line in server.stdout_lines:
        if line.strip():
            json.loads(line)  # stdout carries the MCP transport and nothing else


def test_serve_init_keeps_an_existing_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    assert _run_kairn("init", str(workspace)).returncode == 0
    learned = _run_kairn(
        "learn",
        str(workspace),
        "--content",
        "Existing knowledge must survive serve --init",
        "--type",
        "decision",
        "--confidence",
        "high",
    )
    assert learned.returncode == 0, learned.stderr
    config_before = (workspace / "config.yaml").read_bytes()

    server = _StdioServer("--init", str(workspace))
    try:
        server.handshake()
        assert _status(server)["nodes"] == 1
    finally:
        server.close()

    assert (workspace / "config.yaml").read_bytes() == config_before


def test_serve_without_init_still_refuses_an_empty_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "empty"
    workspace.mkdir()

    result = _run_kairn("serve", str(workspace))

    assert result.returncode != 0
    assert result.stdout == "", "stdout is the MCP transport; the error belongs on stderr"
    assert "No database" in result.stderr
    assert "--init" in result.stderr, "the error should name the flag that fixes it"
    assert not (workspace / "kairn.db").exists(), "plain serve must not create a store"
