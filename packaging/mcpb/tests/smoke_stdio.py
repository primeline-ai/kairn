"""Start a packed bundle and drive one real MCP handshake over stdio.

Not a pytest module on purpose: it takes the unpacked bundle path as an argument
and is invoked by CI after `mcpb pack`, so what it exercises is the artifact that
would be distributed rather than the source tree it was built from.

Exit 0 only if the server reports its own name and returns a non-empty tool list.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

TIMEOUT_S = 300


def _read_message(proc: subprocess.Popen[str], deadline: float) -> dict | None:
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                return None
            continue
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                # FastMCP writes diagnostics to stderr, but never assume a
                # stray stdout line is fatal - skip it and keep reading.
                continue
    return None


def main(bundle_dir: str) -> int:
    path = Path(bundle_dir).resolve()
    proc = subprocess.Popen(
        ["uv", "run", "--directory", str(path), "src/server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    deadline = time.time() + TIMEOUT_S

    def send(payload: dict) -> None:
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()

    try:
        send({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "ci-smoke", "version": "1"},
            },
        })
        reply = _read_message(proc, deadline)
        if reply is None:
            print("no initialize reply", file=sys.stderr)
            print(proc.stderr.read()[-4000:], file=sys.stderr)
            return 1
        info = reply.get("result", {}).get("serverInfo", {})
        if info.get("name") != "kairn":
            print(f"unexpected serverInfo: {info}", file=sys.stderr)
            return 1

        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        listed = _read_message(proc, deadline) or {}
        tools = listed.get("result", {}).get("tools", [])
        if not tools:
            print("tools/list returned nothing", file=sys.stderr)
            print(proc.stderr.read()[-4000:], file=sys.stderr)
            return 1

        print(f"bundle OK: {info}, {len(tools)} tools")
        return 0
    finally:
        proc.kill()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: smoke_stdio.py <unpacked-bundle-dir>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
