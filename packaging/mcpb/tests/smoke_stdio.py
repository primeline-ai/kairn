"""Start a packed bundle and drive one real MCP handshake over stdio.

Not a pytest module on purpose: it takes the unpacked bundle path as an argument
and runs after `mcpb pack`, so what it exercises is the artifact that would be
distributed rather than the source tree it was built from.

The I/O here is deliberately not the obvious version. `readline()` blocks, so a
deadline checked between reads cannot fire while a server sits with stdout open
and never writes - which is the most likely bundle failure (dependency
resolution stalls, or the server hangs in init) and would otherwise present as a
red job with no output at the workflow timeout. Reader threads make the deadline
real. Draining stderr concurrently matters for the same reason: a server logging
more than a pipe buffer to stderr deadlocks before it can answer.

Exit 0 only if the server names itself and returns a non-empty tool list.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

TIMEOUT_S = 300


def _pump(stream, sink) -> None:
    for line in iter(stream.readline, ""):
        sink(line)
    stream.close()


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

    stdout_q: queue.Queue[str] = queue.Queue()
    stderr_lines: list[str] = []
    for stream, sink in ((proc.stdout, stdout_q.put), (proc.stderr, stderr_lines.append)):
        threading.Thread(target=_pump, args=(stream, sink), daemon=True).start()

    deadline = time.time() + TIMEOUT_S

    def fail(reason: str) -> int:
        print(reason, file=sys.stderr)
        print("--- server stderr ---", file=sys.stderr)
        sys.stderr.write("".join(stderr_lines[-200:]))
        return 1

    def send(payload: dict) -> None:
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()

    def recv() -> dict | None:
        while time.time() < deadline:
            try:
                line = stdout_q.get(timeout=1.0).strip()
            except queue.Empty:
                if proc.poll() is not None and stdout_q.empty():
                    return None
                continue
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return None

    try:
        send({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "ci-smoke", "version": "1"},
            },
        })
        reply = recv()
        if reply is None:
            return fail("no initialize reply within the deadline")
        info = reply.get("result", {}).get("serverInfo", {})
        if info.get("name") != "kairn":
            return fail(f"unexpected serverInfo: {info}")

        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        listed = recv() or {}
        tools = listed.get("result", {}).get("tools", [])
        if not tools:
            return fail("tools/list returned nothing")

        print(f"bundle OK: {info}, {len(tools)} tools")
        return 0
    finally:
        proc.kill()
        proc.wait(timeout=10)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: smoke_stdio.py <unpacked-bundle-dir>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
