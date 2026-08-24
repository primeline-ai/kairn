"""Entry point for the Kairn MCP Bundle.

The bundle ships no Kairn source of its own. It depends on the published
`kairn-ai` release and starts that server, so a bundle install and a
`pip install kairn-ai` run the same code.

The workspace directory comes from KAIRN_WORKSPACE, which the host fills in
from the bundle's `user_config.workspace` field. It falls back to ~/.kairn,
matching the default the CLI prints in `kairn init`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _workspace() -> Path:
    raw = os.environ.get("KAIRN_WORKSPACE", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".kairn"


def main() -> int:
    workspace = _workspace()
    try:
        workspace.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # stdout is the MCP transport - diagnostics must go to stderr only.
        print(f"kairn: cannot create workspace {workspace}: {exc}", file=sys.stderr)
        return 1

    try:
        from kairn.server import create_server
    except ImportError as exc:
        print(f"kairn: kairn-ai is not importable: {exc}", file=sys.stderr)
        return 1

    server = create_server(str(workspace / "kairn.db"))
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
