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


DEFAULT_WORKSPACE = Path.home() / ".kairn"


def _workspace() -> Path:
    """Resolve the workspace directory, refusing an unsubstituted placeholder.

    The manifest sets KAIRN_WORKSPACE to ${user_config.workspace}, whose default
    is ${HOME}/.kairn. A host that does not perform that substitution hands over
    the literal string. Path.expanduser only expands "~", so "${HOME}/.kairn"
    would be created as a directory named "${HOME}" relative to wherever the
    server happened to start - putting the user's knowledge base somewhere they
    will never find it, and losing it on the next install. Treat any surviving
    ${...} as "not configured" and use the default instead.
    """
    raw = os.environ.get("KAIRN_WORKSPACE", "").strip()
    if not raw or "${" in raw:
        return DEFAULT_WORKSPACE
    return Path(os.path.expandvars(raw)).expanduser()


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
