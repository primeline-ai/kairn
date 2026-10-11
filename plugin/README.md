# Kairn plugin

Kairn gives your agent a local knowledge graph with full-text recall and a
skill for recalling project knowledge before acting and saving useful
decisions, lessons, and fixes afterwards. Saving and recalling require no
LLM call.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first,
and make sure its `uvx` command is on your PATH. The client runs
`uvx --from kairn-ai==0.3.1 kairn serve --init` over stdio, opening `~/.kairn`.
First start downloads the package and creates an empty local store if needed.
Knowledge, experiences, and their graph relationships are stored in
`~/.kairn`. The store itself stays on this machine. Recalled knowledge is
returned to your agent and becomes part of its conversation with the model.

First start: a cold start downloads the package. Claude Code gives an MCP
server 30 seconds to start. Run
`uvx --from kairn-ai==0.3.1 kairn --help` once in a terminal.
It downloads and caches the package, then exits. Alternatively, start Claude
Code with `MCP_TIMEOUT=120000` to raise the limit to 120 seconds.
`MCP_TIMEOUT` is measured in milliseconds; see
[Claude Code's MCP guide](https://code.claude.com/docs/en/mcp-quickstart).

Desktop apps read PATH when they start. [Claude Desktop's Code tab](https://code.claude.com/docs/en/desktop)
on macOS takes it from your shell profile; on Windows, from the user and system
environment variables. After installing uv, quit and reopen the app. If a
desktop app still cannot start the server, it does not see `uvx`. For Codex,
remove the plugin with `codex plugin remove kairn@kairn`, find the full path
with `command -v uvx` (macOS and Linux) or `where.exe uvx` (Windows), and add
Kairn by hand with
`codex mcp add kairn -- /full/path/to/uvx --from kairn-ai==0.3.1 kairn serve --init`.
Set `startup_timeout_sec = 120` under `[mcp_servers.kairn]` in
`~/.codex/config.toml`, because the first start downloads the package.

The local MCP server works in Claude Code, Cowork on your computer, Codex CLI,
and Codex desktop. It does not load in claude.ai chat on the web.

## Install

Claude Code:

```sh
claude plugin marketplace add primeline-ai/kairn
claude plugin install kairn@kairn
```

Codex:

```sh
codex plugin marketplace add primeline-ai/kairn
codex plugin add kairn@kairn
```

## Existing stores

If you already use another folder, such as `~/brain` from the main README,
the plugin starts a new empty store in `~/.kairn`.
It does not migrate your existing knowledge.
`KAIRN_WORKSPACE` does not move the plugin's store;
it only changes where config.yaml is read from.
For a lasting setup with another folder, skip the plugin and
add Kairn by hand:
`claude mcp add kairn -- uvx --from kairn-ai==0.3.1 kairn serve --init ~/brain`
(Codex: `codex mcp add kairn -- uvx --from kairn-ai==0.3.1 kairn serve --init ~/brain`).

Codex keys plugin MCP servers by bare name, so a hand-added `kairn` server
and the plugin's `kairn` server collapse into one entry. Claude Code shows
both as `kairn` and `plugin:kairn:kairn`, each on its own store. Remove the
hand-added entry with `claude mcp remove kairn` or `codex mcp remove kairn`,
or skip the plugin.

## Non-interactive Codex

Non-interactive `codex exec` does not ask for approval, so plugin tools need
approval in `config.toml` first. Enable the server and use per-tool approval
for the tools the run needs. This example approves all five tools named by
the bundled kairn-memory skill: `kn_recall`, `kn_context`, `kn_memories` and
`kn_status` for reading, and `kn_learn` for saving:

```toml
[plugins."kairn@kairn".mcp_servers.kairn]
enabled = true

[plugins."kairn@kairn".mcp_servers.kairn.tools.kn_recall]
approval_mode = "approve"

[plugins."kairn@kairn".mcp_servers.kairn.tools.kn_context]
approval_mode = "approve"

[plugins."kairn@kairn".mcp_servers.kairn.tools.kn_memories]
approval_mode = "approve"

[plugins."kairn@kairn".mcp_servers.kairn.tools.kn_status]
approval_mode = "approve"

[plugins."kairn@kairn".mcp_servers.kairn.tools.kn_learn]
approval_mode = "approve"
```

With per-tool approval tables for the tools a run needs, `codex exec` calls
them. A tool without its own table falls back to the server's default approval.
With no approval configured, `codex exec` refuses plugin tools with
`MCP tool call requires approval, but approval policy is never`.
Tools that write or delete knowledge, such as `kn_remove` and `kn_prune`,
stay refused when left out of this list and the server default does not
approve them.

As a shorter alternative, approve the entire server:

```toml
[plugins."kairn@kairn".mcp_servers.kairn]
enabled = true
default_tools_approval_mode = "approve"
```

This auto-approves every tool, including tools that write or delete knowledge.
