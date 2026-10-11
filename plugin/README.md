# Kairn plugin

Kairn gives your agent a local knowledge graph with full-text recall and a
skill for recalling project knowledge before acting and saving useful
decisions, lessons, and fixes afterwards. Saving and recalling require no
LLM call.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first,
and make sure its `uvx` command is on your PATH. The client runs
`uvx --from kairn-ai==0.3.0 kairn serve --init ~/.kairn` over stdio.
First start downloads the package and creates an empty local store if needed.
Knowledge, experiences, and their graph relationships are stored in
`~/.kairn`. Nothing in that store leaves the machine through this local
server; your agent receives tool results in its conversation.

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
the plugin starts a new empty store in `~/.kairn`. It does not migrate your
existing knowledge. Point the plugin at your folder only by editing the
path argument in both plugin MCP files: `.mcp.json` and `.mcp.codex.json`.

## Non-interactive Codex

Non-interactive `codex exec` refuses plugin tool calls until you enable and
approve the server in your Codex `config.toml`:

```toml
[plugins."kairn@kairn".mcp_servers.kairn]
enabled = true
default_tools_approval_mode = "approve"
```

This auto-approves every tool, including tools that write or delete knowledge.
For narrower approval, omit the blanket default approval and configure
individual tools using this form:

```toml
[plugins."kairn@kairn".mcp_servers.kairn]
enabled = true

[plugins."kairn@kairn".mcp_servers.kairn.tools.kn_recall]
approval_mode = "approve"
```

Repeat the per-tool table for each tool you want to approve. The per-tool
form has not yet been tested; the server-wide block has been tested.
