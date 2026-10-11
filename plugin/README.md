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
`~/.kairn`. The store itself stays on this machine. Recalled knowledge is
returned to your agent and becomes part of its conversation with the model.

First start: a cold start downloads the package. Claude Code gives an MCP
server 30 seconds to start. Run
`uvx --from kairn-ai==0.3.0 kairn --help` once in a terminal.
It downloads and caches the package, then exits. Alternatively, start Claude
Code with `MCP_TIMEOUT=120000` to raise the limit to 120 seconds.
`MCP_TIMEOUT` is measured in milliseconds; see
[Claude Code's MCP guide](https://code.claude.com/docs/en/mcp-quickstart).

Apps started from the macOS Dock or a desktop launcher may not see the PATH
your terminal uses, so `uvx` can be missing there while it works in a terminal.

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
A plugin update or reinstall replaces the files in the client's plugin
cache, so a hand edit there is lost.
For a lasting setup with another folder, skip the plugin and add Kairn by
hand: `claude mcp add kairn -- uvx --from kairn-ai==0.3.0 kairn serve --init ~/brain`
(Codex: `codex mcp add kairn -- uvx --from kairn-ai==0.3.0 kairn serve --init ~/brain`).

Codex keys plugin MCP servers by bare name, so a hand-added `kairn` server
and the plugin's `kairn` server collapse into one entry. Claude Code shows
both as `kairn` and `plugin:kairn:kairn`, each on its own store. Remove the
hand-added entry with `claude mcp remove kairn` or `codex mcp remove kairn`,
or skip the plugin.

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
