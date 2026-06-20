# Changelog

All notable changes to `fastfold-agent-cli` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Deep Agents (LangGraph) runtime** as the default agent engine
  (`agent.runtime=deepagents`), with native progressive skill discovery so many
  installed skills no longer bloat the system prompt.
- **Programmatic Tool Calling (PTC)** tool mode (`agent.tool_mode=ptc`, now the
  default): domain tools are exposed as Python callables inside the sandbox and
  discovered via a compact catalog plus a `search_tools` helper, significantly
  reducing per-turn input tokens and removing the OpenAI tool-count ceiling.
  `agent.tool_mode=native` restores per-tool schemas.
- **Data management commands**: `fastfold data list` (catalog) and
  `fastfold data pull-all` (download every auto-downloadable dataset), plus a
  `/data` interactive command (`/data list | status | pull <name> | pull-all`).
- **Dataset step in `fastfold setup`**: optionally download datasets during the
  wizard with a multi-select (all auto-downloadable datasets preselected).
  Non-interactive via `--datasets depmap,msigdb` / `--datasets all` /
  `--skip-datasets`.
- Configurable tool-trace rendering: `agent.group_tool_traces` and
  `agent.tool_trace_detail_limit` keep the current/last tool call in full detail
  and progressively collapse older ones to compact, still-named lines.
- `shell.run` gains a `working_dir` parameter and auto-extracts a leading
  `cd <dir> &&` so chained-directory commands are no longer blocked.

### Changed

- Minimum Python version is now **3.11** (required by deepagents); install and
  docs updated accordingly.
- Token usage display in the footer/toolbar now shows fresh (non-cached) input
  tokens, subtracting Anthropic prompt-cache reads.
- Tool-call durations under a second now render in milliseconds (e.g. `42ms`)
  instead of `0.0s`.
- `write_todos` renders as a checklist in the interactive trace output.

### Fixed

- Auth, rate-limit, and connection failures now show a concise, actionable
  message instead of dumping a LangGraph/Anthropic traceback.
- Tests no longer clobber the real `~/.fastfold-cli/config.json`: an autouse
  fixture redirects the CLI config directory to a temp path for every test.
- Resolved an `OSError: [Errno 7] Argument list too long` when many/long-named
  skills were installed, by moving skill discovery onto the deepagents runtime.
