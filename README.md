# Fastfold Agent CLI

An agent CLI for scientists with local and cloud integrations.

Built on top of [Claude Agent SDK](https://github.com/anthropics/anthropic-sdk-python) and [CellType CLI](https://github.com/celltype/cli).

### Prerequisites

<details>
<summary>Install `uv`</summary>

- **Python 3.10+** (recommended: let `uv` install managed interpreters).
- **`uv`** — [Installing uv](https://docs.astral.sh/uv/getting-started/installation/). Quick options:

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Alternatively: `winget install --id=astral-sh.uv -e` (see Astral docs for other methods).

After installing `uv`, close and reopen your terminal or PowerShell so `PATH` picks up the `uv` executable.

</details>

### Quick install

Requires **Python 3.10+** and prerequisites above.

**Linux / macOS:**

```bash
uv tool install "fastfold-agent-cli[all]" --python 3.10
```

**Windows (cmd/PowerShell):**

```bash
uv tool install "fastfold-agent-cli[win_build]" --python 3.10
```

<details>
<summary>Windows installation notes</summary>

#### Recommended: WSL2 + Ubuntu (full `[all]` stack)

`tiledbsoma` does not publish usable native Windows wheels, so `[all]` on cmd/PowerShell usually fails. Use WSL instead:

1. Install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) (Ubuntu recommended).
2. Open an **Ubuntu** terminal and install `uv` + Python (see [Prerequisites](#prerequisites) above).
3. Run the same install command inside WSL:

```bash
uv tool install "fastfold-agent-cli[all]" --python 3.10
```

</details>

### Authentication

```bash
# Interactive setup wizard (recommended)
# Choose provider(s) first (interactive toggle list), then enter keys
fastfold setup

# Or choose provider(s) explicitly (comma-separated)
fastfold setup --provider anthropic
fastfold setup --provider openai
fastfold setup --provider anthropic,openai

# Or set directly
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export FASTFOLD_API_KEY="sk-..."

# Non-interactive (CI/scripting)
fastfold setup --api-key sk-ant-... --fastfold-api-key sk-...
fastfold setup --provider openai --openai-api-key sk-... --fastfold-api-key sk-...
```

Provider selection:

```bash
fastfold config set llm.provider anthropic
fastfold config set llm.model claude-sonnet-4-5-20250929
fastfold config set llm.anthropic_api_key sk-ant-...

fastfold config set llm.provider openai
fastfold config set llm.model gpt-4o
fastfold config set llm.openai_api_key sk-...

# Legacy fallback (Anthropic only, still supported)
fastfold config set llm.api_key sk-ant-...
```

## Getting Started

```bash
# Start interactive session
fastfold

# Single query
fastfold "What are the top degradation targets for this compound?"

# Validate setup
fastfold doctor

# List available tools
fastfold tool list

# List loaded skills
fastfold skill list
```

### Interactive commands

Inside `fastfold` interactive mode:

- `/help` — command reference + examples
- `/tools` — list all tools with status
- `/agents N <query>` — run N parallel research agents
- `/sessions`, `/resume` — session lifecycle
- `/copy`, `/export` — save/share outputs
- `/usage` — token and cost tracking

### Quick examples

**Target prioritization**
```
fastfold "I have a CRBN molecular glue. Proteomics shows it degrades
          IKZF1, GSPT1, and CK1α. Which target should I prioritize?"
```

**Protein folding**
```
fastfold "Fold this sequence with boltz-2 and find the binding pockets: MALWMRLLPLL..."
```

**Combination strategy**
```
fastfold "My lead compound is immune-cold. What combination strategy should I use?"
```

## Key Features

### 190+ Domain Tools

| Category | Examples |
|---|---|
| **Target** | Neosubstrate scoring, degron prediction, co-essentiality networks |
| **Chemistry** | SAR analysis, fingerprint similarity, scaffold clustering |
| **Expression** | L1000 signatures, pathway enrichment, TF activity, immune scoring |
| **Viability** | Dose-response modeling, PRISM screening, therapeutic windows |
| **Biomarker** | Mutation sensitivity, resistance profiling, dependency validation |
| **Clinical** | Indication mapping, population sizing, TCGA stratification |
| **Safety** | Anti-target flagging, multi-modal profiling, SALL4 risk |
| **Structure** | AlphaFold fetch, docking, binding sites, MD simulation |
| **Folding** | Fastfold AI Cloud: boltz-2, monomer, multimer, simplefold_* |
| **Literature** | PubMed, OpenAlex, ChEMBL search |
| **DNA** | ORF finding, codon optimization, primer design, Gibson/Golden Gate assembly |

### Agent Skills

Fastfold CLI ships with a bundled skill catalog and supports user-installed skills:

```bash
fastfold skill list          # see loaded skills

# Install additional skills (requires npx), check https://skills.sh
npx skills addd <owner/repo>
```

### Data Management

```bash
fastfold data pull depmap    # DepMap CRISPR, mutations, expression
fastfold data pull prism     # PRISM cell viability
fastfold data pull msigdb    # Gene sets
fastfold data pull alphafold     # Protein structures (on-demand)

# Or point to existing data
fastfold config set data.depmap /path/to/depmap/
```

### Reports

```bash
fastfold report list         # list reports
fastfold report publish      # convert latest .md to .html
fastfold report show         # open in browser
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `fastfold` fails at startup | `fastfold doctor` |
| No API key | `fastfold setup` or `export ANTHROPIC_API_KEY=...` |
| Data not found | `fastfold data pull <dataset>` |
| **`tiledbsoma` / WinError installing `[all]` on Windows** | Prefer **[WSL2 + Ubuntu](https://learn.microsoft.com/en-us/windows/wsl/install)** and **`[all]`** in Linux. Native Windows: **`[win_build]`** or **`[chemistry,biology,ml,analysis]`** — see **Quick install**. |
| Missing dependency (pip fallback) | `pip install "fastfold-agent-cli[all]"` |
| **`ModuleNotFoundError: No module named 'termios'`** (interactive `fastfold` on Windows) | Upgrade **`fastfold-agent-cli` ≥ 0.0.36** (e.g. `uv tool install "fastfold-agent-cli[win_build]" --python 3.10 --upgrade`). |
| **`CLINotFoundError`** / **`WinError 206`** when spawning Claude (**Windows**) | **`≥ 0.0.42`**: run **`fastfold autofix`** (or interactive **`/autofix`**) and re-run **`fastfold doctor`**. This release also avoids oversized Windows CLI command lines by moving large system instructions into the stdin payload path. If still blocked, set shorter **`UV_TOOL_DIR`** ([uv tools directory](https://docs.astral.sh/uv/reference/storage/#tools)) + reinstall, or set **`FASTFOLD_CLAUDE_CODE_CLI`** to a known good global Claude Code launcher path. |
| Session lost | `fastfold --continue` |

## Contributing

```bash
git clone https://github.com/fastfold-ai/fastfold-agent-cli.git
cd fastfold-agent-cli
uv venv --python 3.12 && uv sync
fastfold setup
pytest tests/
```

## License

MIT — see LICENSE

## Credits

Based on [CellType CLI](https://github.com/celltype/cli)
