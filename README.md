# Fastfold Agent CLI

Where scientists and AI agents work together doing real science.

Ask questions in natural language. Fastfold CLI plans the analysis, selects the right tools, executes them, validates results, and returns data-backed conclusions. Integrates with [Fastfold AI Cloud](https://fastfold.ai) for GPU compute, protein folding, workflow orchestration, and team collaboration.

Built on top of [Claude Agent SDK](https://github.com/anthropics/anthropic-sdk-python) and [CellType CLI](https://github.com/celltype/cli).

### Prerequisites

- **Python 3.10+** (recommended: let `uv` install managed interpreters).
- **`uv`** — install from the Astral docs: [Installing uv](https://docs.astral.sh/uv/getting-started/installation/). Quick options:
  - **macOS / Linux** (standalone installer):

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

  - **Windows** (PowerShell):

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

    Alternatively: `winget install --id=astral-sh.uv -e` (see Astral docs for other methods).

    After installing **`uv`**, close and reopen your terminal or PowerShell so **`PATH`** picks up the `uv` executable.

### Quick install

Requires **Python 3.10+** and prerequisites above.

```bash
uv tool install "fastfold-agent-cli[all]" --python 3.10
```

**Windows users — prefer WSL2 + Ubuntu (`[all]`):** **`tiledbsoma`** does not publish usable native Windows wheels, so **`[all]`** on cmd/PowerShell usually fails. Install **Windows Subsystem for Linux** following Microsoft’s guide: **[Install WSL](https://learn.microsoft.com/en-us/windows/wsl/install)** (recommended default distro **Ubuntu**), open an **Ubuntu** terminal, install **`uv`** + Python there, then run **`uv tool install "fastfold-agent-cli[all]" --python 3.10`** inside WSL.

**Staying on native Windows cmd/PowerShell:** use **`[win_build]`** instead of **`[all]`** (same stack minus **`scanpy` / `cellxgene-census` / `tiledbsoma`**), or explicitly:

```bash
uv tool install "fastfold-agent-cli[chemistry,biology,ml,analysis]" --python 3.10
```

Convenience extra (**CLI ≥ `0.0.36`** on PyPI; **0.0.36+** fixes interactive `fastfold` on native Windows):

```bash
uv tool install "fastfold-agent-cli[win_build]" --python 3.10
```

### Authentication

```bash
# Interactive setup wizard (recommended — configures Anthropic + Fastfold AI Cloud keys)
fastfold setup

# Or set directly
export ANTHROPIC_API_KEY="sk-ant-..."
export FASTFOLD_API_KEY="sk-..."

# Non-interactive (CI/scripting)
fastfold setup --api-key sk-ant-... --fastfold-api-key sk-...
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
| **`ModuleNotFoundError: No module named 'termios'`** (interactive `fastfold` on Windows) | Upgrade to **`fastfold-agent-cli` 0.0.36+** (e.g. `uv tool install "fastfold-agent-cli[win_build]" --python 3.10 --upgrade`). |
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
