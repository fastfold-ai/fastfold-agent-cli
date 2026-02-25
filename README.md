# Fastfold Agent CLI

Where scientists and AI agents work together doing real science.

Ask questions in natural language. Fastfold CLI plans the analysis, selects the right tools, executes them, validates results, and returns data-backed conclusions. Integrates with [Fastfold AI Cloud](https://fastfold.ai) for GPU compute, protein folding, workflow orchestration, and team collaboration.

Built on top of [Claude Agent SDK](https://github.com/anthropics/anthropic-sdk-python) and [CellType CLI](https://github.com/celltype/cli).

### Quick install

Requires **Python 3.10+**.

```bash
uv tool install "fastfold-agent-cli[all]" --python 3.10
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
| Missing dependency | `pip install "fastfold-agent-cli[all]"` |
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
