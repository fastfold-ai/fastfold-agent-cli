# Fastfold Agent CLI

<p align="center">
  <a href="https://github.com/fastfold-ai/fastfold-agent-cli/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/fastfold-ai/fastfold-agent-cli/ci.yml?branch=main&label=CI&logo=github" alt="CI status" /></a>
  <a href="https://codecov.io/gh/fastfold-ai/fastfold-agent-cli"><img src="https://img.shields.io/codecov/c/github/fastfold-ai/fastfold-agent-cli?branch=main&logo=codecov&label=coverage" alt="Codecov coverage" /></a>
  <a href="https://skills.sh/fastfold-ai/skills"><img src="https://img.shields.io/badge/Skills-000000?logo=vercel&logoColor=white" alt="Skills" /></a>
  <a href="https://pypi.org/project/fastfold-agent-cli/"><img src="https://img.shields.io/pypi/v/fastfold-agent-cli.svg?label=agent-cli&color=22c55e" alt="fastfold-agent-cli on PyPI" /></a>
  <a href="https://hub.docker.com/r/fastfold/fastfold-agent-cli"><img src="https://img.shields.io/docker/v/fastfold/fastfold-agent-cli?sort=semver&label=docker&logo=docker" alt="Docker Hub" /></a>
  <a href="https://fastfoldcommunity.slack.com/join/shared_invite/zt-3azbiw7qv-majshu97TymCma_TwFT6mw#/shared-invite/email"><img src="https://img.shields.io/badge/Slack-Join%20Slack-4A154B?logo=slack" alt="Join Slack" /></a>
</p>

![Fastfold CLI demo](img/demo.gif)

Fastfold Agent CLI is an agentic research environment for drug discovery and computational biology. Think of it as a coding agent, but for biology. You ask a question in natural language and it plans and executes multi-step scientific workflows using 190+ specialized tools, installable skills, Fastfold Cloud compute, and a persistent Python sandbox. It is intentionally narrow, built for the real, tool-heavy research workflows scientists run rather than for being a general-purpose chatbot.

Our mission is to bring the best tools to scientists wherever they work: on the cloud, on local compute, university HPC, or inside the enterprise.

Under the hood it runs on a [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) (LangChain / LangGraph) agentic loop with **Programmatic Tool Calling (PTC)** and native **progressive skill discovery**. A single agent plans, calls domain tools as Python inside a persistent sandbox, self-corrects, and synthesizes a report. Anthropic, OpenAI, and local or OpenAI-compatible models all work through the same loop.

Many of the tools and prompts trace back to [CellType](https://github.com/celltype/celltype-agent); see [Acknowledgements](#acknowledgements) for credits.


### Why Fastfold CLI

You get 190+ tools, 30+ database APIs, and multi-step planning, plus what Fastfold adds for composability and extensibility:

- **GPU compute models & workflows**: run heavy scientific workflows like folding, protein design, and MD simulation on Fastfold Cloud, your own compute, or providers like Modal, Boltzbio, and more.
- **Installable skills**: discover, add, and share workflows natively (`fastfold skills find`, `fastfold skills add <github url>`).
- **Any model**: Anthropic, OpenAI, or local/open models like Gemma, Qwen, and DeepSeek through endpoints like **Ollama**, **Unsloth**, **oMLX**, **DS4**, **llama.cpp**, and **LM Studio** via `/model`, `/model-manager`, or `fastfold setup`.

### Quick install

<details>
<summary>Install <code>uv</code> (for native CLI install)</summary>

- **Python 3.11+** (recommended: let `uv` install managed interpreters).
- **uv** — [Installing uv](https://docs.astral.sh/uv/getting-started/installation/). Quick options:

**Linux / macOS:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (cmd/PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Alternatively: `winget install --id=astral-sh.uv -e` (see Astral docs for other methods).

After installing `uv`, close and reopen your terminal or PowerShell so `PATH` picks up the `uv` executable.

</details>

**Linux / macOS:**

```bash
uv tool install "fastfold-agent-cli[all]" --python 3.11
```

**Windows (cmd/PowerShell):**

```bash
uv tool install "fastfold-agent-cli[win_build]" --python 3.11
```

<details>
<summary>Install via WSL2 + Ubuntu (full <code>[all]</code> stack, recommended)</summary>

`tiledbsoma` does not publish usable native Windows wheels, so `[all]` on cmd/PowerShell usually fails. Use WSL instead:

1. Install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) (Ubuntu recommended).
2. Open an **Ubuntu** terminal and install `uv` + Python (see [Prerequisites](#prerequisites) above).
3. Run the same install command inside WSL:

```bash
uv tool install "fastfold-agent-cli[all]" --python 3.11
```

</details>

### Docker

```bash
docker run -it --rm \
  -v fastfold-cli:/root/.fastfold-cli \
  fastfold/fastfold-agent-cli:latest
```

### Authentication

The interactive setup wizard is the easiest way to get started. It lets you pick provider(s) from a toggle list, then enter keys:

```bash
fastfold setup
```

To skip the toggle list, pass the provider(s) explicitly (comma-separated):

```bash
fastfold setup --provider anthropic
fastfold setup --provider openai
fastfold setup --provider openai_compatible
fastfold setup --provider anthropic,openai
```

Prefer environment variables? Set keys directly:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export FASTFOLD_API_KEY="sk-..."
export BOLTZ_API_KEY="sk_bc_..."
```

### Provider integrations (`/keys`)

Use `/keys` (interactive) or `fastfold keys` (shell) to see integration status, masked previews, config keys, and setup links.

| Service | Env var | Config key | Unlocks | Get key / setup |
| --- | --- | --- | --- | --- |
| Anthropic | `ANTHROPIC_API_KEY` | `llm.anthropic_api_key` | Claude model access (default provider) | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) |
| OpenAI | `OPENAI_API_KEY` | `llm.openai_api_key` | OpenAI model access | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| OpenAI-compatible | `OPENAI_COMPATIBLE_API_KEY` | `llm.openai_compatible_api_key` / profile key | Local/self-hosted OpenAI-compatible endpoints | [Ollama API docs](https://docs.ollama.com/api/introduction) |
| Fastfold AI Cloud | `FASTFOLD_API_KEY` | `api.fastfold_cloud_key` | Fastfold cloud skills and integrations | [cloud.fastfold.ai/api-keys](https://cloud.fastfold.ai/api-keys) |
| Boltz | `BOLTZ_API_KEY` | `api.boltz_api_key` | Boltz API skills (`boltz` skill modes) | [api.boltz.bio/console](https://api.boltz.bio/console) |
| IBM RXN | `IBM_RXN_API_KEY` | `api.ibm_rxn_key` | Retrosynthesis workflows | [rxn.res.ibm.com](https://rxn.res.ibm.com) |
| Lens.org | `LENS_API_KEY` | `api.lens_key` | Patent search workflows | [lens.org subscriptions](https://www.lens.org/lens/user/subscriptions) |
| SendGrid | `SENDGRID_API_KEY` | `notification.sendgrid_api_key` | Email notification delivery | [sendgrid.com](https://sendgrid.com) |
| Lambda Labs | `LAMBDA_API_KEY` | `compute.lambda_api_key` | GPU compute job submission | [cloud.lambdalabs.com](https://cloud.lambdalabs.com) |
| RunPod | `RUNPOD_API_KEY` | `compute.runpod_api_key` | GPU compute job submission | [runpod.io](https://www.runpod.io) |

Boltz quick setup from interactive mode:

```bash
/keys set-boltz
```

That flow prompts for `BOLTZ_API_KEY`, then offers to install:
- the Fastfold Boltz skill (`fastfold-ai/skills@skills/boltz`)
- the `boltz-api` CLI (`curl -fsSL https://install.boltz.bio/boltz-api/install.sh | sh`)

For CI or scripting, pass keys non-interactively:

```bash
fastfold setup --api-key sk-ant-... --fastfold-api-key sk-...
fastfold setup --provider openai --openai-api-key sk-... --fastfold-api-key sk-...
fastfold setup --provider anthropic --api-key sk-ant-... --fastfold-api-key sk-... --boltz-api-key sk_bc_...
```

Local/compatible endpoints can also be configured in one line:

```bash
fastfold setup --provider openai_compatible --profile-label "Ollama Local" --profile-template ollama --profile-endpoint http://localhost:11434/v1 --profile-key ollama
fastfold setup --provider openai_compatible --profile-label "Unsloth Local" --profile-template unsloth --profile-endpoint http://localhost:8888/v1 --profile-key sk-unsloth-...
fastfold setup --provider openai_compatible --profile-label "oMLX Local" --profile-template omlx --profile-endpoint http://localhost:8000/v1 --profile-key sk-omlx-...
fastfold setup --provider openai_compatible --profile-label "DS4 Local" --profile-template ds4 --profile-endpoint http://localhost:8000/v1 --profile-key dsv4-local
fastfold setup --provider openai_compatible --profile-label "llama.cpp Local" --profile-template llama_cpp --profile-endpoint http://localhost:8080/v1
fastfold setup --provider openai_compatible --profile-label "LM Studio Local" --profile-template lm_studio --profile-endpoint http://localhost:1234/v1
```

### Local and OpenAI-compatible models (Ollama, Unsloth, oMLX, DS4, llama.cpp, LM Studio, and other gateways)

Fastfold supports local/self-hosted OpenAI-compatible endpoints in both `fastfold setup` and interactive mode:

- `/model`: model/provider selection only
- `/model-manager`: add/edit/delete compatible profiles and inspect endpoint health/model discovery

Recommended (interactive):

```bash
fastfold setup --provider openai_compatible
```

The setup wizard will guide you through:

1. Endpoint type selection:
  - `Ollama` (`/api/tags`)
  - `Unsloth` (`/v1/models`, auth)
  - `oMLX` (`/v1/models`, auth)
  - `DS4` (`/v1/models`)
  - `llama.cpp` (`/v1/models`)
  - `LM Studio` (`/v1/models`)
  - `Other OpenAI-compatible` (`/v1/models` then `/api/tags`)
2. Endpoint base URL:
  - Ollama default: `http://localhost:11434/v1`
  - Unsloth default: `http://localhost:8888/v1`
  - oMLX default: `http://localhost:8000/v1`
  - DS4 default: `http://localhost:8000/v1`
  - llama.cpp default: `http://localhost:8080/v1`
  - LM Studio default: `http://localhost:1234/v1`
3. API key prompt (backend-aware):
  - Ollama commonly uses `ollama` placeholder key
  - Unsloth uses your Unsloth Studio key
  - DS4, llama.cpp, and LM Studio can use optional/custom keys depending on your local server configuration
4. Profile summary preview (label/template/endpoint), then model discovery + selection (or manual model ID entry)

### Inference engine install references

Use these upstream projects/docs to install and run local or self-hosted LLM inference engines before adding profiles in Fastfold:

- **DS4 (DeepSeek v4 Flash/PRO engine):** [antirez/ds4](https://github.com/antirez/ds4)
- **llama.cpp:** [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- **LM Studio OpenAI-compatible server:** [LM Studio OpenAI Compatibility](https://lmstudio.ai/docs/developer/openai-compat)
- **Ollama:** [ollama/ollama](https://github.com/ollama/ollama)
- **oMLX:** [jundot/omlx](https://github.com/jundot/omlx)
- **Unsloth:** [unslothai/unsloth](https://github.com/unslothai/unsloth)

Scripted profile setup flags:

```bash
fastfold setup --provider openai_compatible \
  --profile-label "Custom Gateway" \
  --profile-template other \
  --profile-endpoint https://gateway.example.com/v1 \
  --profile-key sk-... \
  --profile-default-model gpt-oss-120b \
  --set-default-profile
```

You can also configure directly:

```bash
# Ollama
fastfold config set llm.provider openai
fastfold config set llm.openai_base_url http://localhost:11434/v1
fastfold config set llm.openai_compatible_backend ollama
fastfold config set llm.openai_compatible_api_key ollama
fastfold config set llm.model llama3.1

# Unsloth
fastfold config set llm.provider openai
fastfold config set llm.openai_base_url http://localhost:8888/v1
fastfold config set llm.openai_compatible_backend unsloth
fastfold config set llm.openai_compatible_api_key sk-unsloth-...
fastfold config set llm.model <unsloth-model-id>
```

Environment variables for compatible endpoints:

```bash
export OPENAI_BASE_URL="http://localhost:11434/v1"
export OPENAI_COMPATIBLE_API_KEY="ollama"
```

Inside interactive mode:

- Run `/model` to switch across configured providers/models.
- Run `/model-manager` to add/edit/delete OpenAI-compatible profiles and view endpoint health/model lists.

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
fastfold skills list
```

### Interactive commands

Inside `fastfold` interactive mode (run `/help` for the full reference):

**Discover**

- `/help`: show command reference with examples
- `/tools`: list all tools with status (stable/experimental)
- `/data`: manage local datasets (`/data list`, `/data status`, `/data pull <name>`, `/data pull-all`)
- `/skills`: list currently loaded skills
- `/skills-find [query]`: discover installable skills from the catalog
- `/skills-add <source>`: install a skill from GitHub/local path/name
- `/skills-remove <name>`: remove a globally-installed skill
- `/case-study`: run/list curated case studies (`/case-study list`)

**Models & configuration**

- `/model`: switch LLM model/provider interactively
- `/model-manager`: manage OpenAI-compatible profiles (add/edit/delete + diagnostics)
- `/settings`: configure UI and agent preferences
- `/config`: show active runtime configuration
- `/keys`: show API key setup status by service (`/keys profile`, `/keys set-compatible <profile_id>`, `/keys set-boltz`)

**Run control**

- `/agents N <query>`: run a query with N parallel research agents
- `/plan`: toggle plan mode (preview & approve before executing)
- `/interrupt`: interrupt the active generation (add `!` to force)
- `/compact`: compress session context for longer runs
- `/tasks`: show background task watcher status (`/tasks refresh` for live probe)

**Sessions & output**

- `/new`: start a new empty session
- `/sessions`: list saved sessions (or delete: `/sessions delete <id>`)
- `/resume`: resume a previous session by id/index
- `/usage`: show session token/cost usage
- `/copy`: copy the last answer to clipboard
- `/export`: export current session transcript to markdown
- `/export-share`: export session, send to Slack, and save to library
- `/notebook`: export current session as Jupyter notebook (`.ipynb`)

**Maintenance**

- `/upgrade`: upgrade `fastfold-agent-cli` via uv
- `/doctor`: run readiness diagnostics and fix hints
- `/clear`: clear the screen
- `/exit`: exit the terminal

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


| Category       | Examples                                                                    |
| -------------- | --------------------------------------------------------------------------- |
| **Target**     | Neosubstrate scoring, degron prediction, co-essentiality networks           |
| **Chemistry**  | SAR analysis, fingerprint similarity, scaffold clustering                   |
| **Expression** | L1000 signatures, pathway enrichment, TF activity, immune scoring           |
| **Viability**  | Dose-response modeling, PRISM screening, therapeutic windows                |
| **Biomarker**  | Mutation sensitivity, resistance profiling, dependency validation           |
| **Clinical**   | Indication mapping, population sizing, TCGA stratification                  |
| **Safety**     | Anti-target flagging, multi-modal profiling, SALL4 risk                     |
| **Structure**  | AlphaFold fetch, docking, binding sites, MD simulation                      |
| **Folding**    | Fastfold AI Cloud: boltz-2, monomer, multimer, simplefold_*                 |
| **Literature** | PubMed, OpenAlex, ChEMBL search                                             |
| **DNA**        | ORF finding, codon optimization, primer design, Gibson/Golden Gate assembly |


### Agent runtime & efficiency

The agent runs on a [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) (LangGraph) runtime with native **progressive skill discovery**, so skill details load on demand and you can install many skills without bloating the prompt. Anthropic prompt caching is enabled automatically, and the footer shows fresh (non-cached) input tokens so the numbers stay meaningful.

**Programmatic Tool Calling (PTC)** is the default tool mode (`agent.tool_mode=ptc`): the agent calls domain tools as Python functions inside a persistent sandbox and discovers them through a compact catalog plus `search_tools`, instead of injecting every tool's JSON schema. This significantly reduces per-turn input tokens and removes the OpenAI tool-count ceiling. Set `agent.tool_mode=native` to restore per-tool schemas.

Tool-call traces keep the current/last call in full detail and progressively collapse older ones to compact named lines (tune with `agent.tool_trace_detail_limit`; set `agent.group_tool_traces=false` for fully verbose output).

### Agent Skills

Fastfold ships with a bundled skill catalog and lets you discover, add, and manage skills natively. Installed skills live in `~/.fastfold-cli/skills/` and are picked up automatically.

List what's loaded:

```bash
fastfold skills list
```

Discover skills from the catalog:

```bash
fastfold skills find                 # browse all
fastfold skills find "protein design"  # filter by query
```

Add a skill from a GitHub URL, `owner/repo@subpath`, a local path, or a catalog name:

```bash
fastfold skills add https://github.com/fastfold-ai/skills/tree/main/skills/fold
fastfold skills add fastfold-ai/skills@skills/fold
fastfold add skills ./my-skill        # alias for `skills add`
```

Keep skills current (sync the Fastfold catalog — adds new + updates existing — and re-install other tracked skills):

```bash
fastfold skills upgrade                 # sync catalog + update installed
fastfold skills upgrade --catalog-only  # only refresh the Fastfold catalog
fastfold skills upgrade --no-catalog    # only update already-installed skills
```

Inspect or remove:

```bash
fastfold skills info fold
fastfold skills remove fold
fastfold skills delete --all   # remove ALL user-installed skills (asks to confirm)
```

Inside the interactive session you can use the slash commands `/skills`, `/skills-find [query]`, `/skills-add <source>`, and `/skills-remove <name>`. Install uses a native `git clone` and falls back to `npx skills add` (skills.sh) when needed.

`fastfold setup` also offers to install skills interactively — it live-fetches the current Fastfold catalog, lets you multi-select, suggests the community collections below, and accepts custom sources. It prefers `npx skills add` when Node is available, otherwise uses `git`. Non-interactive: `fastfold setup --skills "fastfold-ai/skills@skills/fold,..."` or `--skip-skills`.

#### Community skill collections

You can also install skills from other providers. Each command installs the whole collection (use `npx skills add ...` instead if you prefer Node):

- **K-Dense-AI** — scientific agent skills ([repo](https://github.com/K-Dense-AI/scientific-agent-skills))

```bash
fastfold skills add K-Dense-AI/scientific-agent-skills
```

- **Anthropic** — life-sciences skills ([repo](https://github.com/anthropics/life-sciences#skills))

```bash
fastfold skills add anthropics/life-sciences
```

- **DeepMind** — science skills ([repo](https://github.com/google-deepmind/science-skills))

```bash
fastfold skills add google-deepmind/science-skills
```

**Create your own skill** with the bundled `skill-creator` (scaffold, validate, package), and let the agent discover skills via the `find-skills` skill. To let the agent install skills itself, enable `fastfold config set skills.allow_agent_install true`.

### Data Management

```bash
fastfold data list           # catalog: description, size, auto/manual
fastfold data status         # what's downloaded locally
fastfold data pull depmap    # DepMap CRISPR, mutations, expression
fastfold data pull prism     # PRISM cell viability
fastfold data pull msigdb    # Gene sets
fastfold data pull alphafold # Protein structures (on-demand)
fastfold data pull-all       # every auto-downloadable dataset (depmap is ~580MB)

# Or point to existing data
fastfold config set data.depmap /path/to/depmap/
```

`fastfold setup` also offers an optional dataset step with a multi-select (all
auto-downloadable datasets preselected). Non-interactive:
`fastfold setup --datasets depmap,msigdb` (or `--datasets all` / `--skip-datasets`).

The same actions are available inside the interactive session via `/data`
(`/data list`, `/data status`, `/data pull <name>`, `/data pull-all`).

### Reports

```bash
fastfold report list         # list reports
fastfold report publish      # convert latest .md to .html
fastfold report show         # open in browser
```

### Explore Fastfold Apps

Browse the Fastfold Apps catalog at [https://cloud.fastfold.ai/apps](https://cloud.fastfold.ai/apps)
- Fold model options include: ESM-1b, IntelliFold, OpenFold 3, AlphaFold2, Boltz-1, Boltz-2, Chai-1, and SimpleFold.
- MD workflow options include: OpenMM Calvados and OpenMMDL.
- Protein Design workflows coming soon: Boltzgen and Bindcraft.

![Fastfold Apps Catalog](img/apps.png)


## Benchmarks

**Coming soon.** Generic Q&A leaderboards don't capture what matters here, so we're crafting a comprehensive benchmark focused on **industry-specific drug-discovery and computational-biology use cases**: the multi-step, tool-heavy workflows scientists actually run (target prioritization, folding, protein design, MD, omics analysis, and more), measured across multiple model backends.

We are looking for help. If you have a real-world use case you'd like represented, or want to contribute tasks, datasets, or scoring rubrics, please [open an issue](https://github.com/fastfold-ai/fastfold-agent-cli/issues) or say hi on [Slack](https://fastfoldcommunity.slack.com/join/shared_invite/zt-3azbiw7qv-majshu97TymCma_TwFT6mw#/shared-invite/email). The goal is an honest, reproducible measure of how the agent performs across many domains.

## Contributing

Contributions are welcome, from bug reports and docs fixes to new tools and skills.

Clone the repo and set up a development environment:

```bash
git clone https://github.com/fastfold-ai/fastfold-agent-cli.git
cd fastfold-agent-cli
uv venv --python 3.12 && uv sync
fastfold setup
```

Run the test suite before opening a PR:

```bash
pytest tests/          # full suite
pytest tests/ -v       # verbose
pytest tests/test_cli.py::test_name   # a single test
```

Adding a new tool? Tools live in `src/tools/` and register with the `@registry.register(...)` decorator. Each tool's name prefix must match its category, it should accept `**kwargs`, and it must return a dict with a `"summary"` key. Use lazy imports for data loaders inside the function body. See `CLAUDE.md` for the full tool pattern and conventions.

A few guidelines:

- Keep changes focused and add tests for new behavior (tests mock data loaders, so they don't require real datasets).
- Match the existing code style and run `pytest` locally until green.
- Open a PR with a clear description of the change and why it's needed.

## License

MIT

## Acknowledgements

Fastfold Agent CLI stands on the shoulders of excellent open-source work:

- **[CellType](https://github.com/celltype/celltype-agent):** most of the 190+ domain tools and prompt definitions are derived from or inspired by CellType's open-source computational-biology agent, which reports a state-of-the-art **90% on BixBench-Verified-50**.
- **[BixBench](https://github.com/Future-House/BixBench)** and the **[BixBench-Verified-50](https://huggingface.co/datasets/phylobio/BixBench-Verified-50)** dataset: the evaluation that grounds the domain-agent results above.
- **[Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)** (LangChain / LangGraph): the agentic runtime Fastfold is built on today, providing progressive skill discovery and the foundation for Programmatic Tool Calling.
- **[open-ptc-agent](https://github.com/Chen-zexi/open-ptc-agent):** the reference that shaped Fastfold's Programmatic Tool Calling (PTC) approach to keeping per-turn tokens low.

