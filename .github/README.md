# GitHub Actions

## CI jobs (every push / PR)

| Job | Needs secrets? | What it runs |
|-----|----------------|--------------|
| **Tests & coverage** | No | `compileall` + import smoke, CLI smoke (`--version`, `skills list`, `tool list`), `pytest` + coverage, knowledge benchmark, wheel build/install smoke |
| **Docker image** | No | `docker compose build` + CLI smoke (`--version`, `skills list`, `tool list`) |
| **Live data source smoke** | No | Hits public databases (PubMed, OpenAlex, UniProt, ClinicalTrials.gov, …). `continue-on-error: true` so external drift does not block merges. |

## Manual / tag-triggered jobs

| Job | Needs secrets? | When |
|-----|----------------|--------------|
| **Publish Docker image** | `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` | On `v*.*.*` git tags + **Actions → Publish Docker image → Run workflow**. Pushes `fastfold/fastfold-agent-cli:latest` and `:VERSION` for `linux/amd64` + `linux/arm64`. |

## Optional repository secrets

| Secret | Required? | Purpose |
|--------|-----------|---------|
| `CODECOV_TOKEN` | No (public repos upload without it) | Coverage badge on README; add for private repos at [codecov.io](https://codecov.io) |
| `DOCKERHUB_USERNAME` | For Docker publish only | Docker Hub login (e.g. `fastfold`) |
| `DOCKERHUB_TOKEN` | For Docker publish only | Docker Hub access token with push access |
| `ANTHROPIC_API_KEY` | No for CI today | Only needed if you re-enable live LLM e2e matrix tests |

**You do not need API keys** for the default PR/push CI path.

## Badges (README)

- **CI:** `img.shields.io/github/actions/workflow/status/.../ci.yml?branch=main`
- **Coverage:** `img.shields.io/codecov/c/github/fastfold-ai/fastfold-agent-cli` (activates after first CI upload)

## Smoke / integration env vars

| Variable | Purpose |
|----------|---------|
| `RUN_DATA_SMOKE=1` | Enable live public data-source smoke tests (`tests/test_api_smoke.py`) |
| `DATA_SMOKE_STRICT=1` | Fail on degraded responses (default in CI smoke job) |
| `RUN_E2E_MATRIX=1` | Enable live LLM prompt matrix (requires `ANTHROPIC_API_KEY`) |
| `E2E_MATRIX_LIMIT` | Max prompts in matrix (default: 10) |
| `E2E_MATRIX_STRICT` | Strict assertions in matrix mode |
| `E2E_MATRIX_MAX_FAILED_QUERIES` | Max allowed failures in strict mode |

Legacy names (`RUN_API_SMOKE`, `API_SMOKE_STRICT`, `CT_*`) still work.

## Run checks locally

```bash
# Syntax + imports (same as CI)
python -m compileall -q src
python -c "import agent, api, cli, data, kb, models, reports, skills, tools, ui; from _version import __version__; print(__version__)"

# CLI health
fastfold --version
fastfold skills list
fastfold tool list | head -20

# Unit + coverage (same as CI)
pytest tests/ -q -m "not data_smoke and not integration" \
  --cov=agent --cov=api --cov=cli --cov=data --cov=kb --cov=models \
  --cov=reports --cov=skills --cov=tools --cov=ui \
  --cov-report=xml:coverage.xml --cov-fail-under=55

fastfold knowledge benchmark --strict --min-pass-rate 0.9

# Wheel build smoke
pip install build && python -m build --outdir dist/ && pip install dist/*.whl

# Live data source smoke (PubMed, UniProt, OpenAlex, …)
RUN_DATA_SMOKE=1 DATA_SMOKE_STRICT=1 pytest tests/test_api_smoke.py -q

# Docker smoke
docker compose build
docker compose run --rm -T fastfold --version
docker compose run --rm -T fastfold skills list
```

## Publish Docker image to Docker Hub

Repository secrets (Settings → Secrets → Actions):

- `DOCKERHUB_USERNAME` — e.g. `fastfold`
- `DOCKERHUB_TOKEN` — Docker Hub access token with push permission

**Automated:** push a version tag (`git tag v0.0.50 && git push origin v0.0.50`) or run **Actions → Publish Docker image → Run workflow**.

Publishes multi-arch (`linux/amd64`, `linux/arm64`):

- `fastfold/fastfold-agent-cli:latest`
- `fastfold/fastfold-agent-cli:<version>`

**Manual push from your machine:**

```bash
VERSION=$(grep '^version = ' pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
docker login
docker buildx build --platform linux/amd64,linux/arm64 \
  -t fastfold/fastfold-agent-cli:latest \
  -t fastfold/fastfold-agent-cli:${VERSION} \
  --build-arg VERSION=${VERSION} \
  --push .
```

## Future CI ideas (not enabled yet)

- **`ruff check`** — lint gate once existing ~190 style issues are cleared or scoped
- **`fastfold release-check --no-live`** — doctor + tests + benchmark in one command (needs doctor to tolerate missing API keys in CI)
- **Import boundary check** — `python -c` imports for every top-level package to catch packaging regressions early
- **Skill install smoke** — `fastfold skills add fastfold-ai/skills@skills/find-skills` in Docker or a clean venv
