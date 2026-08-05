# Dynamic Reflexive Memory – Developer Guide

## Current Status
- Primary entry point is `main.py`, which launches the PySide6 GUI by default and falls back to CLI mode automatically in headless environments.
- Data layer combines Redis (working memory), ChromaDB (episodic/semantic stores), and persisted review cycles handed off to the GUI telemetry panels.
- Strict typing enforced through Pydantic models, dataclasses, and Pyright strict mode for core modules.
- Local runtime first; external providers are optional and configured through environment variables or `config/config.json`.
- `uv` is supported for environment creation, installs, and command execution; plain `pip`/`venv` remains valid.

## Environment & Tooling
- Target interpreter: **Python 3.12**; create a dedicated virtual environment per workspace.
- Linting and formatting: Ruff (`ruff check` and `ruff format`).
- Type checking: `pyright` in strict mode before publishing changes.
- Testing stack: `pytest`, `pytest-asyncio`, `pytest-cov`, `hypothesis`, and `vcrpy` for HTTP fixtures.

## Detailed Getting Started
1. **Create the synchronized developer environment**
   ```bash
   uv sync --all-extras --frozen
   ```
   Fallback without `uv`:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
2. **Launch local services**
   ```bash
   docker compose up -d
   ```
   - Redis listens on port `6379`.
   - ChromaDB persists via the Python package (no container needed).
   - Ollama (port `11434`) can serve local models; pull models with `ollama pull <model>`.
3. **Run DRM**
   ```bash
   # GUI
   uv run python main.py --mode gui

   # CLI/task runner
   uv run python main.py --mode cli --task "Draft integration plan" --feedback "Looks good"

   # Explicit raw output for a trusted local terminal only
   uv run python main.py --mode cli --task "Draft integration plan" --show-result
   ```
   CLI workflow failures return a non-zero exit code. Default console logs remain redacted; `--show-result` is an explicit opt-in that writes raw task output to stdout.

## Configuration & Secrets
- Copy `config/config.example.json` to `config/config.json` and update provider credentials before first run.
- `config/config.json` holds workflow routing, model identifiers, and storage paths; keep values explicit to honor KISS/DRY.
- `config/logging.conf` defines structured log routing (telemetry, spans, metrics). Extend handlers here instead of in code.
- Environment variables load through `.env` (managed by `python-dotenv`). Never commit `.env` files.
- Missing or invalid credentials raise descriptive `WorkflowError` exceptions with contextual hints; propagate new errors through custom exception types.

## Environment Variables (.env)
| Variable | Purpose |
| --- | --- |
| `AZURE_OPENAI_API_KEY` | Required for Azure-hosted LLM calls via LiteLLM. |
| `AZURE_OPENAI_ENDPOINT` | Base endpoint for Azure OpenAI deployments. |
| `AZURE_OPENAI_API_VERSION` | Optional version override for Azure OpenAI REST API. |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Overrides the default `text-embedding-3-large` embedding deployment. |
| `OLLAMA_BASE_URL` | Points DRM to a remote/local Ollama instance (auto-detected on WSL2, but overridable). |
| `DRM_MEMORY_LOG_PATH` | Repoints the memory revision log from `data/logs/memory_revisions.jsonl`. |
| `DRM_MEMORY_AUDIT_LOG_MODE` | Defaults to `redacted`; set explicitly to `full` only when raw revision payloads and replay are required in a trusted local environment. |

## Testing & Quality Gates
- Install dev requirements and run the full gate before merging:
  ```bash
  uv run --all-extras --frozen ruff check .
  uv run --all-extras --frozen ruff format --check .
  uv run --all-extras --frozen pyright
  ```
- Pytest remains a local regression tool and is not a required CI gate:
  ```bash
  uv run --all-extras --frozen pytest --cov --cov-fail-under=85 --disable-warnings -q
  ```
- Without `uv`, run the same commands directly from the activated virtualenv.
- Favor `hypothesis` strategies for boundary inputs (long prompts, Unicode edge cases, malformed JSON payloads).
- Redis acceptance provisions a temporary Docker container on a dynamically assigned loopback port; real Chroma acceptance uses a temporary persistence directory and a deterministic local embedding function, so neither test requires a provider or project data.
- Mock outbound HTTP/database calls with `pytest-mock`, `vcrpy`, or async test clients to keep suites deterministic.
- Keep coverage ≥85% on core logic modules; justify exceptions in PR descriptions if temporary.

## Memory & Telemetry
- **Memory Revision Log**: mutations append redacted audit records to `data/logs/memory_revisions.jsonl`; use `DRM_MEMORY_LOG_PATH` to override for CI. Each application initialization atomically prunes records older than 30 days, discards malformed entries, and re-chains the retained log. Set `DRM_MEMORY_AUDIT_LOG_MODE=full` only for a trusted local audit session that explicitly requires raw payload replay.
- **Semantic Graph**: embeddings and relationship weights live in ChromaDB. Drift mitigation routines decay weights to prioritize fresh context.
- **Drift Analytics**: every controller run records latency, verdicts, mitigation plans, and SLO breaches. Access programmatically via `MemoryManager.list_drift_analytics()` or inspect in the GUI Drift Trends tab.
- **Prompt trust boundary**: retrieved working, episodic, semantic, relation, and review records are rendered as untrusted reference data and bounded to 6,000 characters; the current task instruction remains separate.
- **Observability**: extend `drm.metrics` and `drm.span` loggers for custom sinks. CLI logs retain execution metadata but omit prompts, results, feedback, and review text by default. Wrap external I/O with timeout-aware calls and surface actionable exception messages.

## CI & Automation
- GitHub Actions workflow `.github/workflows/ci.yml` runs `uv sync --all-extras --frozen` and then Ruff plus strict Pyright on each push/PR.
- Runtime dependencies are pinned in both `pyproject.toml` and `requirements.txt`; audit transitive dependencies regularly.
- Local CI-like verification can be run through `uv run --all-extras --frozen` after a frozen sync.

## Utilities
- **Memory Seeding**: run `python scripts/seed_memory.py` to populate demo working/episodic/semantic/review entries for GUI demos.
- **Manual Drift Review**: leverage the GUI "Memory Snapshot" and telemetry tabs to inspect the last five revisions and drift advisories.

## GUI Overview
- Workflow selector + task input wires into LiveTaskLoop executions without leaving the GUI.
- Background threads keep the UI responsive; status widgets stream live updates.
- Panels expose working/episodic/semantic/review slices, drift advisories, telemetry charts, and review history.
- Settings editor inside the GUI allows on-the-fly config adjustments that persist across sessions (window size, workflow preference, etc.).

## Roadmap
1. Harden real provider acceptance with redacted, case-specific evidence.
2. WebSocket exposure for telemetry feed to enable external dashboards.
3. CLI/webhook exports for drift analytics to plug into monitoring pipelines.
