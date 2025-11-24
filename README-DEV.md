# Dynamic Reflexive Memory – Developer Guide

## Current Status
- Primary entry point is `main.py`, which launches the PySide6 GUI by default and falls back to CLI mode automatically in headless environments.
- Data layer combines Redis (working memory), ChromaDB (episodic/semantic stores), and persisted review cycles handed off to the GUI telemetry panels.
- Strict typing enforced through Pydantic models, dataclasses, and pyright strict mode for core modules.
- Local runtime first; external providers are optional and configured through environment variables or `config/config.json`.

## Environment & Tooling
- Target interpreter: **Python 3.12**; create a dedicated virtual environment per workspace.
- Formatting: `black` (line length 88) + `isort`; linting via `ruff --fix`.
- Type checking: `pyright --strict` (or `npx pyright` if installed globally) before publishing changes.
- Testing stack: `pytest`, `pytest-asyncio`, `pytest-cov`, `hypothesis`, and `vcrpy` for HTTP fixtures.

## Detailed Getting Started
1. **Create and activate the virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
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
   python main.py --mode gui

   # CLI/task runner
   python main.py --mode cli --task "Draft integration plan" --feedback "Looks good"
   ```

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

## Testing & Quality Gates
- Install dev requirements and run the full gate before merging:
  ```bash
  pytest -n auto --cov=src --cov-fail-under=85 --disable-warnings -q
  pyright
  ruff check --fix
  black .
  ```
- Favor `hypothesis` strategies for boundary inputs (long prompts, Unicode edge cases, malformed JSON payloads).
- Mock outbound HTTP/database calls with `pytest-mock`, `vcrpy`, or async test clients to keep suites deterministic.
- Keep coverage ≥90% on core logic modules; justify exceptions in PR descriptions if temporary.

## Memory & Telemetry
- **Memory Revision Log**: mutations append to `data/logs/memory_revisions.jsonl`; use `DRM_MEMORY_LOG_PATH` to override for CI. Treat the log as append-only for auditability.
- **Semantic Graph**: embeddings and relationship weights live in ChromaDB. Drift mitigation routines decay weights to prioritize fresh context.
- **Drift Analytics**: every controller run records latency, verdicts, mitigation plans, and SLO breaches. Access programmatically via `MemoryManager.list_drift_analytics()` or inspect in the GUI Drift Trends tab.
- **Observability**: extend `drm.metrics` and `drm.span` loggers for custom sinks. Wrap external I/O with timeout-aware calls and surface actionable exception messages.

## CI & Automation
- GitHub Actions workflow `.github/workflows/ci.yml` runs `pytest`, coverage, and `mypy/pyright` on each push/PR; extend with lint checks as needed.
- Prefer `uv pip compile` / lockfiles for dependency pinning; audit transitive dependencies regularly.

## Utilities
- **Memory Seeding**: run `python scripts/seed_memory.py` to populate demo working/episodic/semantic/review entries for GUI demos.
- **Manual Drift Review**: leverage the GUI "Memory Snapshot" and telemetry tabs to inspect the last five revisions and drift advisories.

## GUI Overview
- Workflow selector + task input wires into LiveTaskLoop executions without leaving the GUI.
- Background threads keep the UI responsive; status widgets stream live updates.
- Panels expose working/episodic/semantic/review slices, drift advisories, telemetry charts, and review history.
- Settings editor inside the GUI allows on-the-fly config adjustments that persist across sessions (window size, workflow preference, etc.).

## Roadmap
1. Redis integration tests using docker-compose fixtures.
2. WebSocket exposure for telemetry feed to enable external dashboards.
3. CLI/webhook exports for drift analytics to plug into monitoring pipelines.
