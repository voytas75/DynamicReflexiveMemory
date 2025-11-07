# Dynamic Reflexive Memory Core

Dynamic Reflexive Memory (DRM) evolves into an adaptive memory substrate for LLM workflows. The system blends short-
term Redis state, long-term ChromaDB embeddings, and reflective review cycles
to deliver continuity between reasoning sessions.

## Features

- Local-first Python runtime with optional PySide6 GUI dashboard.
- Configurable LiteLLM routing across `fast`, `reasoning`, and `local` workflows.
- Unified memory manager with Redis working memory and ChromaDB episodic/
  semantic stores.
- Hybrid automated/human review loop feeding self-adjustment heuristics.
- Strictly typed configuration and models via Pydantic and dataclasses.
- Append-only memory revision log for audit trails and rollback instrumentation.
- Remembers the most recent workflow preference and GUI window size between sessions.

## Getting Started

The project targets **Python 3.12**. Create a virtual environment with that interpreter before installing dependencies.

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run services**

   ```bash
   docker compose up -d
   ```

   - Redis (port 6379) persists working memory.
   - ChromaDB persists via the Python package; no container is required.
   - Ollama (port 11434) is available for local inference; pull models with `ollama pull <model>` and set `OLLAMA_BASE_URL` when using the `local` workflow.

3. **Launch the app**

   ```bash
   python main.py --mode gui        # GUI (requires PySide6)
   python main.py --mode cli --task "Draft integration plan" --feedback "Looks good"  # CLI fallback with optional human review
   ```

   The runner detects headless environments automatically and will fall back to
   CLI mode when no graphical backend is available.

## Configuration

- `config/config.json` controls workflows, providers, and memory locations.
- `config/config.example.json` provides a starter template; copy it to `config/config.json` and adjust credentials before first run.
- `config/logging.conf` sets structured logging output.
- Environment variables are loaded automatically from a local `.env` file (via `python-dotenv`) if present.
- Provider credentials:
  - Azure OpenAI: set `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`,
    and optionally `AZURE_OPENAI_API_VERSION`.
  - Ollama: override `OLLAMA_BASE_URL` when running a remote instance.
  - Missing credentials cause a descriptive `WorkflowError` to surface.
- Enable LiteLLM debugging by setting `llm.enable_debug` to `true` in the
  configuration when deeper request/response tracing is needed.
- Automated review can use a distinct provider by setting
  `review.auto_reviewer_provider`; if omitted it inherits the default workflow's
  provider.
- Embeddings (default: Azure OpenAI `text-embedding-3-large`):
  - Set `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` if your embedding deployment name
    differs from the model identifier.

## Testing

```bash
pytest
```

The suite includes integration coverage for the live task loop, controller drift
logic, and memory revision logging. Use `pytest -k "live_task_loop"` to focus on
the orchestration path or `pytest -k revisions` to inspect the logging tests.

## Memory Revision Log

- Memory mutations are appended to `data/logs/memory_revisions.jsonl` by default.
- Override the location with `DRM_MEMORY_LOG_PATH=/custom/revisions.jsonl` for
  ephemeral runs or CI pipelines.
- The GUI "Memory Snapshot" panel surfaces the five most recent revisions
  alongside other telemetry for quick inspection.

## Semantic Graph

- Semantic concepts automatically link to recent nodes, and the prompt engine
  surfaces their strongest relationships for context-rich task execution.
- Drift mitigation routines decay relation weights and prune stale working
  memory whenever the controller detects performance issues, keeping the active
  context focused.

## Observability

- Memory operations emit metrics via the `drm.metrics` logger and structured
  span events via `drm.span`. Extend `config/logging.conf` to route these
  channels to your preferred observability stack.

## CI & Quality Checks

- `pytest` executes the regression suite; GitHub Actions workflow `.github/workflows/ci.yml` runs it on every push/PR.
- `mypy .` enforces the strict type configuration defined in `pyproject.toml`.
- Extend the workflow with additional linters (e.g., `ruff`, `black`) as the codebase grows.

## Memory Seeding

Populate demo data to exercise the GUI and review pipelines:

```bash
python scripts/seed_memory.py
```

The script injects a working-memory task plus episodic, semantic, and review
entries. Re-run to append additional samples.

## GUI Overview

The PySide6 dashboard displays:

- Workflow selector and task input wired to the LiveTaskLoop for launching runs without leaving the GUI.
- Background task execution keeps the interface responsive while runs progress, showing live status updates.
- Live memory slices (working, episodic, semantic, review) pulled via
  `MemoryManager`.
- Latest drift advisory from the self-adjusting controller plus history pulled
  from working memory keys.
- Tabbed panels for recent outputs and review history so you can inspect results and audits side-by-side.
- Manual refresh button to inspect updates after CLI or automated runs.

## Roadmap

1. Implement Redis integration tests with docker-compose fixtures.
2. Expand GUI to display live memory metrics and review history.
3. Add persistence and analytics for drift detection outcomes.
