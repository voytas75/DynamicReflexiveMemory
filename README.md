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

3. **Launch the app**

   ```bash
   python main.py --mode gui        # GUI (requires PySide6)
   python main.py --mode cli --task "Draft integration plan"  # CLI fallback
   ```

## Configuration

- `config/config.json` controls workflows, providers, and memory locations.
- `config/logging.conf` sets structured logging output.
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
