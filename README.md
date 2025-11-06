# Prompt Manager — Dynamic Reflexive Memory Core

Dynamic Reflexive Memory (DRM) evolves the original Prompt Manager application
into an adaptive memory substrate for LLM workflows. The system blends short-
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
   - ChromaDB (port 8000) stores episodic, semantic, and review embeddings.

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

## Testing

```bash
pytest
```

## Memory Seeding

Populate demo data to exercise the GUI and review pipelines:

```bash
python scripts/seed_memory.py
```

The script injects a working-memory task plus episodic, semantic, and review
entries. Re-run to append additional samples.

## GUI Overview

The PySide6 dashboard displays:

- Live memory slices (working, episodic, semantic, review) pulled via
  `MemoryManager`.
- Latest drift advisory from the self-adjusting controller.
- Manual refresh button to inspect updates after CLI or automated runs.

## Roadmap

1. Implement Redis and ChromaDB integration tests with docker-compose fixtures.
2. Expand GUI to display live memory metrics and review history.
3. Add persistence and analytics for drift detection outcomes.
