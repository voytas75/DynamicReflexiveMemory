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

   - Redis (port 6379) for working memory.
   - Optional: ChromaDB server if not using the embedded client.

3. **Launch the app**

   ```bash
   python main.py --mode gui        # GUI (requires PySide6)
   python main.py --mode cli --task "Draft integration plan"  # CLI fallback
   ```

## Configuration

- `config/config.json` controls workflows, providers, and memory locations.
- `config/logging.conf` sets structured logging output.
- Adjust the LiteLLM provider credentials via environment variables expected by
  LiteLLM (e.g., `AZURE_OPENAI_API_KEY`).

## Testing

```bash
pytest
```

## Roadmap

1. Implement Redis and ChromaDB integration tests with docker-compose fixtures.
2. Expand GUI to display live memory metrics and review history.
3. Add persistence and analytics for drift detection outcomes.

