# Dynamic Reflexive Memory

Dynamic Reflexive Memory (DRM) is an adaptive memory substrate for LLM workflows that unifies short-term Redis state, long-term ChromaDB embeddings, and reflective review cycles to maintain continuity between reasoning sessions.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d
```

## Quick Start

```bash
# Set environment variables
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com"
export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-large"
export OLLAMA_BASE_URL="http://localhost:11434"

# Launch DRM (GUI or CLI)
python main.py --mode gui
python main.py --mode cli --task "Draft integration plan"
```

## Features

- Local-first Python runtime with optional PySide6 GUI dashboard.
- Configurable LiteLLM routing across fast, reasoning, and local workflows.
- Unified memory manager spanning Redis working memory and ChromaDB episodic/semantic stores with reflective review cycles.
- Drift analytics, telemetry panels, and self-adjustment heuristics that surface controller health and automate mitigation plans.

## Developer

- Read the full developer guide in [README-DEV.md](README-DEV.md) for environment details, .env guidance, and deep-dive documentation.
- Track release history in [CHANGELOG.md](CHANGELOG.md).

## License

Released under the [MIT License](LICENSE).
