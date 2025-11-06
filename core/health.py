"""Startup diagnostics for Dynamic Reflexive Memory external dependencies.

Updates:
    v0.1 - 2025-11-06 - Added runtime checks for Redis, ChromaDB, litellm, and
        credential prerequisites with informative warnings.
"""

from __future__ import annotations

import logging
import os
from importlib import metadata
from pathlib import Path
from typing import List

from config.settings import AppConfig
from core.exceptions import HealthCheckError


LOGGER = logging.getLogger("drm.health")


def run_startup_checks(config: AppConfig) -> List[str]:
    """Validate external dependencies; return warnings when falling back."""

    warnings: List[str] = []
    warnings.extend(_check_redis(config))
    warnings.extend(_check_chromadb(config))
    warnings.extend(_check_litellm())
    warnings.extend(_check_credentials(config))
    return warnings


def _check_redis(config: AppConfig) -> List[str]:
    try:
        import redis  # type: ignore
    except ImportError:
        return [
            "redis client not installed; using in-memory working memory fallback.",
        ]

    redis_cfg = config.memory.redis
    client = redis.Redis(  # type: ignore[attr-defined]
        host=redis_cfg.host,
        port=redis_cfg.port,
        db=redis_cfg.db,
        socket_timeout=1,
    )
    try:
        client.ping()
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise HealthCheckError(f"Redis connectivity check failed: {exc}") from exc
    return []


def _check_chromadb(config: AppConfig) -> List[str]:
    try:
        import chromadb  # type: ignore
    except ImportError:
        return [
            "chromadb not installed; long-term memory will not persist between runs.",
        ]

    chroma_cfg = config.memory.chromadb
    persist_dir = Path(chroma_cfg.persist_directory)
    try:
        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(persist_dir))  # type: ignore[attr-defined]
        client.list_collections()
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise HealthCheckError(f"ChromaDB availability check failed: {exc}") from exc
    return []


def _check_litellm() -> List[str]:
    try:
        import litellm  # noqa: F401  # type: ignore
    except ImportError:
        raise HealthCheckError("litellm is required for workflow execution but is missing.")

    try:
        installed = metadata.version("litellm")
    except metadata.PackageNotFoundError:
        return []

    expected = "1.61.15"
    if installed != expected:
        return [
            f"litellm version {installed} detected; expected {expected}."
        ]
    return []


def _check_credentials(config: AppConfig) -> List[str]:
    warnings: List[str] = []
    azure_workflows = [
        name
        for name, workflow in config.llm.workflows.items()
        if workflow.provider.lower() == "azure"
    ]
    if azure_workflows:
        missing = [
            var
            for var in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT")
            if not os.getenv(var)
        ]
        if missing:
            warnings.append(
                "Azure OpenAI credentials missing: " + ", ".join(missing)
            )

    embedding = config.embedding
    if embedding and embedding.provider.lower() == "azure":
        if not os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"):
            warnings.append(
                "Azure embedding deployment not set; defaulting to model identifier."
            )

    return warnings
