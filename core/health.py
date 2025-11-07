"""Startup diagnostics for Dynamic Reflexive Memory external dependencies.

Updates:
    v0.1 - 2025-11-06 - Added runtime checks for Redis, ChromaDB, litellm, and
        credential prerequisites with informative warnings.
    v0.2 - 2025-11-07 - Treated configured Azure embedding deployments as valid.
"""

from __future__ import annotations

import logging
import os
from importlib import metadata
from pathlib import Path
from typing import Any, List, cast

from config.settings import AppConfig, EmbeddingConfig
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
        import redis as redis_module
    except ImportError:
        return [
            "redis client not installed; using in-memory working memory fallback.",
        ]

    redis_cfg = config.memory.redis
    client = cast(Any, redis_module).Redis(
        host=redis_cfg.host,
        port=redis_cfg.port,
        db=redis_cfg.db,
        socket_timeout=1,
    )
    try:
        client.ping()
    except Exception as exc:  # pragma: no cover - runtime dependent
        LOGGER.warning(
            "Redis connectivity check failed (%s); using in-memory fallback.", exc
        )
        return [
            "redis unavailable; using in-memory working memory fallback.",
        ]
    return []


def _check_chromadb(config: AppConfig) -> List[str]:
    try:
        import chromadb as chromadb_module
    except ImportError:
        return [
            "chromadb not installed; long-term memory will not persist between runs.",
        ]

    chroma_cfg = config.memory.chromadb
    persist_dir = Path(chroma_cfg.persist_directory)
    try:
        persist_dir.mkdir(parents=True, exist_ok=True)
        client = cast(Any, chromadb_module).PersistentClient(path=str(persist_dir))
        client.list_collections()
    except Exception as exc:  # pragma: no cover - runtime dependent
        LOGGER.warning(
            "ChromaDB availability check failed (%s); using in-memory memory store.",
            exc,
        )
        return [
            "chromadb unavailable; in-memory persistence activated.",
        ]
    return []


def _check_litellm() -> List[str]:
    try:
        import litellm  # noqa: F401
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
        if not _is_azure_embedding_configured(embedding):
            warnings.append(
                "Azure embedding deployment not set; defaulting to model identifier."
            )

    return warnings


_OPENAI_STANDARD_EMBEDDING_MODELS = {
    "text-embedding-3-large",
    "text-embedding-3-small",
}


def _is_azure_embedding_configured(embedding: EmbeddingConfig) -> bool:
    """Return True when an Azure embedding deployment name is configured."""
    if os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"):
        return True

    model_name = embedding.model.strip()
    if not model_name:
        return False

    return model_name.lower() not in _OPENAI_STANDARD_EMBEDDING_MODELS
