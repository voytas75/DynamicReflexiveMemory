"""Helpers for loading and validating DRM configuration files.

Updates: v0.1 - 2025-11-06 - Added Pydantic-based loader for core configuration.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator

from core.exceptions import ConfigError

CONFIG_FILE = Path(__file__).with_name("config.json")


class WorkflowModelConfig(BaseModel):
    """Configuration for a single LLM workflow."""

    provider: str = Field(..., description="LLM provider label, e.g. azure or ollama.")
    model: str = Field(..., description="Model identifier understood by LiteLLM.")
    temperature: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for the workflow.",
    )


class WorkflowTimeoutConfig(BaseModel):
    """Timeout and retry configuration for workflow execution."""

    request_seconds: int = Field(..., gt=0)
    retry_attempts: int = Field(3, ge=0)
    retry_backoff_seconds: int = Field(2, ge=0)


class LLMConfig(BaseModel):
    """Global LLM routing configuration."""

    default_workflow: str = Field(..., min_length=1)
    workflows: Dict[str, WorkflowModelConfig]
    timeouts: WorkflowTimeoutConfig

    @model_validator(mode="after")
    def _ensure_default_present(self) -> "LLMConfig":
        if self.default_workflow not in self.workflows:
            raise ValueError(
                f"default_workflow '{self.default_workflow}' is not defined "
                "in workflows."
            )
        return self


class RedisConfig(BaseModel):
    """Redis-backed working memory configuration."""

    host: str = Field("localhost")
    port: int = Field(6379, ge=0)
    db: int = Field(0, ge=0)
    ttl_seconds: int = Field(900, ge=1)


class ChromaDBConfig(BaseModel):
    """ChromaDB long-term storage configuration."""

    persist_directory: str = Field(..., min_length=1)
    collection: str = Field("drm_memory", min_length=1)


class MemoryConfig(BaseModel):
    """Aggregated memory configuration."""

    redis: RedisConfig
    chromadb: ChromaDBConfig


class ReviewConfig(BaseModel):
    """Hybrid review workflow configuration."""

    enabled: bool = True
    auto_reviewer_model: Optional[str] = Field(
        default=None,
        description="Model identifier used for automated audits.",
    )


class TelemetryConfig(BaseModel):
    """Logging and telemetry configuration."""

    log_level: str = Field("INFO")


class EmbeddingConfig(BaseModel):
    """Vector embedding provider configuration."""

    provider: str = Field(..., description="Embedding provider identifier, e.g. azure.")
    model: str = Field(..., description="Embedding model or deployment name.")


class AppConfig(BaseModel):
    """Complete application configuration payload."""

    version: str
    llm: LLMConfig
    memory: MemoryConfig
    review: ReviewConfig
    embedding: Optional[EmbeddingConfig] = None
    telemetry: TelemetryConfig


def resolve_config_path(path: Optional[Path] = None) -> Path:
    """Resolve the configuration path, defaulting to the packaged config file."""
    resolved = path or CONFIG_FILE
    if not resolved.exists():
        raise ConfigError(f"Configuration file not found at {resolved}")
    return resolved


def load_app_config(path: Optional[Path] = None) -> AppConfig:
    """Load and validate the application configuration from JSON."""
    config_path = resolve_config_path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"Unable to read configuration: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in configuration: {exc}") from exc

    try:
        return AppConfig.model_validate(payload)
    except ValidationError as exc:
        raise ConfigError(f"Configuration validation failed: {exc}") from exc


@lru_cache(maxsize=1)
def get_app_config(path: Optional[Path] = None) -> AppConfig:
    """Memoised accessor for the application configuration."""
    return load_app_config(path)
