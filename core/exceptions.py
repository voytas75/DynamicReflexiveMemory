"""Custom exception hierarchy for the Dynamic Reflexive Memory system.

This module centralises the error types emitted across subsystems to keep
error handling explicit and consistent with the AGENTS directives.

Updates: v0.1 - 2025-11-06 - Defined DRM error hierarchy for configuration,
memory, workflow, and review concerns.
"""

from __future__ import annotations


class DRMError(Exception):
    """Base class for all DRM-specific errors."""


class ConfigError(DRMError):
    """Raised when application configuration is missing or invalid."""


class MemoryError(DRMError):
    """Raised when memory layer interactions fail."""


class WorkflowError(DRMError):
    """Raised when LLM workflow selection or execution fails."""


class ReviewError(DRMError):
    """Raised when automated or human review processes encounter issues."""

