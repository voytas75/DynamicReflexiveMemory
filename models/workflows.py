"""Workflow-related data models for LLM task execution.

Updates: v0.1 - 2025-11-06 - Introduced task request/result models and
workflow selection metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from uuid import uuid4


@dataclass(slots=True)
class TaskRequest:
    """Represents a user task routed through the DRM executor."""

    workflow: str
    prompt: str
    task_id: str = field(default_factory=lambda: str(uuid4()))
    context: Dict[str, object] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class TaskResult:
    """Holds the outcome of a completed LLM task."""

    workflow: str
    content: str
    latency_seconds: float
    metadata: Dict[str, object] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class WorkflowSelection:
    """Details the reasoning for choosing a particular workflow."""

    workflow: str
    rationale: str
    score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, object] = field(default_factory=dict)
