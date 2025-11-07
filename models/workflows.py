"""Workflow-related data models for LLM task execution.

Updates:
    v0.1 - 2025-11-06 - Introduced task request/result models and workflow selection metadata.
    v0.2 - 2025-11-07 - Added TaskRunOutcome summary for live execution loop.
    v0.3 - 2025-11-07 - Adopted timezone-aware timestamps for workflow metadata.
    v0.4 - 2025-11-07 - Captured drift mitigation summaries in task outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

from uuid import uuid4

from models.memory import ReviewRecord


def _utcnow() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class TaskRequest:
    """Represents a user task routed through the DRM executor."""

    workflow: str
    prompt: str
    task_id: str = field(default_factory=lambda: str(uuid4()))
    context: Dict[str, object] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)


@dataclass(slots=True)
class TaskResult:
    """Holds the outcome of a completed LLM task."""

    workflow: str
    content: str
    latency_seconds: float
    metadata: Dict[str, object] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)


@dataclass(slots=True)
class WorkflowSelection:
    """Details the reasoning for choosing a particular workflow."""

    workflow: str
    rationale: str
    score: float
    timestamp: datetime = field(default_factory=_utcnow)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class TaskRunOutcome:
    """Summarises a single live task execution."""

    selection: WorkflowSelection
    request: TaskRequest
    result: TaskResult
    review: ReviewRecord
    drift_advisory: Optional[str] = None
    mitigation_summary: Optional[Dict[str, object]] = None
