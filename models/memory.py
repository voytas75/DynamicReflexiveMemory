"""Typed data structures representing DRM memory layers.

Updates:
    v0.1 - 2025-11-06 - Added dataclasses for working, episodic, semantic, and review memory records.
    v0.2 - 2025-11-07 - Added structured review fields for automated audit parsing.
    v0.3 - 2025-11-07 - Switched to timezone-aware timestamps across memory records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


def _utcnow() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class WorkingMemoryItem:
    """Represents a short-lived item stored in Redis-backed working memory."""

    key: str
    payload: Dict[str, object]
    ttl_seconds: int
    created_at: datetime = field(default_factory=_utcnow)


@dataclass(slots=True)
class EpisodicMemoryEntry:
    """Captured experience intended for sequential recall."""

    id: str
    content: str
    metadata: Dict[str, object]
    timestamp: datetime = field(default_factory=_utcnow)


@dataclass(slots=True)
class SemanticNode:
    """Graph node describing a concept in semantic memory."""

    id: str
    label: str
    definition: str
    sources: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=_utcnow)
    relations: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ReviewRecord:
    """Outcome of an automated or human review cycle."""

    id: str
    task_reference: str
    verdict: str
    notes: Optional[str] = None
    quality_score: Optional[float] = None
    suggestions: List[str] = field(default_factory=list)
    auto_verdict: Optional[str] = None
    created_at: datetime = field(default_factory=_utcnow)
