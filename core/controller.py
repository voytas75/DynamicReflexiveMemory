"""Self-adjustment logic and drift detection for DRM.

Updates: v0.1 - 2025-11-06 - Implemented feedback-driven controller with rolling
metrics and drift advisories.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Deque, Optional

from config.settings import AppConfig
from models.memory import ReviewRecord
from models.workflows import TaskResult, WorkflowSelection

LOGGER = logging.getLogger("drm.controller")


@dataclass(slots=True)
class PerformanceSnapshot:
    """Tracks performance metrics for the controller window."""

    latency: float
    verdict: str
    workflow: str


class SelfAdjustingController:
    """Monitors execution metrics and recommends adjustments."""

    def __init__(self, config: AppConfig, window_size: int = 10) -> None:
        self._config = config
        self._logger = LOGGER
        self._window_size = window_size
        self._history: Deque[PerformanceSnapshot] = deque(maxlen=window_size)

    def register_result(
        self,
        selection: WorkflowSelection,
        result: TaskResult,
        review: Optional[ReviewRecord] = None,
    ) -> Optional[str]:
        """Record the result and review, returning drift advisory text if needed."""
        verdict = review.verdict if review else "unknown"
        snapshot = PerformanceSnapshot(
            latency=result.latency_seconds,
            verdict=verdict,
            workflow=selection.workflow,
        )
        self._history.append(snapshot)
        self._logger.debug(
            "Registered performance snapshot: workflow=%s latency=%.2fs verdict=%s",
            snapshot.workflow,
            snapshot.latency,
            snapshot.verdict,
        )
        return self._assess_drift()

    def _assess_drift(self) -> Optional[str]:
        if len(self._history) < max(3, self._window_size // 2):
            return None

        latencies = [sample.latency for sample in self._history]
        average_latency = mean(latencies)
        high_latency = sum(lat > average_latency * 1.5 for lat in latencies)
        negative_reviews = sum(
            sample.verdict.lower().startswith(("fail", "reject"))
            for sample in self._history
        )

        if high_latency >= 3 or negative_reviews >= 2:
            advisory = (
                "Performance drift detected. "
                f"High latency count={high_latency}, negative reviews={negative_reviews}. "
                "Consider reweighting memory context or switching to reasoning workflow."
            )
            self._logger.warning(advisory)
            return advisory
        return None
