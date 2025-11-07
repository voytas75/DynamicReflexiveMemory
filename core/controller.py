"""Self-adjustment logic and drift detection for DRM.

Updates:
    v0.1 - 2025-11-06 - Implemented feedback-driven controller with rolling metrics
        and drift advisories.
    v0.2 - 2025-11-06 - Tracked last advisory for GUI display.
    v0.3 - 2025-11-06 - Added adaptive workflow bias adjustments derived from drift
        feedback.
    v0.4 - 2025-11-07 - Logged workflow bias snapshots when drift triggers.
    v0.5 - 2025-11-07 - Added SLO tracking and mitigation planning for drift responses.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Deque, Dict, List, Optional

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

    def __init__(
        self,
        config: AppConfig,
        window_size: int = 10,
        latency_slo_seconds: float = 5.0,
        quality_slo: float = 0.85,
    ) -> None:
        self._config = config
        self._logger = LOGGER
        self._window_size = window_size
        self._history: Deque[PerformanceSnapshot] = deque(maxlen=window_size)
        self._last_advisory: Optional[str] = None
        self._workflow_biases: Dict[str, float] = {}
        self._decay = 0.9
        self._latency_slo = max(0.1, latency_slo_seconds)
        self._quality_slo = quality_slo
        self._last_plan: Dict[str, object] = {}

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
        slo_breaches = self._evaluate_slos(result, review)
        advisory = self._assess_drift()
        self._update_biases(selection, result, review, advisory is not None)
        self._last_plan = self._build_mitigation_plan(
            selection,
            result,
            review,
            advisory,
            slo_breaches,
        )
        self._last_advisory = advisory
        return advisory

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
    def _evaluate_slos(
        self,
        result: TaskResult,
        review: Optional[ReviewRecord],
    ) -> List[str]:
        """Return a list of breached service objectives for the current run."""
        breaches: List[str] = []
        if result.latency_seconds > self._latency_slo:
            breaches.append("latency")
        if (
            review
            and review.quality_score is not None
            and review.quality_score < self._quality_slo
        ):
            breaches.append("quality")
        return breaches

    def _update_biases(
        self,
        selection: WorkflowSelection,
        result: TaskResult,
        review: Optional[ReviewRecord],
        drift_detected: bool,
    ) -> None:
        # decay existing adjustments so preferences settle over time
        for workflow, bias in list(self._workflow_biases.items()):
            new_bias = bias * self._decay
            if abs(new_bias) < 0.01:
                self._workflow_biases.pop(workflow)
            else:
                self._workflow_biases[workflow] = new_bias

        workflow = selection.workflow
        verdict = (review.verdict if review else "unknown").lower()
        latency = result.latency_seconds
        average_latency = (
            mean(sample.latency for sample in self._history)
            if self._history
            else latency
        )

        # Reward fast, positive runs slightly
        if verdict.startswith("pass") and latency <= average_latency * 1.1:
            self._bump_bias(workflow, 0.05)

        # Penalise failed runs
        if verdict.startswith("fail"):
            self._bump_bias(workflow, -0.25)

        # Encourage reasoning workflow when drift detected
        if drift_detected:
            self._bump_bias(workflow, -0.3)
            reasoning_workflow = self._find_workflow("reasoning")
            if reasoning_workflow and reasoning_workflow != workflow:
                self._bump_bias(reasoning_workflow, 0.25)
            LOGGER.info(
                "Drift detected; workflow biases updated: %s",
                self._workflow_biases,
            )

        # Promote alternating fast/reasoning usage when latency spikes
        if latency > average_latency * 1.4:
            fast_workflow = self._find_workflow("fast")
            reasoning_workflow = self._find_workflow("reasoning")
            target = reasoning_workflow if workflow == fast_workflow else fast_workflow
            if target:
                self._bump_bias(target, 0.1)
    def _build_mitigation_plan(
        self,
        selection: WorkflowSelection,
        result: TaskResult,
        review: Optional[ReviewRecord],
        advisory: Optional[str],
        slo_breaches: List[str],
    ) -> Dict[str, object]:
        plan: Dict[str, object] = {}
        if advisory:
            plan["drift_advisory"] = advisory
        if slo_breaches:
            plan["slo_breaches"] = slo_breaches

        recommended: List[str] = []
        if "latency" in slo_breaches:
            recommended.extend(
                [
                    "reroute_to_reasoning_workflow",
                    "increase_llm_timeout",
                ]
            )
        if "quality" in slo_breaches:
            recommended.append("expand_context_retrieval")
        if advisory:
            recommended.append("trigger_memory_mitigation")
        if result.latency_seconds > self._latency_slo * 1.5:
            recommended.append("reduce_fast_workflow_bias")
        if (
            review
            and review.verdict.lower().startswith(("fail", "reject"))
            and selection.workflow != self._find_workflow("reasoning")
        ):
            recommended.append("escalate_to_reasoning")

        if recommended:
            deduped = list(dict.fromkeys(recommended))
            plan["recommended_actions"] = deduped

        if self._history and (plan or slo_breaches or advisory or recommended):
            latencies = [sample.latency for sample in self._history]
            negatives = sum(
                sample.verdict.lower().startswith(("fail", "reject"))
                for sample in self._history
            )
            plan["window_metrics"] = {
                "latency_avg": round(mean(latencies), 3),
                "window_size": len(self._history),
                "negative_reviews": negatives,
            }

        return plan

    def _bump_bias(self, workflow: str, delta: float) -> None:
        current = self._workflow_biases.get(workflow, 0.0)
        updated = max(-1.0, min(1.0, current + delta))
        self._workflow_biases[workflow] = updated
        self._logger.debug(
            "Workflow bias updated: %s -> %.2f (delta %.2f)", workflow, updated, delta
        )

    def _find_workflow(self, hint: str) -> Optional[str]:
        hint_lower = hint.lower()
        for name in self._config.llm.workflows.keys():
            if name.lower() == hint_lower:
                return name
        return None

    @property
    def last_advisory(self) -> Optional[str]:
        """Return the most recent drift advisory, if any."""
        return self._last_advisory

    @property
    def workflow_biases(self) -> Dict[str, float]:
        """Current controller preference weights per workflow."""
        return dict(self._workflow_biases)

    @property
    def last_plan(self) -> Dict[str, object]:
        """Return the last mitigation plan issued by the controller."""
        return dict(self._last_plan)
