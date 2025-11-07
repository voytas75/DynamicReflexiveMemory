"""Tests for drift detection and bias adjustment behaviour."""

from __future__ import annotations

from uuid import uuid4

from config.settings import load_app_config
from core.controller import SelfAdjustingController
from models.memory import ReviewRecord
from models.workflows import TaskResult, WorkflowSelection


def test_controller_emits_drift_and_bias_updates() -> None:
    """High-latency failures should trigger drift advisories and bias shifts."""

    config = load_app_config()
    controller = SelfAdjustingController(config, window_size=4)

    selection = WorkflowSelection(workflow="fast", rationale="test", score=1.0)
    result = TaskResult(workflow="fast", content="", latency_seconds=8.0)

    review = ReviewRecord(
        id=str(uuid4()),
        task_reference="test-task",
        verdict="fail",
        notes="Simulated failure",
    )

    for _ in range(4):
        controller.register_result(selection, result, review)

    advisory = controller.last_advisory
    assert advisory is not None and "Performance drift" in advisory

    biases = controller.workflow_biases
    assert biases.get("fast", 0) < 0
    if "reasoning" in config.llm.workflows:
        assert biases.get("reasoning", 0) > 0

    plan = controller.last_plan
    assert plan
    assert "recommended_actions" in plan
    assert "slo_breaches" in plan


def test_controller_records_slo_breaches() -> None:
    """Controller should capture SLO breaches in the mitigation plan."""

    config = load_app_config()
    controller = SelfAdjustingController(
        config,
        window_size=3,
        latency_slo_seconds=0.1,
        quality_slo=0.95,
    )

    selection = WorkflowSelection(workflow="fast", rationale="test", score=1.0)
    result = TaskResult(workflow="fast", content="", latency_seconds=0.5)
    review = ReviewRecord(
        id=str(uuid4()),
        task_reference="task-slo",
        verdict="pass",
        notes="Quality below threshold",
        quality_score=0.5,
    )

    controller.register_result(selection, result, review)

    plan = controller.last_plan
    assert plan
    assert set(plan.get("slo_breaches", [])) == {"latency", "quality"}
    assert "expand_context_retrieval" in plan.get("recommended_actions", [])
