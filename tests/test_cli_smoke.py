"""Smoke tests for the CLI entry point."""

from __future__ import annotations

import logging

from config.settings import load_app_config
from main import run_cli
from models.memory import ReviewRecord
from models.workflows import TaskRequest, TaskResult, TaskRunOutcome, WorkflowSelection


class _StubLoop:
    def __init__(self, _config) -> None:  # pragma: no cover - simple stub
        pass

    def run_task(self, *, task: str, workflow_override=None, human_feedback=None):
        assert task == "demo"
        assert human_feedback == "note"
        selection = WorkflowSelection(workflow="fast", rationale="stub", score=1.0)
        request = TaskRequest(workflow="fast", prompt=task)
        result = TaskResult(workflow="fast", content="ok", latency_seconds=0.1)
        review = ReviewRecord(
            id="review",
            task_reference=request.task_id,
            verdict="pass",
            notes="stub",
            suggestions=["none"],
            quality_score=0.9,
            auto_verdict="pass",
        )
        return TaskRunOutcome(
            selection=selection,
            request=request,
            result=result,
            review=review,
            drift_advisory="watch drift",
            mitigation_summary={"working_pruned": 2},
        )


def test_run_cli_emits_feedback(monkeypatch, caplog) -> None:
    monkeypatch.setattr("main.LiveTaskLoop", _StubLoop)
    config = load_app_config()

    caplog.set_level(logging.INFO, logger="drm.cli")

    run_cli(config, task="demo", workflow=None, human_feedback="note")

    assert "Human feedback applied" in caplog.text
    assert "Mitigation actions" in caplog.text
