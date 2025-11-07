"""Smoke tests for the CLI entry point."""

from __future__ import annotations

import logging

from config.settings import load_app_config
from core.user_settings import UserSettingsManager
from main import run_cli
from models.memory import ReviewRecord
from models.workflows import TaskRequest, TaskResult, TaskRunOutcome, WorkflowSelection


class _StubLoop:
    last_instance: "_StubLoop | None" = None

    def __init__(self, _config, user_settings=None) -> None:  # pragma: no cover - simple stub
        self.last_override = None
        type(self).last_instance = self

    def run_task(self, *, task: str, workflow_override=None, human_feedback=None):
        assert task == "demo"
        assert human_feedback == "note"
        self.last_override = workflow_override
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


def test_run_cli_prefers_saved_workflow(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("main.LiveTaskLoop", _StubLoop)
    config = load_app_config()

    settings_path = tmp_path / "settings.json"
    user_settings = UserSettingsManager(settings_path)
    user_settings.update(last_workflow="reasoning")

    run_cli(config, task="demo", human_feedback="note", user_settings=user_settings)

    assert _StubLoop.last_instance is not None
    assert _StubLoop.last_instance.last_override == "reasoning"
