"""Smoke tests for the CLI entry point."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from config.settings import load_app_config, resolve_config_path
from core.user_settings import UserSettingsManager
from main import run_cli
from models.memory import ReviewRecord
from models.workflows import TaskRequest, TaskResult, TaskRunOutcome, WorkflowSelection


class _StubLoop:
    last_instance: "_StubLoop | None" = None

    def __init__(
        self,
        _config: object,
        user_settings: object = None,
    ) -> None:  # pragma: no cover - simple stub
        self.last_override: str | None = None
        type(self).last_instance = self

    def run_task(
        self,
        *,
        task: str,
        workflow_override: str | None = None,
        human_feedback: str | None = None,
    ) -> TaskRunOutcome:
        assert task == "demo"
        assert human_feedback in {"note", "sensitive-human-feedback"}
        self.last_override = workflow_override
        selection = WorkflowSelection(workflow="fast", rationale="stub", score=1.0)
        request = TaskRequest(workflow="fast", prompt="sensitive-compiled-prompt")
        result = TaskResult(
            workflow="fast", content="sensitive-task-result", latency_seconds=0.1
        )
        review = ReviewRecord(
            id="review",
            task_reference=request.task_id,
            verdict="pass",
            notes="sensitive-review-notes",
            suggestions=["sensitive-review-suggestion"],
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


def test_run_cli_emits_feedback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr("main.LiveTaskLoop", _StubLoop)
    config = load_app_config(resolve_config_path(Path("config/config.example.json")))

    caplog.set_level(logging.INFO, logger="drm.cli")

    run_cli(
        config,
        task="demo",
        workflow=None,
        human_feedback="sensitive-human-feedback",
    )

    assert "Human feedback recorded" in caplog.text
    assert "Mitigation actions" in caplog.text
    for secret in (
        "sensitive-compiled-prompt",
        "sensitive-task-result",
        "sensitive-review-notes",
        "sensitive-review-suggestion",
        "sensitive-human-feedback",
    ):
        assert secret not in caplog.text


def test_run_cli_prefers_saved_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("main.LiveTaskLoop", _StubLoop)
    config = load_app_config(resolve_config_path(Path("config/config.example.json")))

    settings_path = tmp_path / "settings.json"
    user_settings = UserSettingsManager(settings_path)
    user_settings.update(last_workflow="reasoning")

    run_cli(config, task="demo", human_feedback="note", user_settings=user_settings)

    assert _StubLoop.last_instance is not None
    assert _StubLoop.last_instance.last_override == "reasoning"
