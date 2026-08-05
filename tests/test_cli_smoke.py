"""Smoke tests for the CLI entry point."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from config.settings import load_app_config, resolve_config_path
from core.exceptions import WorkflowError
from core.user_settings import UserSettingsManager
from main import main, run_cli
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


class _FailingLoop:
    """Stub a workflow failure without contacting a provider."""

    def __init__(self, _config: object, user_settings: object = None) -> None:
        del user_settings

    def run_task(
        self,
        *,
        task: str,
        workflow_override: str | None = None,
        human_feedback: str | None = None,
    ) -> TaskRunOutcome:
        del task, workflow_override, human_feedback
        raise WorkflowError("provider detail must not reach the terminal")


def test_run_cli_emits_feedback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr("main.LiveTaskLoop", _StubLoop)
    config = load_app_config(resolve_config_path(Path("config/config.example.json")))

    caplog.set_level(logging.INFO, logger="drm.cli")

    exit_code = run_cli(
        config,
        task="demo",
        workflow=None,
        human_feedback="sensitive-human-feedback",
    )

    assert exit_code == 0
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

    assert (
        run_cli(
            config,
            task="demo",
            human_feedback="note",
            user_settings=user_settings,
        )
        == 0
    )

    assert _StubLoop.last_instance is not None
    assert _StubLoop.last_instance.last_override == "reasoning"


def test_run_cli_returns_nonzero_for_workflow_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr("main.LiveTaskLoop", _FailingLoop)
    config = load_app_config(resolve_config_path(Path("config/config.example.json")))

    caplog.set_level(logging.INFO, logger="drm.cli")

    assert run_cli(config, task="demo", human_feedback="note") == 1
    assert "provider detail must not reach the terminal" not in caplog.text


def test_run_cli_prints_result_only_with_explicit_flag(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("main.LiveTaskLoop", _StubLoop)
    config = load_app_config(resolve_config_path(Path("config/config.example.json")))

    assert run_cli(config, task="demo", human_feedback="note") == 0
    assert capsys.readouterr().out == ""

    assert run_cli(config, task="demo", human_feedback="note", show_result=True) == 0

    assert capsys.readouterr().out == "sensitive-task-result\n"


def test_main_propagates_cli_exit_code_and_show_result_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr("main.resolve_config_path", lambda: Path("config/config.json"))
    monkeypatch.setattr("main.get_app_config", lambda _path: config)
    monkeypatch.setattr("main.setup_logging", lambda _path: None)
    monkeypatch.setattr("main.UserSettingsManager", lambda: object())
    monkeypatch.setattr("main.run_startup_checks", lambda _config: [])

    def _run_cli(*args: object, **kwargs: object) -> int:
        del args
        captured.update(kwargs)
        return 1

    monkeypatch.setattr("main.run_cli", _run_cli)

    assert main(["--mode", "cli", "--show-result"]) == 1
    assert captured["show_result"] is True
