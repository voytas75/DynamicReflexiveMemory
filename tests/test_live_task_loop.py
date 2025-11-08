"""Integration-style tests for the live task loop with stubbed executors."""

from __future__ import annotations

from typing import Optional

from config.settings import load_app_config
from core.controller import SelfAdjustingController
from core.live_loop import LiveTaskLoop
from core.memory_manager import MemoryManager
from models.memory import ReviewRecord
from models.workflows import TaskRequest, TaskResult, WorkflowSelection


class _StubExecutor:
    def __init__(self) -> None:
        self._selection = WorkflowSelection(
            workflow="fast",
            rationale="stub executor",
            score=1.0,
        )

    def select_workflow(self, requested: Optional[str] = None) -> WorkflowSelection:
        return self._selection

    def execute(self, request: TaskRequest) -> TaskResult:
        return TaskResult(
            workflow=request.workflow,
            content="Stub outcome\nActionable summary",
            latency_seconds=0.05,
            metadata={"attempts": 1, "stub": True},
        )


class _StubReviewEngine:
    def perform_review(
        self,
        request: TaskRequest,
        result: TaskResult,
        human_feedback: Optional[str] = None,
    ) -> ReviewRecord:
        return ReviewRecord(
            id="review-stub",
            task_reference=request.task_id,
            verdict="pass",
            notes="Stub review",
            suggestions=["Keep stubbing"],
            quality_score=0.9,
            auto_verdict="pass",
        )


def test_live_task_loop_persists_memory(monkeypatch, tmp_path) -> None:
    """Running the live loop should persist artefacts and log revisions."""

    monkeypatch.setenv("DRM_MEMORY_LOG_PATH", str(tmp_path / "revisions.jsonl"))
    monkeypatch.setattr("core.memory_manager.redis_module", None)
    monkeypatch.setattr("core.memory_manager.chromadb_module", None)
    monkeypatch.setattr("core.memory_manager.chroma_embeddings_module", None)

    config = load_app_config()
    memory_manager = MemoryManager(config)
    controller = SelfAdjustingController(config, window_size=3)

    loop = LiveTaskLoop(
        config,
        memory_manager=memory_manager,
        executor=_StubExecutor(),
        review_engine=_StubReviewEngine(),
        controller=controller,
    )

    outcome = loop.run_task("Summarise integration behaviour")

    assert outcome.result.content.startswith("Stub outcome")
    assert outcome.review.verdict == "pass"
    assert outcome.drift_advisory is None
    assert outcome.mitigation_summary is None

    episodic = memory_manager.list_layer("episodic")
    semantic = memory_manager.list_layer("semantic")
    reviews = memory_manager.list_layer("review")
    analytics = memory_manager.list_layer("analytics")

    assert episodic and semantic and reviews and analytics
    assert any(item.get("workflow") == "fast" for item in analytics)

    analytics_records = memory_manager.list_drift_analytics()
    assert analytics_records
    assert analytics_records[-1].workflow == "fast"

    history = memory_manager.get_revision_history(limit=10)
    assert history
    layers = {entry.get("layer") for entry in history}
    assert {"episodic", "review", "analytics"}.issubset(layers)
