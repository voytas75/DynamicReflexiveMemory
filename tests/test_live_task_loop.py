"""Integration-style tests for the live task loop with stubbed executors.

Updates:
    v0.1 - 2026-05-11 - Added drift, review failure, and hydration branch coverage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, cast

import pytest

from config.settings import load_app_config
from core.controller import SelfAdjustingController
from core.exceptions import MemoryError, ReviewError
from core.live_loop import LiveTaskLoop
from core.memory_manager import MemoryManager
from models.memory import EpisodicMemoryEntry, ReviewRecord, SemanticNode
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


class _FailingReviewEngine:
    def perform_review(
        self,
        request: TaskRequest,
        result: TaskResult,
        human_feedback: Optional[str] = None,
    ) -> ReviewRecord:
        raise ReviewError("review unavailable")


class _AdvisoryController:
    @property
    def last_advisory(self) -> Optional[str]:
        return None

    @property
    def last_plan(self) -> dict[str, object]:
        return {"slo_breaches": ["latency"], "action": "mitigate"}

    @property
    def workflow_biases(self) -> dict[str, float]:
        return {"fast": -0.1, "reasoning": 0.2}

    def register_result(
        self,
        selection: WorkflowSelection,
        result: TaskResult,
        review: ReviewRecord,
    ) -> str:
        return "Latency drift detected."


def test_live_task_loop_persists_memory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
        executor=cast(Any, _StubExecutor()),
        review_engine=cast(Any, _StubReviewEngine()),
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


def test_live_task_loop_persists_drift_advisory_and_mitigation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DRM_MEMORY_LOG_PATH", str(tmp_path / "revisions.jsonl"))
    monkeypatch.setattr("core.memory_manager.redis_module", None)
    monkeypatch.setattr("core.memory_manager.chromadb_module", None)
    monkeypatch.setattr("core.memory_manager.chroma_embeddings_module", None)

    config = load_app_config()
    memory_manager = MemoryManager(config)
    memory_manager.record_semantic(
        SemanticNode(
            id="existing",
            label="Existing",
            definition="Existing semantic node",
            sources=["fast"],
        )
    )
    loop = LiveTaskLoop(
        config,
        memory_manager=memory_manager,
        executor=cast(Any, _StubExecutor()),
        review_engine=cast(Any, _StubReviewEngine()),
        controller=cast(Any, _AdvisoryController()),
    )

    outcome = loop.run_task("Trigger drift path")

    assert outcome.drift_advisory == "Latency drift detected."
    assert outcome.mitigation_summary is not None
    assert outcome.mitigation_summary["action"] == "mitigate"
    assert any(
        item.key.endswith(":drift") for item in memory_manager.list_working_items()
    )
    analytics = memory_manager.list_drift_analytics()
    assert analytics[-1].slo_breaches == ("latency",)


def test_live_task_loop_reraises_review_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DRM_MEMORY_LOG_PATH", str(tmp_path / "revisions.jsonl"))
    monkeypatch.setattr("core.memory_manager.redis_module", None)
    monkeypatch.setattr("core.memory_manager.chromadb_module", None)
    monkeypatch.setattr("core.memory_manager.chroma_embeddings_module", None)

    config = load_app_config()
    loop = LiveTaskLoop(
        config,
        memory_manager=MemoryManager(config),
        executor=cast(Any, _StubExecutor()),
        review_engine=cast(Any, _FailingReviewEngine()),
        controller=SelfAdjustingController(config, window_size=3),
    )

    with pytest.raises(ReviewError, match="review unavailable"):
        loop.run_task("Fail review")


def test_live_task_loop_private_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DRM_MEMORY_LOG_PATH", str(tmp_path / "revisions.jsonl"))
    monkeypatch.setattr("core.memory_manager.redis_module", None)
    monkeypatch.setattr("core.memory_manager.chromadb_module", None)
    monkeypatch.setattr("core.memory_manager.chroma_embeddings_module", None)

    config = load_app_config()
    memory_manager = MemoryManager(config)
    loop = LiveTaskLoop(
        config,
        memory_manager=memory_manager,
        executor=cast(Any, _StubExecutor()),
        review_engine=cast(Any, _StubReviewEngine()),
        controller=SelfAdjustingController(config, window_size=3),
    )

    long_task = "x" * 90
    long_result = TaskResult(
        workflow="fast",
        content=("y" * 260) + "\nsecond line",
        latency_seconds=0.1,
    )
    assert loop._build_semantic_label(long_task).endswith("…")
    assert loop._build_semantic_definition(long_result).endswith("…")
    assert (
        loop._build_semantic_definition(
            TaskResult(workflow="fast", content="", latency_seconds=0.1)
        )
        == ""
    )

    memory_manager.record_episodic(
        EpisodicMemoryEntry(id="old", content="old", metadata={"timestamp": "bad"})
    )
    memory_manager.record_review(
        ReviewRecord(
            id="review",
            task_reference="task",
            verdict="pass",
            notes="ok",
            suggestions=["one"],
            quality_score=0.5,
            auto_verdict="pass",
        )
    )
    hydrated = loop._load_recent_reviews(limit=1)
    assert hydrated and hydrated[0].id == "review"
    assert loop._hydrate_review({"quality_score": "not-a-number"}) is not None

    assert loop._safe_layer_query("episodic", "", limit=1)
    assert loop._coerce_timestamp("not-a-date").tzinfo is not None


def test_live_task_loop_handles_memory_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DRM_MEMORY_LOG_PATH", str(tmp_path / "revisions.jsonl"))
    monkeypatch.setattr("core.memory_manager.redis_module", None)
    monkeypatch.setattr("core.memory_manager.chromadb_module", None)
    monkeypatch.setattr("core.memory_manager.chroma_embeddings_module", None)

    class FailingMemoryManager(MemoryManager):
        def record_review(self, record: ReviewRecord) -> None:
            raise MemoryError("review failed")

        def record_episodic(self, entry: EpisodicMemoryEntry) -> None:
            raise MemoryError("episodic failed")

        def put_working_item(self, item: Any) -> None:
            raise MemoryError("working failed")

        def list_layer(self, layer: str) -> list[dict[str, object]]:
            raise MemoryError("list failed")

        def query_layer(
            self, layer: str, query: str, limit: int = 5
        ) -> list[dict[str, object]]:
            raise MemoryError("query failed")

    config = load_app_config()
    loop = LiveTaskLoop(
        config,
        memory_manager=FailingMemoryManager(config),
        executor=cast(Any, _StubExecutor()),
        review_engine=cast(Any, _StubReviewEngine()),
        controller=SelfAdjustingController(config, window_size=3),
    )

    loop._persist_review(
        ReviewRecord(id="review", task_reference="task", verdict="pass")
    )
    loop._safe_record_episodic(
        EpisodicMemoryEntry(id="episode", content="content", metadata={})
    )
    loop._store_working_item("key", {"value": 1})
    assert loop._safe_layer_query("episodic", "query", limit=1) == []
    assert loop._safe_layer_slice("episodic", limit=1) == []
    assert loop._load_recent_reviews(limit=1) == []
