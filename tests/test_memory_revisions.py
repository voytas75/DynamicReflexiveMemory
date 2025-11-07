"""Tests for revision logging behaviour in the memory manager."""

from __future__ import annotations

from config.settings import load_app_config
from core.memory_manager import MemoryManager
from models.memory import (
    EpisodicMemoryEntry,
    ReviewRecord,
    SemanticNode,
    WorkingMemoryItem,
)


def test_memory_revision_log_records_changes(monkeypatch, tmp_path) -> None:
    """Ensure all memory layers append to the revision log."""

    monkeypatch.setenv("DRM_MEMORY_LOG_PATH", str(tmp_path / "revisions.jsonl"))
    monkeypatch.setattr("core.memory_manager.redis_module", None)
    monkeypatch.setattr("core.memory_manager.chromadb_module", None)
    monkeypatch.setattr("core.memory_manager.chroma_embeddings_module", None)

    config = load_app_config()
    manager = MemoryManager(config)

    manager.put_working_item(
        WorkingMemoryItem(key="task:test", payload={"value": 1}, ttl_seconds=10)
    )
    manager.record_episodic(
        EpisodicMemoryEntry(
            id="episode-test",
            content="Integration test content.",
            metadata={"source": "pytest"},
        )
    )
    manager.record_semantic(
        SemanticNode(
            id="concept-test",
            label="Concept",
            definition="Definition",
            sources=["unit"],
        )
    )
    manager.record_review(
        ReviewRecord(id="review-test", task_reference="task:test", verdict="pass")
    )

    history = manager.get_revision_history(limit=10)
    assert len(history) >= 4
    layers = {entry.get("layer") for entry in history}
    assert {"working", "episodic", "semantic", "review"}.issubset(layers)
    assert (tmp_path / "revisions.jsonl").exists()
