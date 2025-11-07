"""Tests for revision logging behaviour in the memory manager."""

from __future__ import annotations

import json

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


def test_revision_log_verification_and_replay(monkeypatch, tmp_path) -> None:
    """Revision log should expose verification and replay helpers."""

    log_path = tmp_path / "revisions.jsonl"
    monkeypatch.setenv("DRM_MEMORY_LOG_PATH", str(log_path))
    monkeypatch.setattr("core.memory_manager.redis_module", None)
    monkeypatch.setattr("core.memory_manager.chromadb_module", None)
    monkeypatch.setattr("core.memory_manager.chroma_embeddings_module", None)

    config = load_app_config()
    manager = MemoryManager(config)

    manager.record_episodic(
        EpisodicMemoryEntry(
            id="episode-replay",
            content="Drift mitigation retrospective.",
            metadata={"topic": "drift"},
        )
    )
    manager.record_review(
        ReviewRecord(
            id="review-replay",
            task_reference="episode-replay",
            verdict="pass",
            quality_score=0.91,
        )
    )

    assert manager.verify_revision_log()

    episodic_state = manager.replay_revision_state("episodic")
    assert any(entry.get("id") == "episode-replay" for entry in episodic_state)

    # Corrupt the ledger tail to confirm verification fails.
    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records
    records[-1]["hash"] = "0" * 64
    log_path.write_text(
        "\n".join(json.dumps(record, default=str) for record in records) + "\n",
        encoding="utf-8",
    )
    assert not manager.verify_revision_log()


def test_query_layer_prefers_relevant_results(monkeypatch, tmp_path) -> None:
    """Querying episodic memory should surface the most relevant entries."""

    monkeypatch.setenv("DRM_MEMORY_LOG_PATH", str(tmp_path / "revisions.jsonl"))
    monkeypatch.setattr("core.memory_manager.redis_module", None)
    monkeypatch.setattr("core.memory_manager.chromadb_module", None)
    monkeypatch.setattr("core.memory_manager.chroma_embeddings_module", None)

    config = load_app_config()
    manager = MemoryManager(config)

    manager.record_episodic(
        EpisodicMemoryEntry(
            id="episode-mitigation",
            content="Review drift mitigation plan alignment.",
            metadata={"tags": ["drift", "mitigation"]},
        )
    )
    manager.record_episodic(
        EpisodicMemoryEntry(
            id="episode-ui",
            content="Updated GUI styling for telemetry panel.",
            metadata={"tags": ["ui"]},
        )
    )

    results = manager.query_layer("episodic", "drift mitigation plan", limit=1)
    assert results
    top_hit = results[0]
    assert top_hit.get("id") == "episode-mitigation"
