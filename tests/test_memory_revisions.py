"""Tests for revision logging and mitigation behaviour in the memory manager.

Updates:
    v0.1 - 2026-05-11 - Added mitigation, analytics hydration, and metric snapshot coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.settings import load_app_config
from core.exceptions import MemoryError
from core.memory_manager import MemoryManager
from models.memory import (
    DriftAnalyticsRecord,
    EpisodicMemoryEntry,
    ReviewRecord,
    SemanticNode,
    WorkingMemoryItem,
)


def test_memory_revision_log_records_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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


def test_revision_log_verification_and_replay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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


def test_query_layer_prefers_relevant_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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


def test_memory_mitigation_prunes_and_decays(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DRM_MEMORY_LOG_PATH", str(tmp_path / "revisions.jsonl"))
    monkeypatch.setattr("core.memory_manager.redis_module", None)
    monkeypatch.setattr("core.memory_manager.chromadb_module", None)
    monkeypatch.setattr("core.memory_manager.chroma_embeddings_module", None)

    manager = MemoryManager(load_app_config())
    for index in range(3):
        manager.put_working_item(
            WorkingMemoryItem(
                key=f"task:{index}",
                payload={"index": index},
                ttl_seconds=10,
            )
        )

    manager.record_semantic(
        SemanticNode(
            id="node-a",
            label="A",
            definition="First node",
            relations={"node:node-b": 0.8, "external": 0.5},
        )
    )
    manager.record_semantic(
        SemanticNode(
            id="node-b",
            label="B",
            definition="Second node",
            relations={"node:node-a": 0.02},
        )
    )

    summary = manager.apply_drift_mitigation(
        task_id="task:test",
        max_working_items=1,
        relation_decay=0.5,
    )

    assert summary["working_pruned"] == 2
    assert summary["semantic_nodes_updated"] == 2
    assert len(manager.list_working_items()) == 1
    node_a = manager.get_semantic_node("node-a")
    assert node_a is not None
    assert node_a.relations["node:node-b"] == pytest.approx(0.4)
    node_b = manager.get_semantic_node("node-b")
    assert node_b is not None
    assert "node:node-a" not in node_b.relations


def test_memory_manager_validates_layers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DRM_MEMORY_LOG_PATH", str(tmp_path / "revisions.jsonl"))
    monkeypatch.setattr("core.memory_manager.redis_module", None)
    monkeypatch.setattr("core.memory_manager.chromadb_module", None)
    monkeypatch.setattr("core.memory_manager.chroma_embeddings_module", None)

    manager = MemoryManager(load_app_config())

    with pytest.raises(MemoryError, match="Unsupported memory layer"):
        manager.list_layer("unknown")
    with pytest.raises(MemoryError, match="Unsupported memory layer"):
        manager.query_layer("unknown", "query")
    with pytest.raises(MemoryError, match="Unsupported revision replay layer"):
        manager.replay_revision_state("unknown")


def test_drift_analytics_hydration_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DRM_MEMORY_LOG_PATH", str(tmp_path / "revisions.jsonl"))
    monkeypatch.setattr("core.memory_manager.redis_module", None)
    monkeypatch.setattr("core.memory_manager.chromadb_module", None)
    monkeypatch.setattr("core.memory_manager.chroma_embeddings_module", None)

    manager = MemoryManager(load_app_config())
    manager.put_working_item(
        WorkingMemoryItem(
            key="task:test:drift",
            payload={"advisory": "slow"},
            ttl_seconds=10,
        )
    )
    manager.record_drift_analytics(
        DriftAnalyticsRecord(
            id="analytics-test",
            task_reference="task:test",
            workflow="fast",
            latency_seconds=1.25,
            verdict="pass",
            slo_breaches=("latency",),
            drift_advisory="slow",
            workflow_biases={"fast": 0.1},
            mitigation_plan={"working_pruned": 1},
        )
    )

    records = manager.list_drift_analytics()
    assert records[-1].id == "analytics-test"
    assert records[-1].slo_breaches == ("latency",)

    metrics = manager.snapshot_metrics()
    assert metrics["working_items"] == 1
    assert metrics["drift_advisories"] == 1
    assert metrics["analytics_records"] == 1


def test_semantic_linking_and_neighbors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DRM_MEMORY_LOG_PATH", str(tmp_path / "revisions.jsonl"))
    monkeypatch.setattr("core.memory_manager.redis_module", None)
    monkeypatch.setattr("core.memory_manager.chromadb_module", None)
    monkeypatch.setattr("core.memory_manager.chroma_embeddings_module", None)

    manager = MemoryManager(load_app_config())
    manager.record_semantic(
        SemanticNode(id="source", label="Source", definition="Source node")
    )
    manager.record_semantic(
        SemanticNode(id="target", label="Target", definition="Target node")
    )

    manager.link_semantic_nodes("source", "target", weight=2.0)
    neighbours = manager.get_semantic_neighbors("source")

    assert neighbours
    assert neighbours[0][0].id == "target"
    assert neighbours[0][1] == pytest.approx(1.0)
    assert manager.get_semantic_neighbors("missing") == []
