"""Tests covering semantic graph utilities in the memory manager."""

from __future__ import annotations

import pytest

from config.settings import load_app_config
from core.memory_manager import MemoryManager
from models.memory import SemanticNode, WorkingMemoryItem


def _bootstrap_manager(monkeypatch, tmp_path) -> MemoryManager:
    monkeypatch.setenv("DRM_MEMORY_LOG_PATH", str(tmp_path / "revisions.jsonl"))
    monkeypatch.setattr("core.memory_manager.redis_module", None)
    monkeypatch.setattr("core.memory_manager.chromadb_module", None)
    monkeypatch.setattr("core.memory_manager.chroma_embeddings_module", None)
    config = load_app_config()
    return MemoryManager(config)


def test_link_semantic_nodes_updates_relations(monkeypatch, tmp_path) -> None:
    manager = _bootstrap_manager(monkeypatch, tmp_path)

    node_a = SemanticNode(id="concept:a", label="Alpha", definition="Alpha concept")
    node_b = SemanticNode(id="concept:b", label="Beta", definition="Beta concept")

    manager.record_semantic(node_a)
    manager.record_semantic(node_b)

    manager.link_semantic_nodes(node_a.id, node_b.id, 0.75)

    forward = manager.get_semantic_neighbors(node_a.id)
    reverse = manager.get_semantic_neighbors(node_b.id)

    assert forward and forward[0][0].id == node_b.id
    assert reverse and reverse[0][0].id == node_a.id
    assert forward[0][1] == pytest.approx(0.75, abs=1e-6)


def test_list_semantic_nodes_returns_ordered(monkeypatch, tmp_path) -> None:
    manager = _bootstrap_manager(monkeypatch, tmp_path)

    first = SemanticNode(id="concept:old", label="Old", definition="Old concept")
    second = SemanticNode(id="concept:new", label="New", definition="New concept")

    manager.record_semantic(first)
    manager.record_semantic(second)

    nodes = manager.list_semantic_nodes()
    assert [node.id for node in nodes][-2:] == [first.id, second.id]

    limited = manager.list_semantic_nodes(limit=1)
    assert [node.id for node in limited] == [second.id]


def test_apply_drift_mitigation_prunes_working(monkeypatch, tmp_path) -> None:
    manager = _bootstrap_manager(monkeypatch, tmp_path)

    for index in range(3):
        item = WorkingMemoryItem(
            key=f"task:test:{index}",
            payload={"value": index},
            ttl_seconds=60,
        )
        manager.put_working_item(item)

    summary = manager.apply_drift_mitigation(
        task_id="task:test",
        max_working_items=1,
        relation_decay=0.5,
    )

    assert summary.get("working_pruned") == 2
