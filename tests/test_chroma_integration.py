"""Integration-style tests for Chroma persistence using stubs."""

from __future__ import annotations

import types
from pathlib import Path

from config.settings import load_app_config
from core.memory_manager import MemoryManager
from models.memory import SemanticNode


class _StubCollection:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def upsert(self, ids=None, documents=None, metadatas=None, **_: object):  # pragma: no cover - stub
        if not ids or not documents:
            return
        self._store[ids[0]] = documents[0]

    def get(self, *, where=None, include=None, ids=None):  # pragma: no cover - stub
        if ids:
            document = self._store.get(ids[0])
            return {"documents": [document] if document else []}
        return {"documents": list(self._store.values())}


class _StubClient:
    def __init__(self, path: str) -> None:  # pragma: no cover - simple stub
        self._path = path
        self._collection = _StubCollection()

    def get_or_create_collection(self, name: str, embedding_function=None):
        return self._collection

    def list_collections(self):  # pragma: no cover - stub
        return []


def test_semantic_roundtrip_with_chroma_stub(monkeypatch, tmp_path: Path) -> None:
    chroma_stub = types.SimpleNamespace(PersistentClient=_StubClient)

    monkeypatch.setenv("DRM_MEMORY_LOG_PATH", str(tmp_path / "revisions.jsonl"))
    monkeypatch.setattr("core.memory_manager.redis_module", None)
    monkeypatch.setattr("core.memory_manager.chromadb_module", chroma_stub)
    monkeypatch.setattr("core.memory_manager.chroma_embeddings_module", None)

    config = load_app_config()
    config.memory.chromadb.persist_directory = str(tmp_path / "chroma")

    manager = MemoryManager(config)
    node = SemanticNode(id="concept:stub", label="Stub", definition="Stub concept")

    manager.record_semantic(node)

    stored = manager.list_layer("semantic")
    assert stored

    retrieved = manager.get_semantic_node(node.id)
    assert retrieved is not None
    assert retrieved.id == node.id
