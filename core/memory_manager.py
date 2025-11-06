"""Hybrid memory manager combining Redis and ChromaDB layers.

Updates: v0.1 - 2025-11-06 - Implemented Redis/ChromaDB memory scaffolding with
graceful fallbacks for development environments.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional

from config.settings import AppConfig
from core.exceptions import MemoryError
from models.memory import (
    EpisodicMemoryEntry,
    ReviewRecord,
    SemanticNode,
    WorkingMemoryItem,
)

try:  # pragma: no cover - optional dependency
    import redis  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully
    redis = None

try:  # pragma: no cover - optional dependency
    import chromadb  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully
    chromadb = None

LOGGER = logging.getLogger("drm.memory")


class RedisMemoryStore:
    """Adapter around Redis for working memory operations."""

    def __init__(self, config: AppConfig) -> None:
        redis_cfg = config.memory.redis
        self._host = redis_cfg.host
        self._port = redis_cfg.port
        self._db = redis_cfg.db
        self._ttl_seconds = redis_cfg.ttl_seconds
        self._client: Optional["redis.Redis"] = None
        self._fallback: Dict[str, WorkingMemoryItem] = {}
        self._logger = logging.getLogger("drm.memory.redis")

        if redis is None:
            self._logger.warning(
                "Redis python client unavailable; using in-memory fallback store."
            )
        else:
            try:
                self._client = redis.Redis(
                    host=self._host,
                    port=self._port,
                    db=self._db,
                    socket_timeout=2,
                )
                # probe the connection lazily
                self._client.ping()
            except Exception as exc:  # pragma: no cover - runtime check
                self._logger.error(
                    "Redis connection failed (%s); falling back to in-memory store.",
                    exc,
                )
                self._client = None

    def put(self, item: WorkingMemoryItem) -> None:
        """Store an item in working memory."""
        try:
            if self._client:
                self._client.setex(
                    name=item.key,
                    time=item.ttl_seconds or self._ttl_seconds,
                    value=json.dumps(asdict(item), default=str),
                )
            else:
                self._fallback[item.key] = item
        except Exception as exc:  # pragma: no cover - client failure path
            raise MemoryError(f"Failed to store working memory item: {exc}") from exc

    def get(self, key: str) -> Optional[WorkingMemoryItem]:
        """Retrieve an item from working memory."""
        try:
            if self._client:
                payload = self._client.get(key)
                if payload is None:
                    return None
                data = json.loads(payload)
                return WorkingMemoryItem(
                    key=data["key"],
                    payload=data["payload"],
                    ttl_seconds=data["ttl_seconds"],
                    created_at=datetime.fromisoformat(data["created_at"])
                    if data.get("created_at")
                    else datetime.utcnow(),
                )
            return self._fallback.get(key)
        except Exception as exc:  # pragma: no cover
            raise MemoryError(f"Failed to load working memory item: {exc}") from exc


class ChromaMemoryStore:
    """Adapter around ChromaDB for episodic, semantic, and review memory."""

    def __init__(self, config: AppConfig) -> None:
        chroma_cfg = config.memory.chromadb
        self._persist_directory = chroma_cfg.persist_directory
        self._collection_name = chroma_cfg.collection
        self._logger = logging.getLogger("drm.memory.chroma")

        self._client = None
        self._collection = None
        self._fallback: Dict[str, Dict[str, dict]] = {
            "episodic": {},
            "semantic": {},
            "review": {},
        }

        if chromadb is None:
            self._logger.warning(
                "ChromaDB client unavailable; persisting memory in process only."
            )
            return

        try:  # pragma: no cover - needs chromadb runtime
            self._client = chromadb.PersistentClient(
                path=self._persist_directory  # type: ignore[arg-type]
            )
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name
            )
        except Exception as exc:
            self._logger.error(
                "ChromaDB connection failed (%s); falling back to in-memory store.",
                exc,
            )
            self._client = None

    def _store_in_collection(self, layer: str, entry_id: str, payload: dict) -> None:
        """Store data either via ChromaDB or fallback map."""
        if self._collection is None:  # fallback path
            self._fallback[layer][entry_id] = payload
            return

        try:  # pragma: no cover - depends on chromadb
            self._collection.upsert(
                ids=[f"{layer}:{entry_id}"],
                documents=[json.dumps(payload)],
                metadatas=[{"layer": layer}],
            )
        except Exception as exc:
            raise MemoryError(
                f"Failed to persist {layer} memory entry {entry_id}: {exc}"
            ) from exc

    def add_episodic(self, entry: EpisodicMemoryEntry) -> None:
        """Persist an episodic memory entry."""
        self._store_in_collection("episodic", entry.id, asdict(entry))

    def add_semantic(self, node: SemanticNode) -> None:
        """Persist a semantic concept node."""
        self._store_in_collection("semantic", node.id, asdict(node))

    def add_review(self, record: ReviewRecord) -> None:
        """Persist a review record entry."""
        self._store_in_collection("review", record.id, asdict(record))

    def list_layer(self, layer: str) -> List[dict]:
        """Return layer payloads from the available store."""
        if self._collection is None:
            return list(self._fallback[layer].values())

        try:  # pragma: no cover - depends on chromadb
            query = self._collection.get(
                where={"layer": layer},
                include=["documents"],
            )
            documents = query.get("documents", [])
            return [json.loads(doc) for doc in documents]
        except Exception as exc:
            raise MemoryError(f"Failed to query {layer} memory: {exc}") from exc


class MemoryManager:
    """Coordinates memory interactions across layers."""

    def __init__(self, config: AppConfig) -> None:
        self._redis_store = RedisMemoryStore(config)
        self._chroma_store = ChromaMemoryStore(config)
        self._logger = LOGGER

    def put_working_item(self, item: WorkingMemoryItem) -> None:
        """Store a working memory item with error handling."""
        self._logger.debug("Storing working memory item %s", item.key)
        self._redis_store.put(item)

    def get_working_item(self, key: str) -> Optional[WorkingMemoryItem]:
        """Retrieve a working memory item if present."""
        self._logger.debug("Retrieving working memory item %s", key)
        return self._redis_store.get(key)

    def record_episodic(self, entry: EpisodicMemoryEntry) -> None:
        """Append a new episodic memory entry."""
        self._logger.debug("Recording episodic memory %s", entry.id)
        self._chroma_store.add_episodic(entry)

    def record_semantic(self, node: SemanticNode) -> None:
        """Append a new semantic node."""
        self._logger.debug("Recording semantic node %s", node.id)
        self._chroma_store.add_semantic(node)

    def record_review(self, record: ReviewRecord) -> None:
        """Persist a review record after task execution."""
        self._logger.debug("Recording review %s", record.id)
        self._chroma_store.add_review(record)

    def list_layer(self, layer: str) -> List[dict]:
        """List stored items for the requested layer."""
        if layer not in {"episodic", "semantic", "review"}:
            raise MemoryError(f"Unsupported memory layer requested: {layer}")
        return self._chroma_store.list_layer(layer)
