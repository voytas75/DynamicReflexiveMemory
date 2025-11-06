"""Hybrid memory manager combining Redis and ChromaDB layers.

Updates: v0.1 - 2025-11-06 - Implemented Redis/ChromaDB memory scaffolding with
graceful fallbacks for development environments.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from config.settings import AppConfig
from core.exceptions import MemoryError
from models.memory import (
    EpisodicMemoryEntry,
    ReviewRecord,
    SemanticNode,
    WorkingMemoryItem,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHROMA_CACHE = PROJECT_ROOT / "data" / "chromadb" / "cache"
try:
    DEFAULT_CHROMA_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CHROMA_CACHE_DIR", str(DEFAULT_CHROMA_CACHE))
except OSError:
    pass

try:  # pragma: no cover - optional dependency
    import redis  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully
    redis = None

try:  # pragma: no cover - optional dependency
    import chromadb  # type: ignore
    from chromadb.utils import embedding_functions as chroma_embeddings  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully
    chromadb = None
    chroma_embeddings = None

try:  # pragma: no cover - optional dependency
    from openai import AzureOpenAI  # type: ignore
except ImportError:  # pragma: no cover
    AzureOpenAI = None

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

    def list_items(self, pattern: str = "*") -> List[WorkingMemoryItem]:
        """Return working memory items matching the given pattern."""
        try:
            if self._client:
                items: List[WorkingMemoryItem] = []
                for key in self._client.scan_iter(match=pattern):
                    payload = self._client.get(key)
                    if not payload:
                        continue
                    data = json.loads(payload)
                    items.append(
                        WorkingMemoryItem(
                            key=key.decode("utf-8") if isinstance(key, bytes) else str(key),
                            payload=data["payload"],
                            ttl_seconds=data["ttl_seconds"],
                            created_at=datetime.fromisoformat(data["created_at"])
                            if data.get("created_at")
                            else datetime.utcnow(),
                        )
                    )
                return items
            return list(self._fallback.values())
        except Exception as exc:  # pragma: no cover
            raise MemoryError(f"Failed to enumerate working memory: {exc}") from exc


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
        self._embedding_fn = None

        if chromadb is None:
            self._logger.warning(
                "ChromaDB client unavailable; persisting memory in process only."
            )
            return
        try:
            persist_path = Path(self._persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            cache_dir = persist_path / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("CHROMA_CACHE_DIR", str(cache_dir))
        except OSError as exc:
            self._logger.error(
                "Failed to prepare Chroma directories (%s); falling back to in-memory store.",
                exc,
            )
            return

        self._embedding_fn = self._build_embedding_function(config)

        try:  # pragma: no cover - needs chromadb runtime
            self._client = chromadb.PersistentClient(
                path=self._persist_directory  # type: ignore[arg-type]
            )
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                embedding_function=self._embedding_fn,
            )
        except Exception as exc:
            self._logger.error(
                "ChromaDB connection failed (%s); falling back to in-memory store.",
                exc,
            )
            self._client = None

    def _build_embedding_function(self, config: AppConfig):
        """Construct the embedding function if configuration allows."""
        embedding_cfg = config.embedding
        if embedding_cfg is None:
            return None

        provider = embedding_cfg.provider.lower()
        if provider == "azure":
            return self._build_azure_embedding_function(embedding_cfg)

        if chroma_embeddings is None:
            self._logger.warning(
                "Chroma embedding extras unavailable; install the 'openai' extra to enable embeddings."
            )
            return None

        self._logger.warning(
            "Embedding provider '%s' not supported; using in-memory store.", provider
        )
        return None

    def _build_azure_embedding_function(self, embedding_cfg) -> Optional[object]:
        """Return an embedding function compatible with Azure OpenAI."""
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv(
            "AZURE_OPENAI_API_BASE"
        )
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
        deployment_name = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", embedding_cfg.model
        )

        missing = [
            name
            for name, value in [
                ("AZURE_OPENAI_API_KEY", api_key),
                ("AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_BASE", endpoint),
            ]
            if not value
        ]
        if missing:
            self._logger.warning(
                "Azure embedding credentials missing (%s); using in-memory store.",
                ", ".join(missing),
            )
            return None

        if AzureOpenAI is None:
            self._logger.warning(
                "openai Azure client unavailable; install the 'openai' package to enable embeddings."
            )
            return None

        try:
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint.rstrip("/"),
            )
        except Exception as exc:  # pragma: no cover
            self._logger.error(
                "Failed to initialise Azure OpenAI client (%s); falling back to in-memory store.",
                exc,
            )
            return None

        class AzureEmbeddingFunction:
            """Callable embedding function compatible with Chroma."""

            def __init__(self, azure_client: "AzureOpenAI", deployment: str) -> None:
                self._client = azure_client
                self._deployment = deployment

            def __call__(self, input: Sequence[str]) -> List[List[float]]:  # type: ignore[override]
                return self._embed(list(input))

            def embed_query(self, text: str) -> List[float]:
                return self._embed([text])[0]

            def embed_documents(self, input: Sequence[str]) -> List[List[float]]:
                return self._embed(list(input))

            def _embed(self, texts: Sequence[str]) -> List[List[float]]:
                try:
                    response = self._client.embeddings.create(
                        model=self._deployment,
                        input=list(texts),
                    )
                    return [item.embedding for item in response.data]
                except Exception as exc:  # pragma: no cover
                    raise MemoryError(
                        f"Azure embedding request failed: {exc}"
                    ) from exc

        return AzureEmbeddingFunction(client, deployment_name)

    def _store_in_collection(self, layer: str, entry_id: str, payload: dict) -> None:
        """Store data either via ChromaDB or fallback map."""
        if self._collection is None:  # fallback path
            self._fallback[layer][entry_id] = payload
            return

        try:  # pragma: no cover - depends on chromadb
            self._collection.upsert(
                ids=[f"{layer}:{entry_id}"],
                documents=[json.dumps(payload, default=str)],
                metadatas=[{"layer": layer}],
            )
        except Exception as exc:
            message = str(exc).lower()
            if "permission" in message or "embedding function" in message:
                self._logger.warning(
                    "ChromaDB unavailable (%s); reverting to in-memory store.", exc
                )
                self._collection = None
                self._fallback[layer][entry_id] = payload
                return
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

    def list_working_items(self, pattern: str = "*") -> List[WorkingMemoryItem]:
        """List working memory records for UI inspection."""
        return self._redis_store.list_items(pattern)
