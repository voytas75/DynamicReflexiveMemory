"""Hybrid memory manager combining Redis and ChromaDB layers.

Updates:
    v0.1 - 2025-11-06 - Implemented Redis/ChromaDB memory scaffolding with graceful fallbacks for development environments.
    v0.2 - 2025-11-07 - Normalised timestamps to timezone-aware UTC handling.
    v0.3 - 2025-11-07 - Added revision logging and history export for memory operations.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, cast

from config.settings import AppConfig, EmbeddingConfig
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

redis_module: Any
chromadb_module: Any
chroma_embeddings_module: Any

try:  # pragma: no cover - optional dependency
    import redis as redis_module
except ImportError:  # pragma: no cover - handled gracefully
    redis_module = None

try:  # pragma: no cover - optional dependency
    import chromadb as chromadb_module
    from chromadb.utils import embedding_functions as chroma_embeddings_module
except ImportError:  # pragma: no cover - handled gracefully
    chromadb_module = None
    chroma_embeddings_module = None

try:  # pragma: no cover - optional dependency
    from openai import AzureOpenAI as _AzureOpenAI
except ImportError:  # pragma: no cover
    _AzureOpenAI = None

AzureOpenAI = cast(Any, _AzureOpenAI)

LOGGER = logging.getLogger("drm.memory")

REVISION_LOG_ENV = "DRM_MEMORY_LOG_PATH"
DEFAULT_REVISION_LOG = PROJECT_ROOT / "data" / "logs" / "memory_revisions.jsonl"


class MemoryRevisionLogger:
    """Append-only revision log supporting rollback-aware auditing."""

    def __init__(self, path_override: Optional[Path] = None) -> None:
        log_path = self._resolve_log_path(path_override)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = log_path
        self._lock = Lock()
        self._revision = self._load_last_revision()

    def log(self, layer: str, identifier: str, payload: Dict[str, object]) -> None:
        """Append a revision entry capturing the memory mutation."""
        record = {
            "layer": layer,
            "id": identifier,
            "payload": payload,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._revision += 1
            record["revision"] = self._revision
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str))
                handle.write("\n")

    def history(self, limit: int = 20) -> List[Dict[str, object]]:
        """Return the most recent revision entries up to *limit*."""
        if not self._path.exists():
            return []

        with self._path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()
        selected = lines[-limit:]
        history: List[Dict[str, object]] = []
        for line in selected:
            line = line.strip()
            if not line:
                continue
            try:
                history.append(cast(Dict[str, object], json.loads(line)))
            except json.JSONDecodeError:
                LOGGER.debug("Skipping malformed revision log entry: %s", line)
        return history

    def _resolve_log_path(self, override: Optional[Path]) -> Path:
        if override is not None:
            return override
        env_value = os.getenv(REVISION_LOG_ENV)
        if env_value:
            candidate = Path(env_value)
            if candidate.suffix:
                return candidate
            return candidate / DEFAULT_REVISION_LOG.name
        return DEFAULT_REVISION_LOG

    def _load_last_revision(self) -> int:
        if not self._path.exists():
            return 0
        last_revision = 0
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    revision_raw = record.get("revision")
                    if isinstance(revision_raw, int) and revision_raw > last_revision:
                        last_revision = revision_raw
        except OSError as exc:
            LOGGER.warning("Unable to read revision log %s: %s", self._path, exc)
        return last_revision


class RedisMemoryStore:
    """Adapter around Redis for working memory operations."""

    def __init__(self, config: AppConfig) -> None:
        redis_cfg = config.memory.redis
        self._host = redis_cfg.host
        self._port = redis_cfg.port
        self._db = redis_cfg.db
        self._ttl_seconds = redis_cfg.ttl_seconds
        self._client: Optional[Any] = None
        self._fallback: Dict[str, WorkingMemoryItem] = {}
        self._logger = logging.getLogger("drm.memory.redis")

        if redis_module is None:
            self._logger.warning(
                "Redis python client unavailable; using in-memory fallback store."
            )
        else:
            try:
                self._client = redis_module.Redis(
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
                payload_bytes = self._client.get(key)
                if payload_bytes is None:
                    return None
                data = cast(Dict[str, object], json.loads(payload_bytes.decode("utf-8")))
                key_value = str(data.get("key", ""))
                payload_value = cast(Dict[str, object], data.get("payload", {}))
                ttl_value_raw = data.get("ttl_seconds", self._ttl_seconds)
                ttl_value = int(ttl_value_raw) if isinstance(ttl_value_raw, (int, float)) else self._ttl_seconds
                return WorkingMemoryItem(
                    key=key_value,
                    payload=payload_value,
                    ttl_seconds=ttl_value,
                    created_at=self._coerce_timestamp(data.get("created_at")),
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
                    payload_bytes = self._client.get(key)
                    if not payload_bytes:
                        continue
                    data = cast(Dict[str, object], json.loads(payload_bytes.decode("utf-8")))
                    key_value = (
                        key.decode("utf-8") if isinstance(key, bytes) else str(key)
                    )
                    payload_value = cast(Dict[str, object], data.get("payload", {}))
                    ttl_raw = data.get("ttl_seconds", self._ttl_seconds)
                    ttl_value = int(ttl_raw) if isinstance(ttl_raw, (int, float)) else self._ttl_seconds
                    items.append(
                        WorkingMemoryItem(
                            key=key_value,
                            payload=payload_value,
                            ttl_seconds=ttl_value,
                            created_at=self._coerce_timestamp(data.get("created_at")),
                        )
                    )
                return items
            return list(self._fallback.values())
        except Exception as exc:  # pragma: no cover
            raise MemoryError(f"Failed to enumerate working memory: {exc}") from exc

    @staticmethod
    def _coerce_timestamp(value: Optional[object]) -> datetime:
        """Coerce stored timestamps into timezone-aware datetimes."""
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                return datetime.now(timezone.utc)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc)


class ChromaMemoryStore:
    """Adapter around ChromaDB for episodic, semantic, and review memory."""

    def __init__(self, config: AppConfig) -> None:
        chroma_cfg = config.memory.chromadb
        self._persist_directory = chroma_cfg.persist_directory
        self._collection_name = chroma_cfg.collection
        self._logger = logging.getLogger("drm.memory.chroma")

        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._fallback: Dict[str, Dict[str, Dict[str, object]]] = {
            "episodic": {},
            "semantic": {},
            "review": {},
        }
        self._embedding_fn: Optional[Any] = None

        if chromadb_module is None:
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
            self._client = chromadb_module.PersistentClient(
                path=self._persist_directory
            )
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                embedding_function=cast(Any, self._embedding_fn),
            )
        except Exception as exc:
            self._logger.error(
                "ChromaDB connection failed (%s); falling back to in-memory store.",
                exc,
            )
            self._client = None

    def _build_embedding_function(self, config: AppConfig) -> Optional[object]:
        """Construct the embedding function if configuration allows."""
        embedding_cfg = config.embedding
        if embedding_cfg is None:
            return None

        provider = embedding_cfg.provider.lower()
        if provider == "azure":
            return self._build_azure_embedding_function(embedding_cfg)

        if chroma_embeddings_module is None:
            self._logger.warning(
                "Chroma embedding extras unavailable; install the 'openai' extra to enable embeddings."
            )
            return None

        self._logger.warning(
            "Embedding provider '%s' not supported; using in-memory store.", provider
        )
        return None

    def _build_azure_embedding_function(
        self, embedding_cfg: EmbeddingConfig
    ) -> Optional[object]:
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

        if endpoint is None:
            return None

        if AzureOpenAI is None:
            self._logger.warning(
                "openai Azure client unavailable; install the 'openai' package to enable embeddings."
            )
            return None

        try:
            endpoint_str = endpoint.rstrip("/")
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint_str,
            )
        except Exception as exc:  # pragma: no cover
            self._logger.error(
                "Failed to initialise Azure OpenAI client (%s); falling back to in-memory store.",
                exc,
            )
            return None

        class AzureEmbeddingFunction:
            """Callable embedding function compatible with Chroma."""

            def __init__(self, azure_client: Any, deployment: str) -> None:
                self._client = azure_client
                self._deployment = deployment

            def __call__(self, input: Sequence[str]) -> List[List[float]]:
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

    def _store_in_collection(
        self, layer: str, entry_id: str, payload: Dict[str, object]
    ) -> None:
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

    def list_layer(self, layer: str) -> List[Dict[str, object]]:
        """Return layer payloads from the available store."""
        if self._collection is None:
            return list(self._fallback[layer].values())

        try:  # pragma: no cover - depends on chromadb
            query = self._collection.get(
                where={"layer": layer},
                include=["documents"],
            )
            documents = query.get("documents") or []
            parsed: List[Dict[str, object]] = []
            for doc in documents:
                if isinstance(doc, str):
                    parsed.append(cast(Dict[str, object], json.loads(doc)))
            return parsed
        except Exception as exc:
            raise MemoryError(f"Failed to query {layer} memory: {exc}") from exc


class MemoryManager:
    """Coordinates memory interactions across layers."""

    def __init__(self, config: AppConfig) -> None:
        self._redis_store = RedisMemoryStore(config)
        self._chroma_store = ChromaMemoryStore(config)
        self._logger = LOGGER
        self._revision_logger = MemoryRevisionLogger()

    def put_working_item(self, item: WorkingMemoryItem) -> None:
        """Store a working memory item with error handling."""
        self._logger.debug("Storing working memory item %s", item.key)
        self._redis_store.put(item)
        self._revision_logger.log("working", item.key, asdict(item))

    def get_working_item(self, key: str) -> Optional[WorkingMemoryItem]:
        """Retrieve a working memory item if present."""
        self._logger.debug("Retrieving working memory item %s", key)
        return self._redis_store.get(key)

    def record_episodic(self, entry: EpisodicMemoryEntry) -> None:
        """Append a new episodic memory entry."""
        self._logger.debug("Recording episodic memory %s", entry.id)
        entry_dict = asdict(entry)
        self._chroma_store.add_episodic(entry)
        self._revision_logger.log("episodic", entry.id, entry_dict)

    def record_semantic(self, node: SemanticNode) -> None:
        """Append a new semantic node."""
        self._logger.debug("Recording semantic node %s", node.id)
        node_dict = asdict(node)
        self._chroma_store.add_semantic(node)
        self._revision_logger.log("semantic", node.id, node_dict)

    def record_review(self, record: ReviewRecord) -> None:
        """Persist a review record after task execution."""
        self._logger.debug("Recording review %s", record.id)
        record_dict = asdict(record)
        self._chroma_store.add_review(record)
        self._revision_logger.log("review", record.id, record_dict)

    def list_layer(self, layer: str) -> List[Dict[str, object]]:
        """List stored items for the requested layer."""
        if layer not in {"episodic", "semantic", "review"}:
            raise MemoryError(f"Unsupported memory layer requested: {layer}")
        return self._chroma_store.list_layer(layer)

    def list_working_items(self, pattern: str = "*") -> List[WorkingMemoryItem]:
        """List working memory records for UI inspection."""
        return self._redis_store.list_items(pattern)

    def get_revision_history(self, limit: int = 20) -> List[Dict[str, object]]:
        """Return the latest revision entries for audit or rollback tooling."""
        return self._revision_logger.history(limit)
