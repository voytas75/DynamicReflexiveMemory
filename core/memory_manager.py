"""Hybrid memory manager combining Redis and ChromaDB layers.

Updates:
    v0.1 - 2025-11-06 - Implemented Redis/ChromaDB memory scaffolding with graceful fallbacks for development environments.
    v0.2 - 2025-11-07 - Normalised timestamps to timezone-aware UTC handling.
    v0.3 - 2025-11-07 - Added revision logging and history export for memory operations.
    v0.4 - 2025-11-07 - Added semantic graph utilities and relation management APIs.
    v0.5 - 2025-11-07 - Instrumented memory operations with telemetry spans and metrics.
    v0.6 - 2025-11-07 - Added retrieval helpers, tamper-evident revision hashing, and replay utilities.
    v0.7 - 2025-11-08 - Added resilient Redis reconnection and fallback handling.
    v0.8 - 2025-11-08 - Published telemetry metrics snapshots for GUI monitoring.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from difflib import SequenceMatcher
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, cast

from config.settings import AppConfig, EmbeddingConfig
from core.exceptions import MemoryError
from core.telemetry import emit_metric, log_span, publish_event
from models.memory import (
    EpisodicMemoryEntry,
    ReviewRecord,
    SemanticNode,
    WorkingMemoryItem,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHROMA_CACHE = PROJECT_ROOT / "data" / "chromadb" / "cache"
os.environ.setdefault("CHROMA_TELEMETRY", "FALSE")
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

SEMANTIC_NODE_PREFIX = "node:"
MAX_WORKING_MEMORY_ITEMS = 100
SEMANTIC_RELATION_DECAY = 0.85
SEMANTIC_RELATION_MIN_THRESHOLD = 0.05


class MemoryRevisionLogger:
    """Append-only revision log supporting rollback-aware auditing."""

    def __init__(self, path_override: Optional[Path] = None) -> None:
        log_path = self._resolve_log_path(path_override)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = log_path
        self._lock = Lock()
        self._revision, self._tail_hash = self._load_log_tail()

    def log(self, layer: str, identifier: str, payload: Dict[str, object]) -> None:
        """Append a revision entry capturing the memory mutation."""
        with self._lock:
            record = {
                "layer": layer,
                "id": identifier,
                "payload": payload,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._revision += 1
            record["revision"] = self._revision
            record["prev_hash"] = self._tail_hash
            canonical = json.dumps(
                record,
                default=str,
                sort_keys=True,
            )
            record_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            record["hash"] = record_hash
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str))
                handle.write("\n")
            self._tail_hash = record_hash

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

    def verify(self, limit_revision: Optional[int] = None) -> bool:
        """Return True when the revision log hashes and chain are valid."""
        expected_prev: Optional[str] = None
        for record in self._iter_records(limit_revision):
            if record.get("prev_hash") != expected_prev:
                return False
            computed = self._calculate_hash(record)
            if computed != record.get("hash"):
                return False
            expected_prev = record.get("hash")
        return True

    def replay_layer(
        self, layer: str, limit_revision: Optional[int] = None
    ) -> List[Dict[str, object]]:
        """Reconstruct the layer state up to *limit_revision* by replaying the log."""
        state: Dict[str, Dict[str, object]] = {}
        for record in self._iter_records(limit_revision):
            if record.get("layer") != layer:
                continue
            identifier = str(record.get("id"))
            payload_raw = record.get("payload")
            if isinstance(payload_raw, dict):
                state[identifier] = payload_raw
        return list(state.values())

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

    def _load_log_tail(self) -> Tuple[int, Optional[str]]:
        if not self._path.exists():
            return 0, None
        last_revision = 0
        tail_hash: Optional[str] = None
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
                        tail_hash = record.get("hash")
        except OSError as exc:
            LOGGER.warning("Unable to read revision log %s: %s", self._path, exc)
        return last_revision, tail_hash

    def _iter_records(
        self, limit_revision: Optional[int] = None
    ) -> Iterator[Dict[str, object]]:
        if not self._path.exists():
            return iter(())

        def _iterator() -> Iterator[Dict[str, object]]:
            try:
                with self._path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = cast(Dict[str, object], json.loads(line))
                        except json.JSONDecodeError:
                            continue
                        revision_raw = record.get("revision")
                        if isinstance(limit_revision, int) and isinstance(
                            revision_raw, int
                        ):
                            if revision_raw > limit_revision:
                                break
                        yield record
            except OSError as exc:
                LOGGER.warning("Unable to iterate revision log %s: %s", self._path, exc)

        return _iterator()

    @staticmethod
    def _calculate_hash(record: Dict[str, object]) -> str:
        """Recompute the record hash (ignoring any stored hash value)."""
        payload = dict(record)
        payload.pop("hash", None)
        canonical = json.dumps(payload, default=str, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
            return

        self._initialise_client()

    def put(self, item: WorkingMemoryItem) -> None:
        """Store an item in working memory."""
        if self._ensure_client():
            client = cast(Any, self._client)
            try:
                client.setex(
                    name=item.key,
                    time=item.ttl_seconds or self._ttl_seconds,
                    value=json.dumps(asdict(item), default=str),
                )
                return
            except Exception as exc:  # pragma: no cover - client failure path
                self._logger.warning(
                    "Redis store failed (%s); switching to in-memory fallback.", exc
                )
                self._client = None

        self._fallback[item.key] = item

    def get(self, key: str) -> Optional[WorkingMemoryItem]:
        """Retrieve an item from working memory."""
        if self._ensure_client():
            client = cast(Any, self._client)
            try:
                payload_bytes = client.get(key)
                if payload_bytes is None:
                    return None
                data = cast(Dict[str, object], json.loads(payload_bytes.decode("utf-8")))
                key_value = str(data.get("key", ""))
                payload_value = cast(Dict[str, object], data.get("payload", {}))
                ttl_value_raw = data.get("ttl_seconds", self._ttl_seconds)
                ttl_value = (
                    int(ttl_value_raw)
                    if isinstance(ttl_value_raw, (int, float))
                    else self._ttl_seconds
                )
                return WorkingMemoryItem(
                    key=key_value,
                    payload=payload_value,
                    ttl_seconds=ttl_value,
                    created_at=self._coerce_timestamp(data.get("created_at")),
                )
            except Exception as exc:  # pragma: no cover
                self._logger.warning(
                    "Redis retrieval failed (%s); falling back to in-memory store.",
                    exc,
                )
                self._client = None

        return self._fallback.get(key)

    def delete(self, key: str) -> None:
        """Remove an item from working memory if present."""
        if self._ensure_client():
            client = cast(Any, self._client)
            try:
                client.delete(key)
                return
            except Exception as exc:  # pragma: no cover
                self._logger.warning(
                    "Redis deletion failed (%s); removing item from fallback store.",
                    exc,
                )
                self._client = None

        self._fallback.pop(key, None)

    def list_items(self, pattern: str = "*") -> List[WorkingMemoryItem]:
        """Return working memory items matching the given pattern."""
        if self._ensure_client():
            client = cast(Any, self._client)
            try:
                items: List[WorkingMemoryItem] = []
                for key in client.scan_iter(match=pattern):
                    payload_bytes = client.get(key)
                    if not payload_bytes:
                        continue
                    data = cast(Dict[str, object], json.loads(payload_bytes.decode("utf-8")))
                    key_value = (
                        key.decode("utf-8") if isinstance(key, bytes) else str(key)
                    )
                    payload_value = cast(Dict[str, object], data.get("payload", {}))
                    ttl_raw = data.get("ttl_seconds", self._ttl_seconds)
                    ttl_value = (
                        int(ttl_raw)
                        if isinstance(ttl_raw, (int, float))
                        else self._ttl_seconds
                    )
                    items.append(
                        WorkingMemoryItem(
                            key=key_value,
                            payload=payload_value,
                            ttl_seconds=ttl_value,
                            created_at=self._coerce_timestamp(data.get("created_at")),
                        )
                    )
                return items
            except Exception as exc:  # pragma: no cover
                self._logger.warning(
                    "Redis enumeration failed (%s); returning fallback store state.",
                    exc,
                )
                self._client = None

        return list(self._fallback.values())

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

    def _initialise_client(self) -> None:
        """Initialise the Redis client if the dependency is available."""
        try:
            self._client = self._attempt_connect()
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            self._logger.error(
                "Redis connection failed during initialisation (%s); using in-memory fallback.",
                exc,
            )
            self._client = None

    def _attempt_connect(self) -> Optional[Any]:
        if redis_module is None:
            return None
        client = redis_module.Redis(
            host=self._host,
            port=self._port,
            db=self._db,
            socket_timeout=2,
        )
        client.ping()
        return client

    def _ensure_client(self) -> bool:
        """Ensure a live Redis client is available, reconnecting if needed."""
        if redis_module is None:
            return False
        if self._client is None:
            try:
                self._client = self._attempt_connect()
            except Exception as exc:  # pragma: no cover - runtime dependent
                self._logger.debug("Redis reconnect attempt failed: %s", exc)
                self._client = None
                return False
        return self._client is not None


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
        self._supports_vector_query = False

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
        self._supports_vector_query = self._embedding_fn is not None
        if not self._supports_vector_query:
            self._logger.warning(
                "Chroma embedding function unavailable; storing memory in-process."
            )
            return

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
                payload = list(texts)
                try:
                    with log_span(
                        "embedding.azure",
                        count=len(payload),
                        deployment=self._deployment,
                    ):
                        response = self._client.embeddings.create(
                            model=self._deployment,
                            input=payload,
                        )
                    emit_metric(
                        "embedding.azure.request",
                        value=len(payload) or 1,
                        deployment=self._deployment,
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
            self._fallback[layer][entry_id] = dict(payload)
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

    def get_semantic(self, node_id: str) -> Optional[Dict[str, object]]:
        """Return a semantic node payload by identifier if available."""
        if self._collection is None:
            record = self._fallback["semantic"].get(node_id)
            return dict(record) if record is not None else None

        try:  # pragma: no cover - depends on chromadb
            response = self._collection.get(
                ids=[f"semantic:{node_id}"],
                include=["documents"],
            )
        except Exception as exc:  # pragma: no cover - chromadb failure
            raise MemoryError(f"Failed to fetch semantic node {node_id}: {exc}") from exc

        documents = response.get("documents") or []
        if not documents:
            return None

        document = documents[0]
        if isinstance(document, str):
            return cast(Dict[str, object], json.loads(document))
        if isinstance(document, dict):
            return cast(Dict[str, object], document)
        return None

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

    def query_layer(self, layer: str, query: str, limit: int) -> List[Dict[str, object]]:
        """Return layer entries relevant to *query*, preferring semantic search."""
        limit = max(1, limit)
        normalized_query = query.strip()
        if not normalized_query:
            items = self.list_layer(layer)
            return items[-limit:] if limit < len(items) else items

        if self._collection is not None and self._supports_vector_query:
            try:  # pragma: no cover - depends on chromadb
                response = self._collection.query(
                    query_texts=[normalized_query],
                    where={"layer": layer},
                    n_results=limit,
                    include=["documents", "distances"],
                )
                documents = (response.get("documents") or [[]])[0]
                distances = (response.get("distances") or [[]])[0]
                ranked: List[Dict[str, object]] = []
                for index, document in enumerate(documents):
                    payload = (
                        cast(Dict[str, object], json.loads(document))
                        if isinstance(document, str)
                        else cast(Dict[str, object], document)
                    )
                    if not isinstance(payload, dict):
                        continue
                    score = 0.0
                    if isinstance(distances, list) and index < len(distances):
                        try:
                            distance = float(distances[index])
                            score = 1.0 / (1.0 + distance)
                        except (TypeError, ValueError):
                            score = 0.0
                    enriched = dict(payload)
                    enriched["_score"] = round(score, 6)
                    ranked.append(enriched)
                if ranked:
                    ranked.sort(key=lambda item: item.get("_score", 0.0), reverse=True)
                    return ranked[:limit]
            except Exception as exc:
                self._logger.warning(
                    "Chroma query failed (%s); falling back to local scoring.",
                    exc,
                )

        return self._fallback_search(layer, normalized_query, limit)

    def _fallback_search(
        self,
        layer: str,
        query: str,
        limit: int,
    ) -> List[Dict[str, object]]:
        storage = self._fallback.get(layer, {})
        if not storage:
            return []

        query_lower = query.lower()
        scored: List[Tuple[float, Dict[str, object]]] = []
        for payload in storage.values():
            text = json.dumps(payload, default=str).lower()
            if not text:
                continue
            similarity = SequenceMatcher(None, query_lower, text[:2048]).ratio()
            if query_lower in text:
                similarity += 0.5
            if similarity <= 0.05:
                continue
            candidate = dict(payload)
            candidate["_score"] = round(similarity, 6)
            scored.append((candidate["_score"], candidate))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:limit]]


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
        payload = asdict(item)
        with log_span("memory.put_working", key=item.key):
            self._redis_store.put(item)
            self._revision_logger.log("working", item.key, payload)
        emit_metric("memory.write", layer="working")
        self._publish_metrics_snapshot()

    def get_working_item(self, key: str) -> Optional[WorkingMemoryItem]:
        """Retrieve a working memory item if present."""
        self._logger.debug("Retrieving working memory item %s", key)
        return self._redis_store.get(key)

    def record_episodic(self, entry: EpisodicMemoryEntry) -> None:
        """Append a new episodic memory entry."""
        self._logger.debug("Recording episodic memory %s", entry.id)
        entry_dict = asdict(entry)
        with log_span("memory.record_episodic", id=entry.id):
            self._chroma_store.add_episodic(entry)
            self._revision_logger.log("episodic", entry.id, entry_dict)
        emit_metric("memory.write", layer="episodic")
        self._publish_metrics_snapshot()

    def record_semantic(self, node: SemanticNode) -> None:
        """Append a new semantic node."""
        self._logger.debug("Recording semantic node %s", node.id)
        node_dict = asdict(node)
        with log_span("memory.record_semantic", id=node.id):
            self._chroma_store.add_semantic(node)
            self._revision_logger.log("semantic", node.id, node_dict)
        emit_metric("memory.write", layer="semantic")
        self._publish_metrics_snapshot()

    def record_review(self, record: ReviewRecord) -> None:
        """Persist a review record after task execution."""
        self._logger.debug("Recording review %s", record.id)
        record_dict = asdict(record)
        with log_span("memory.record_review", id=record.id):
            self._chroma_store.add_review(record)
            self._revision_logger.log("review", record.id, record_dict)
        emit_metric("memory.write", layer="review")
        publish_event("review.recorded", review=record_dict)
        self._publish_metrics_snapshot()

    def get_semantic_node(self, node_id: str) -> Optional[SemanticNode]:
        """Return a semantic node by id, or None when unavailable."""
        payload = self._chroma_store.get_semantic(node_id)
        if payload is None:
            return None
        return self._hydrate_semantic(payload)

    def list_semantic_nodes(self, limit: Optional[int] = None) -> List[SemanticNode]:
        """Return semantic nodes ordered by timestamp ascending, optionally limited."""
        payloads = self._chroma_store.list_layer("semantic")
        nodes: List[SemanticNode] = []
        for payload in payloads:
            node = self._hydrate_semantic(payload)
            if node is not None:
                nodes.append(node)
        nodes.sort(key=lambda item: item.timestamp)
        if limit is not None:
            if limit <= 0:
                return []
            return nodes[-limit:]
        return nodes

    def link_semantic_nodes(self, source_id: str, target_id: str, weight: float = 0.5) -> None:
        """Create or update a bidirectional relation between semantic nodes."""
        if source_id == target_id:
            return

        source = self.get_semantic_node(source_id)
        target = self.get_semantic_node(target_id)
        if source is None or target is None:
            self._logger.debug(
                "Unable to link semantic nodes %s -> %s: missing node.",
                source_id,
                target_id,
            )
            return

        weight_clamped = max(0.0, min(1.0, float(weight)))
        updated = False

        forward_key = f"{SEMANTIC_NODE_PREFIX}{target.id}"
        previous_forward = source.relations.get(forward_key)
        if previous_forward is None:
            source.relations[forward_key] = weight_clamped
            updated = True
        else:
            try:
                if abs(float(previous_forward) - weight_clamped) > 1e-6:
                    source.relations[forward_key] = weight_clamped
                    updated = True
            except (TypeError, ValueError):
                source.relations[forward_key] = weight_clamped
                updated = True

        reverse_key = f"{SEMANTIC_NODE_PREFIX}{source.id}"
        previous_reverse = target.relations.get(reverse_key)
        if previous_reverse is None:
            target.relations[reverse_key] = weight_clamped
            updated = True
        else:
            try:
                if abs(float(previous_reverse) - weight_clamped) > 1e-6:
                    target.relations[reverse_key] = weight_clamped
                    updated = True
            except (TypeError, ValueError):
                target.relations[reverse_key] = weight_clamped
                updated = True

        if not updated:
            return

        self.record_semantic(source)
        self.record_semantic(target)
        emit_metric(
            "memory.semantic.link",
            layer="semantic",
            source=source.id,
            target=target.id,
        )

    def get_semantic_neighbors(
        self,
        node_id: str,
        limit: int = 5,
    ) -> List[Tuple[SemanticNode, float]]:
        """Return neighbours for a semantic node ordered by relation weight."""
        node = self.get_semantic_node(node_id)
        if node is None:
            return []

        neighbours: List[Tuple[SemanticNode, float]] = []
        for relation_key, weight_raw in node.relations.items():
            if not relation_key.startswith(SEMANTIC_NODE_PREFIX):
                continue
            target_id = relation_key[len(SEMANTIC_NODE_PREFIX) :]
            neighbour = self.get_semantic_node(target_id)
            if neighbour is None:
                continue
            try:
                weight_value = float(weight_raw)
            except (TypeError, ValueError):
                continue
            neighbours.append((neighbour, weight_value))

        neighbours.sort(key=lambda item: item[1], reverse=True)
        if limit >= 0:
            return neighbours[:limit]
        return neighbours

    def list_layer(self, layer: str) -> List[Dict[str, object]]:
        """List stored items for the requested layer."""
        if layer not in {"episodic", "semantic", "review"}:
            raise MemoryError(f"Unsupported memory layer requested: {layer}")
        return self._chroma_store.list_layer(layer)

    def query_layer(self, layer: str, query: str, limit: int = 5) -> List[Dict[str, object]]:
        """Search stored items for the requested layer."""
        if layer not in {"episodic", "semantic", "review"}:
            raise MemoryError(f"Unsupported memory layer requested: {layer}")
        return self._chroma_store.query_layer(layer, query, limit)

    def list_working_items(self, pattern: str = "*") -> List[WorkingMemoryItem]:
        """List working memory records for UI inspection."""
        return self._redis_store.list_items(pattern)

    def get_revision_history(self, limit: int = 20) -> List[Dict[str, object]]:
        """Return the latest revision entries for audit or rollback tooling."""
        return self._revision_logger.history(limit)

    def verify_revision_log(self, limit_revision: Optional[int] = None) -> bool:
        """Validate stored revision hash chain integrity."""
        return self._revision_logger.verify(limit_revision)

    def replay_revision_state(
        self, layer: str, limit_revision: Optional[int] = None
    ) -> List[Dict[str, object]]:
        """Reconstruct the best-known state for *layer* up to *limit_revision*."""
        if layer not in {"episodic", "semantic", "review", "working"}:
            raise MemoryError(f"Unsupported revision replay layer: {layer}")
        return self._revision_logger.replay_layer(layer, limit_revision)

    def truncate_working_memory(self, max_items: int) -> int:
        """Trim working memory to at most *max_items* entries, oldest first."""
        if max_items <= 0:
            return 0

        items = self._redis_store.list_items()
        if len(items) <= max_items:
            return 0

        items.sort(key=lambda item: item.created_at)
        excess = len(items) - max_items
        pruned = 0
        with log_span(
            "memory.truncate_working",
            total=len(items),
            limit=max_items,
        ):
            for item in items[:excess]:
                try:
                    self._redis_store.delete(item.key)
                except MemoryError as exc:
                    self._logger.error("Failed to prune working memory %s: %s", item.key, exc)
                    continue
                self._revision_logger.log(
                    "working",
                    item.key,
                    {
                        "action": "delete",
                        "reason": "drift_mitigation",
                        "created_at": item.created_at.isoformat(),
                    },
                )
                pruned += 1

        if pruned:
            self._logger.info(
                "Pruned %s working memory entries (limit=%s).",
                pruned,
                max_items,
            )
            emit_metric("memory.working.pruned", value=pruned)
        return pruned

    def decay_semantic_relations(self, decay: float) -> int:
        """Decay semantic relation weights to dampen stale context."""
        if decay <= 0:
            return 0
        if decay >= 1.0:
            return 0

        updated = 0
        nodes = self.list_semantic_nodes()
        with log_span(
            "memory.decay_semantic_relations",
            nodes=len(nodes),
            decay=decay,
        ):
            for node in nodes:
                changed = False
                for relation_key in list(node.relations.keys()):
                    if not relation_key.startswith(SEMANTIC_NODE_PREFIX):
                        continue
                    raw_weight = node.relations[relation_key]
                    try:
                        new_weight = float(raw_weight) * decay
                    except (TypeError, ValueError):
                        del node.relations[relation_key]
                        changed = True
                        continue
                    if new_weight < SEMANTIC_RELATION_MIN_THRESHOLD:
                        del node.relations[relation_key]
                        changed = True
                    else:
                        new_weight = round(new_weight, 4)
                        if abs(new_weight - float(raw_weight)) > 1e-5:
                            node.relations[relation_key] = new_weight
                            changed = True
                if changed:
                    updated += 1
                    try:
                        self.record_semantic(node)
                    except MemoryError as exc:
                        self._logger.error(
                            "Failed to persist decayed relations for %s: %s",
                            node.id,
                            exc,
                        )

        if updated:
            self._logger.info(
                "Decayed semantic relations for %s nodes (decay=%.2f).",
                updated,
                decay,
            )
            emit_metric("memory.semantic.decayed_nodes", value=updated, decay=decay)
        return updated

    def apply_drift_mitigation(
        self,
        *,
        task_id: Optional[str] = None,
        max_working_items: int = MAX_WORKING_MEMORY_ITEMS,
        relation_decay: float = SEMANTIC_RELATION_DECAY,
    ) -> Dict[str, object]:
        """Apply drift mitigation heuristics and return a summary of actions."""

        summary: Dict[str, object] = {}

        with log_span(
            "memory.apply_drift_mitigation",
            task_id=task_id or "n/a",
            max_working=max_working_items,
            relation_decay=relation_decay,
        ):
            pruned = self.truncate_working_memory(max_working_items)
            if pruned:
                summary["working_pruned"] = pruned

            if relation_decay < 1.0:
                decayed = self.decay_semantic_relations(relation_decay)
                if decayed:
                    summary["semantic_nodes_updated"] = decayed

        context = f" for task {task_id}" if task_id else ""
        if summary:
            self._logger.info("Drift mitigation applied%s: %s", context, summary)
            emit_metric("memory.drift.mitigation", value=1, **summary)
            publish_event(
                "memory.drift.mitigation",
                summary=summary,
                task_id=task_id or "n/a",
            )
        else:
            self._logger.debug("Drift mitigation produced no changes%s.", context)

        self._publish_metrics_snapshot()
        return summary

    def snapshot_metrics(self) -> Dict[str, int]:
        """Snapshot counts across memory layers for telemetry."""
        metrics: Dict[str, int] = {}
        working_items: List[WorkingMemoryItem] = []
        try:
            working_items = self.list_working_items()
            metrics["working_items"] = len(working_items)
            metrics["drift_advisories"] = sum(
                1 for item in working_items if item.key.endswith(":drift")
            )
        except Exception as exc:  # pragma: no cover - defensive path
            self._logger.debug("Unable to enumerate working memory for metrics: %s", exc)

        for layer in ("episodic", "semantic", "review"):
            key = f"{layer}_records"
            try:
                metrics[key] = len(self.list_layer(layer))
            except MemoryError as exc:
                self._logger.debug("Unable to enumerate %s layer for metrics: %s", layer, exc)

        return metrics

    def _publish_metrics_snapshot(self) -> None:
        try:
            metrics = self.snapshot_metrics()
        except Exception as exc:  # pragma: no cover - defensive path
            self._logger.debug("Skipping telemetry metrics snapshot: %s", exc)
            return
        if metrics:
            publish_event("memory.metrics", **metrics)

    def _hydrate_semantic(self, payload: Dict[str, object]) -> Optional[SemanticNode]:
        try:
            node_id = str(payload["id"])
            label = str(payload.get("label", "")).strip()
            definition = str(payload.get("definition", "")).strip()
        except KeyError:
            self._logger.debug("Semantic payload missing fields: %s", payload)
            return None

        timestamp = RedisMemoryStore._coerce_timestamp(payload.get("timestamp"))

        sources_raw = payload.get("sources", [])
        if isinstance(sources_raw, list):
            sources = [str(item) for item in sources_raw]
        elif sources_raw:
            sources = [str(sources_raw)]
        else:
            sources = []

        relations_raw = payload.get("relations", {})
        relations: Dict[str, float] = {}
        if isinstance(relations_raw, dict):
            for key, value in relations_raw.items():
                key_str = str(key)
                try:
                    relations[key_str] = float(value)
                except (TypeError, ValueError):
                    continue

        return SemanticNode(
            id=node_id,
            label=label,
            definition=definition,
            sources=sources,
            timestamp=timestamp,
            relations=relations,
        )
