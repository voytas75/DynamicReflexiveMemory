"""Append-only revision-log persistence for Dynamic Reflexive Memory.

Updates:
    v1.0 - 2026-08-05 - Extracted revision logging from core.memory_manager
        without changing its public re-export contract.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, cast

LOGGER = logging.getLogger("drm.memory")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVISION_LOG_ENV = "DRM_MEMORY_LOG_PATH"
REVISION_LOG_MODE_ENV = "DRM_MEMORY_AUDIT_LOG_MODE"
FULL_REVISION_LOG_MODE = "full"
REVISION_LOG_RETENTION_DAYS = 30
DEFAULT_REVISION_LOG = PROJECT_ROOT / "data" / "logs" / "memory_revisions.jsonl"


class MemoryRevisionLogger:
    """Append-only revision log supporting rollback-aware auditing."""

    def __init__(self, path_override: Optional[Path] = None) -> None:
        log_path = self._resolve_log_path(path_override)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = log_path
        self._mode = self._resolve_log_mode()
        self._lock = Lock()
        self._prune_expired_records()
        self._revision, self._tail_hash = self._load_log_tail()

    def log(self, layer: str, identifier: str, payload: Dict[str, object]) -> None:
        """Append a revision entry capturing the memory mutation."""
        with self._lock:
            record: Dict[str, object] = {
                "layer": layer,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if self._mode == FULL_REVISION_LOG_MODE:
                record["id"] = identifier
                record["payload"] = payload
            else:
                record["id_digest"] = self._digest_value(identifier)
                record["payload"] = {"redacted": True}
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
                LOGGER.debug("Skipping malformed revision log entry.")
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
            record_hash = record.get("hash")
            expected_prev = record_hash if isinstance(record_hash, str) else None
        return True

    def replay_layer(
        self, layer: str, limit_revision: Optional[int] = None
    ) -> List[Dict[str, object]]:
        """Reconstruct the layer state up to *limit_revision* by replaying the log."""
        state: Dict[str, Dict[str, object]] = {}
        for record in self._iter_records(limit_revision):
            if record.get("layer") != layer:
                continue
            identifier = str(record.get("id") or record.get("id_digest") or "")
            if not identifier:
                continue
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

    def _resolve_log_mode(self) -> str:
        """Return the explicit full-audit opt-in or the safe redacted default."""
        mode = os.getenv(REVISION_LOG_MODE_ENV, "redacted").strip().lower()
        if mode in {"", "redacted"}:
            return "redacted"
        if mode == FULL_REVISION_LOG_MODE:
            return FULL_REVISION_LOG_MODE
        LOGGER.warning(
            "Unknown %s mode; using redacted audit records.", REVISION_LOG_MODE_ENV
        )
        return "redacted"

    def _prune_expired_records(self) -> None:
        """Remove expired records and preserve a valid hash chain for the remainder."""
        if not self._path.exists():
            return

        cutoff = datetime.now(timezone.utc) - timedelta(
            days=REVISION_LOG_RETENTION_DAYS
        )
        retained: List[Dict[str, object]] = []
        expired_count = 0
        malformed_count = 0

        with self._lock:
            try:
                with self._path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            parsed_record = json.loads(line)
                        except json.JSONDecodeError:
                            malformed_count += 1
                            continue
                        if not isinstance(parsed_record, dict):
                            malformed_count += 1
                            continue
                        record = cast(Dict[str, object], parsed_record)
                        if self._is_expired_record(record, cutoff):
                            expired_count += 1
                        else:
                            retained.append(record)
            except OSError as exc:
                LOGGER.warning("Unable to apply revision-log retention: %s", exc)
                return

            if not expired_count and not malformed_count:
                return

            try:
                self._rewrite_records(retained)
            except OSError as exc:
                LOGGER.warning("Unable to rewrite retained revision log: %s", exc)
                return

        LOGGER.info(
            "Applied %s-day revision-log retention: pruned=%s malformed=%s.",
            REVISION_LOG_RETENTION_DAYS,
            expired_count,
            malformed_count,
        )

    @staticmethod
    def _is_expired_record(record: Dict[str, object], cutoff: datetime) -> bool:
        timestamp = record.get("timestamp")
        if not isinstance(timestamp, str):
            return False
        try:
            parsed = datetime.fromisoformat(timestamp)
        except ValueError:
            return False
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed < cutoff

    def _rewrite_records(self, records: Sequence[Dict[str, object]]) -> None:
        """Atomically rewrite retained records with a fresh valid hash chain."""
        previous_hash: Optional[str] = None
        rewritten: List[Dict[str, object]] = []
        for record in records:
            updated = dict(record)
            updated.pop("hash", None)
            updated["prev_hash"] = previous_hash
            updated["hash"] = self._calculate_hash(updated)
            previous_hash = updated["hash"]
            rewritten.append(updated)

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self._path.parent,
            prefix=f".{self._path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for record in rewritten:
                handle.write(json.dumps(record, default=str))
                handle.write("\n")
        try:
            temporary_path.replace(self._path)
        except OSError:
            temporary_path.unlink(missing_ok=True)
            raise

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

    @staticmethod
    def _digest_value(value: object) -> str:
        """Return an opaque stable identifier for redacted audit records."""
        canonical = json.dumps(value, default=str, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
