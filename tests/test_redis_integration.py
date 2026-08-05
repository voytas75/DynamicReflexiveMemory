"""Integration tests exercising Redis working-memory persistence.

Updates:
    v0.1 - 2025-11-08 - Added Dockerised Redis coverage for TTL expiry,
        reconnection, and fallback behaviour.
    v0.2 - 2026-08-05 - Added deterministic fallback TTL and reconnect migration
        coverage without a Docker dependency.
    v0.3 - 2026-08-05 - Isolated real Redis tests on a dynamic loopback port.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import pytest
import redis

from config.settings import load_app_config
from core.memory_manager import RedisMemoryStore
from models.memory import WorkingMemoryItem

REDIS_IMAGE = "redis:7.4-alpine"
REDIS_CONTAINER_PORT = 6379


class _RedisServiceController:
    """Manage an isolated Redis container on a Docker-assigned loopback port."""

    def __init__(self) -> None:
        self._container_name = f"drm-test-redis-{uuid4().hex}"
        self._port: int | None = None

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError(
                "Redis test port is unavailable before container startup."
            )
        return self._port

    def run(
        self, *args: str, capture_output: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["docker", *args],
            check=True,
            capture_output=capture_output,
            text=True,
        )

    def up(self) -> None:
        self.run(
            "run",
            "--detach",
            "--name",
            self._container_name,
            "--publish",
            f"127.0.0.1::{REDIS_CONTAINER_PORT}",
            REDIS_IMAGE,
            "redis-server",
            "--save",
            "",
            "--appendonly",
            "no",
        )
        self._port = self._published_port()

    def _published_port(self) -> int:
        published = self.run(
            "port",
            self._container_name,
            f"{REDIS_CONTAINER_PORT}/tcp",
            capture_output=True,
        ).stdout.strip()
        if not published:
            raise RuntimeError("Docker did not report a published Redis test port.")
        try:
            return int(published.rsplit(":", maxsplit=1)[1])
        except ValueError as exc:
            raise RuntimeError(
                f"Unexpected Docker port mapping: {published!r}"
            ) from exc

    def stop(self) -> None:
        self.run("stop", self._container_name)

    def start(self) -> None:
        self.run("start", self._container_name)
        self._port = self._published_port()

    def down(self) -> None:
        subprocess.run(
            ["docker", "rm", "--force", self._container_name],
            check=False,
            capture_output=True,
            text=True,
        )

    def ensure_ready(self, timeout: float = 15.0) -> None:
        client = redis.Redis(
            host="127.0.0.1",
            port=self.port,
            db=5,
            socket_timeout=1,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if client.ping():
                    return
            except redis.exceptions.ConnectionError:
                time.sleep(0.25)
        raise RuntimeError(
            "Isolated Redis test container did not become ready in time."
        )


@pytest.fixture(scope="module")
def redis_service() -> Iterator[_RedisServiceController]:
    if shutil.which("docker") is None:
        pytest.skip("Docker client is required for Redis integration tests.")

    controller = _RedisServiceController()
    try:
        controller.up()
        controller.ensure_ready()
        yield controller
    finally:
        controller.down()


def _build_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    port: int,
    ttl: int = 3,
) -> RedisMemoryStore:
    monkeypatch.setenv("DRM_MEMORY_LOG_PATH", str(tmp_path / "revisions.jsonl"))
    config = load_app_config()
    config.memory.redis.host = "127.0.0.1"
    config.memory.redis.port = port
    config.memory.redis.ttl_seconds = ttl
    return RedisMemoryStore(config)


@pytest.mark.integration
def test_redis_working_memory_ttl_expiry(
    redis_service: _RedisServiceController,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = _build_store(monkeypatch, tmp_path, port=redis_service.port, ttl=1)
    item = WorkingMemoryItem(
        key="ttl:test",
        payload={"value": "ephemeral"},
        ttl_seconds=1,
    )

    store.put(item)
    assert store.get(item.key) is not None

    time.sleep(2)
    assert store.get(item.key) is None


@pytest.mark.integration
def test_redis_store_recovers_after_restart(
    redis_service: _RedisServiceController,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = _build_store(monkeypatch, tmp_path, port=redis_service.port)

    first_item = WorkingMemoryItem(
        key="reconnect:initial",
        payload={"attempt": 1},
        ttl_seconds=10,
    )
    store.put(first_item)
    assert store.get(first_item.key) is not None

    redis_service.stop()
    time.sleep(1.0)
    redis_service.start()
    redis_service.ensure_ready()

    second_item = WorkingMemoryItem(
        key="reconnect:subsequent",
        payload={"attempt": 2},
        ttl_seconds=10,
    )

    store.put(second_item)
    assert store.get(second_item.key) is not None


def test_redis_fallback_store_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DRM_MEMORY_LOG_PATH", str(tmp_path / "fallback.jsonl"))
    config = load_app_config()
    config.memory.redis.port = 6390
    config.memory.redis.ttl_seconds = 3

    store = RedisMemoryStore(config)

    item = WorkingMemoryItem(
        key="fallback:item",
        payload={"value": "cached"},
        ttl_seconds=3,
    )
    store.put(item)

    retrieved = store.get(item.key)
    assert retrieved is not None
    assert retrieved.payload == item.payload


class _RecoveringRedisClient:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.fail_get = False

    def ping(self) -> bool:
        return True

    def setex(self, *, name: str, time: int, value: str) -> None:
        self.values[name] = value
        self.ttls[name] = time

    def get(self, key: str) -> bytes | None:
        if self.fail_get:
            raise OSError("Redis unavailable")
        value = self.values.get(key)
        return value.encode("utf-8") if value is not None else None


class _RecoveringRedisModule:
    def __init__(self) -> None:
        self.available = False
        self.client = _RecoveringRedisClient()

    def Redis(self, **_: object) -> _RecoveringRedisClient:
        if not self.available:
            raise OSError("Redis unavailable")
        return self.client


def test_redis_fallback_expires_items_without_redis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("core.memory_manager.redis_module", None)
    store = _build_store(monkeypatch, tmp_path, port=6390, ttl=1)
    expired = WorkingMemoryItem(
        key="fallback:expired",
        payload={"value": "stale"},
        ttl_seconds=1,
        created_at=datetime.now(timezone.utc) - timedelta(seconds=2),
    )

    store.put(expired)

    assert store.get(expired.key) is None
    assert store.list_items() == []


def test_redis_reconnect_migrates_live_fallback_items(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    redis_stub = _RecoveringRedisModule()
    monkeypatch.setattr("core.memory_manager.redis_module", redis_stub)
    store = _build_store(monkeypatch, tmp_path, port=6390, ttl=10)
    item = WorkingMemoryItem(
        key="fallback:recover",
        payload={"value": "preserved"},
        ttl_seconds=10,
    )
    store.put(item)

    redis_stub.available = True

    recovered = store.get(item.key)

    assert recovered is not None
    assert recovered.payload == item.payload
    assert 1 <= redis_stub.client.ttls[item.key] <= item.ttl_seconds
    redis_stub.client.fail_get = True
    assert store.get(item.key) is None
