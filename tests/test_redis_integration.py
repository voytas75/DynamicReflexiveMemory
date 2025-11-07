"""Integration-style tests exercising the Redis working memory store.

Updates:
    v0.1 - 2025-11-08 - Added dockerised Redis coverage for TTL expiry,
        reconnection, and fallback behaviour.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Iterator

import pytest
import redis

from config.settings import load_app_config
from core.memory_manager import RedisMemoryStore
from models.memory import WorkingMemoryItem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"
REDIS_SERVICE = "redis"
TEST_REDIS_PORT = 6379


class _RedisServiceController:
    """Helper for managing the dockerised Redis service during tests."""

    def __init__(self, compose_file: Path) -> None:
        self._compose_file = compose_file

    def run(self, *args: str) -> None:
        command = [
            "docker",
            "compose",
            "-f",
            str(self._compose_file),
            *args,
        ]
        subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))

    def up(self) -> None:
        self.run("up", "-d", REDIS_SERVICE)

    def stop(self) -> None:
        self.run("stop", REDIS_SERVICE)

    def start(self) -> None:
        self.run("start", REDIS_SERVICE)

    def down(self) -> None:
        self.run("rm", "-sf", REDIS_SERVICE)

    def ensure_ready(self, timeout: float = 15.0) -> None:
        client = redis.Redis(host="localhost", port=6379, db=5, socket_timeout=1)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if client.ping():
                    return
            except redis.exceptions.ConnectionError:
                time.sleep(0.25)
        raise RuntimeError("Redis service did not become ready in time")


@pytest.fixture(scope="module")
def redis_service() -> Iterator[_RedisServiceController]:
    if shutil.which("docker") is None:
        pytest.skip("Docker is required for Redis integration tests.")
    if not COMPOSE_FILE.exists():
        pytest.skip("docker-compose.yml not available for Redis integration tests.")

    if _is_port_in_use("127.0.0.1", TEST_REDIS_PORT):
        pytest.skip(
            f"Port {TEST_REDIS_PORT} is already in use; stop the local Redis service or free the port before running integration tests."
        )

    controller = _RedisServiceController(COMPOSE_FILE)
    controller.up()
    try:
        try:
            controller.ensure_ready()
        except RuntimeError as exc:
            pytest.skip(str(exc))
        yield controller
    finally:
        controller.down()


def _is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        result = sock.connect_ex((host, port))
        return result == 0


def _build_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ttl: int = 3) -> RedisMemoryStore:
    monkeypatch.setenv("DRM_MEMORY_LOG_PATH", str(tmp_path / "revisions.jsonl"))
    config = load_app_config()
    config.memory.redis.port = TEST_REDIS_PORT
    config.memory.redis.ttl_seconds = ttl
    return RedisMemoryStore(config)


@pytest.mark.integration
def test_redis_working_memory_ttl_expiry(
    redis_service: _RedisServiceController,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = _build_store(monkeypatch, tmp_path, ttl=1)
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
    store = _build_store(monkeypatch, tmp_path)

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
