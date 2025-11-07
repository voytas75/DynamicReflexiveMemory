"""Tests for startup diagnostics helper functions.

Updates: v0.1 - 2025-11-06 - Added coverage for Redis/Chroma fallbacks and
    health failure handling.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from config import settings
from core.exceptions import HealthCheckError
from core.health import run_startup_checks


def _load_config(tmp_path: Path) -> settings.AppConfig:
    source = Path(__file__).resolve().parent.parent / "config" / "config.json"
    config_path = tmp_path / "config.json"
    config_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return settings.load_app_config(config_path)


def _install_stub_modules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, redis_ping_ok: bool = True) -> None:
    class _RedisClient:
        def __init__(self, **_: object) -> None:
            pass

        def ping(self) -> bool:
            if redis_ping_ok:
                return True
            raise RuntimeError("redis ping failed")

    redis_module = types.SimpleNamespace(Redis=_RedisClient)
    chroma_module = _build_chroma_stub(tmp_path)
    litellm_module = types.SimpleNamespace(__version__="1.61.15")

    monkeypatch.setitem(sys.modules, "redis", redis_module)
    monkeypatch.setitem(sys.modules, "chromadb", chroma_module)
    monkeypatch.setitem(sys.modules, "litellm", litellm_module)

    monkeypatch.setattr("core.health.metadata.version", lambda _: "1.61.15")


def _build_chroma_stub(tmp_path: Path) -> types.SimpleNamespace:
    class _Client:
        def __init__(self, path: str) -> None:
            self._path = path

        def list_collections(self) -> list[object]:  # pragma: no cover - empty stub
            return []

    return types.SimpleNamespace(PersistentClient=_Client)


def test_health_checks_with_stubbed_dependencies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    config.memory.chromadb.persist_directory = str(tmp_path / "chroma")

    _install_stub_modules(monkeypatch, tmp_path)

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "embedding")

    warnings = run_startup_checks(config)
    assert warnings == []


def test_health_checks_raise_on_redis_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    config.memory.chromadb.persist_directory = str(tmp_path / "chroma")

    _install_stub_modules(monkeypatch, tmp_path, redis_ping_ok=False)

    with pytest.raises(HealthCheckError):
        run_startup_checks(config)


def test_health_checks_warn_when_standard_embedding_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    config.memory.chromadb.persist_directory = str(tmp_path / "chroma")
    config.embedding.model = "text-embedding-3-large"

    _install_stub_modules(monkeypatch, tmp_path)

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.delenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", raising=False)

    warnings = run_startup_checks(config)
    assert warnings
    assert any("embedding deployment" in warning for warning in warnings)


def test_health_checks_skip_warning_for_custom_embedding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    config.memory.chromadb.persist_directory = str(tmp_path / "chroma")

    _install_stub_modules(monkeypatch, tmp_path)

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.com")
    monkeypatch.delenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", raising=False)

    warnings = run_startup_checks(config)
    assert warnings == []
