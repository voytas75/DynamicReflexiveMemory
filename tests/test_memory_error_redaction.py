"""Regression tests for safe MemoryManager storage/provider error reporting."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from config.settings import EmbeddingConfig
from core.exceptions import MemoryError
from core.memory_manager import ChromaMemoryStore


def _chroma_store() -> ChromaMemoryStore:
    """Build a minimal Chroma store instance without local runtime services."""
    store = ChromaMemoryStore.__new__(ChromaMemoryStore)
    store._logger = logging.getLogger("drm.memory.chroma")
    store._collection = None
    store._fallback = {
        "episodic": {},
        "semantic": {},
        "review": {},
        "analytics": {},
    }
    return store


def test_azure_embedding_client_initialisation_redacts_failure_details(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = "azure-embedding-client-secret-marker"

    class ClientFailure(Exception):
        """Synthetic provider failure carrying sensitive-looking detail."""

    class FailingAzureClient:
        def __init__(self, **_: object) -> None:
            raise ClientFailure(marker)

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setattr("core.memory_manager.AzureOpenAI", FailingAzureClient)

    caplog.set_level(logging.ERROR, logger="drm.memory.chroma")
    embedding = _chroma_store()._build_azure_embedding_function(
        EmbeddingConfig(provider="azure", model="embedding-model")
    )

    assert embedding is None
    assert marker not in caplog.text
    assert "error_type=ClientFailure" in caplog.text


def test_azure_embedding_request_redacts_failure_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "azure-embedding-request-prompt-marker"

    class ProviderFailure(Exception):
        """Synthetic provider failure carrying sensitive-looking detail."""

    class FailingEmbeddings:
        def create(self, **_: object) -> Any:
            raise ProviderFailure(marker)

    class AzureClient:
        def __init__(self, **_: object) -> None:
            self.embeddings = FailingEmbeddings()

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setattr("core.memory_manager.AzureOpenAI", AzureClient)

    embedding = _chroma_store()._build_azure_embedding_function(
        EmbeddingConfig(provider="azure", model="embedding-model")
    )
    assert embedding is not None

    with pytest.raises(MemoryError, match="Azure embedding request failed") as error:
        embedding.embed_query(marker)

    assert marker not in str(error.value)
    assert error.value.__cause__ is None


def test_chroma_fallback_redacts_permission_failure_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = "chroma-fallback-result-marker"

    class FailingCollection:
        def upsert(self, **_: object) -> None:
            raise PermissionError(f"permission denied: {marker}")

    store = _chroma_store()
    store._collection = FailingCollection()
    caplog.set_level(logging.WARNING, logger="drm.memory.chroma")

    store._store_in_collection("episodic", "episode-id", {"content": marker})

    assert store._collection is None
    assert marker not in caplog.text
    assert "error_type=PermissionError" in caplog.text


def test_chroma_write_error_redacts_failure_details() -> None:
    marker = "chroma-write-provider-secret-marker"

    class ProviderFailure(Exception):
        """Synthetic storage failure carrying sensitive-looking detail."""

    class FailingCollection:
        def upsert(self, **_: object) -> None:
            raise ProviderFailure(marker)

    store = _chroma_store()
    store._collection = FailingCollection()

    with pytest.raises(
        MemoryError, match="Failed to persist episodic memory entry"
    ) as error:
        store._store_in_collection("episodic", "episode-id", {"content": marker})

    assert marker not in str(error.value)
    assert error.value.__cause__ is None
