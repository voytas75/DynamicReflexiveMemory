"""Shared LiteLLM provider routing for task execution and automated review.

Updates:
    v0.1 - 2026-08-05 - Centralised Azure/Ollama model routing and WSL-aware
        Ollama endpoint resolution for task and review parity.
"""

from __future__ import annotations

import ipaddress
import subprocess
from functools import lru_cache
from os import getenv
from typing import Dict, Optional, Tuple

DEFAULT_OLLAMA_BASE = "http://localhost:11434"


class ProviderRoutingError(ValueError):
    """Raised when configured provider routing cannot be resolved safely."""


def resolve_provider_configuration(
    provider: str, model_name: str
) -> Tuple[str, Dict[str, object]]:
    """Return a LiteLLM model identifier and provider kwargs for *provider*."""
    provider_lower = provider.lower()

    if provider_lower == "azure":
        api_key = getenv("AZURE_OPENAI_API_KEY")
        endpoint = getenv("AZURE_OPENAI_ENDPOINT")
        api_version = getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        if not api_key or not endpoint:
            raise ProviderRoutingError(
                "Azure OpenAI credentials missing. Set AZURE_OPENAI_API_KEY and "
                "AZURE_OPENAI_ENDPOINT environment variables."
            )
        base = endpoint.rstrip("/")
        resolved_model = resolve_provider_model(provider, model_name)
        return resolved_model, {
            "api_key": api_key,
            "api_base": base,
            "base_url": base,
            "api_version": api_version,
            "custom_llm_provider": "azure",
        }

    if provider_lower == "ollama":
        base_url = resolve_ollama_base_url()
        resolved_model = resolve_provider_model(provider, model_name)
        return resolved_model, {
            "base_url": base_url,
            "api_base": base_url,
            "custom_llm_provider": "ollama",
        }

    return model_name, {}


def resolve_provider_model(provider: str, model_name: str) -> str:
    """Normalise provider-specific model identifiers without resolving credentials."""
    provider_lower = provider.lower()
    if provider_lower == "azure":
        return model_name if model_name.startswith("azure/") else f"azure/{model_name}"
    if provider_lower == "ollama":
        return (
            model_name
            if model_name.startswith(("ollama/", "ollama_chat/"))
            else f"ollama/{model_name}"
        )
    return model_name


def resolve_ollama_base_url() -> str:
    """Resolve the Ollama endpoint using override, WSL host, then localhost."""
    provided = getenv("OLLAMA_BASE_URL")
    if provided:
        return provided.rstrip("/")

    detected_host = _detect_windows_host_ip()
    if detected_host:
        return f"http://{detected_host}:11434"
    return DEFAULT_OLLAMA_BASE


@lru_cache(maxsize=1)
def _detect_windows_host_ip(timeout_seconds: float = 2.0) -> Optional[str]:
    """Return the WSL2 Windows-host default-route IP when it is available."""
    try:
        with open("/proc/version", encoding="utf-8") as version_file:
            if "microsoft" not in version_file.read().lower():
                return None
    except OSError:
        return None

    try:
        result = subprocess.run(
            ["ip", "route"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0 or not result.stdout:
        return None

    for line in result.stdout.splitlines():
        tokens = line.strip().split()
        if not tokens or tokens[0] != "default":
            continue
        try:
            via_index = tokens.index("via")
        except ValueError:
            continue
        if via_index + 1 >= len(tokens):
            continue
        candidate = tokens[via_index + 1]
        if _is_ipv4(candidate):
            return candidate

    return None


def _is_ipv4(value: str) -> bool:
    """Return ``True`` when *value* is a valid IPv4 address."""
    try:
        return isinstance(ipaddress.ip_address(value), ipaddress.IPv4Address)
    except ValueError:
        return False
