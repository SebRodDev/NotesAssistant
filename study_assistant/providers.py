"""Which model backend answers: the Gemini API (cloud) or Ollama (on this computer)."""

from __future__ import annotations

from .gemini import (
    DEFAULT_GEMINI_CHAT,
    DEFAULT_GEMINI_EMBEDDING,
    KEY_URL,
    GeminiClient,
    best_model,
    get_api_key,
    is_gemini_chat_model,
)
from .ollama import DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL, DEFAULT_HOST, DEFAULT_VISION_MODEL, OllamaClient

GEMINI, OLLAMA = "gemini", "ollama"
DEFAULT_PROVIDER = GEMINI
PROVIDER_LABELS = {GEMINI: "Gemini API · Google's servers", OLLAMA: "Ollama · runs on this Mac"}
DEFAULTS = {
    GEMINI: {"chat_model": DEFAULT_GEMINI_CHAT, "embedding_model": DEFAULT_GEMINI_EMBEDDING,
             "vision_model": DEFAULT_GEMINI_CHAT, "handwriting": False},
    OLLAMA: {"chat_model": DEFAULT_CHAT_MODEL, "embedding_model": DEFAULT_EMBEDDING_MODEL,
             "vision_model": DEFAULT_VISION_MODEL, "handwriting": True},
}
# Ollama keeps the original "models/…" keys so switching back restores the earlier local setup.
_PREFIX = {GEMINI: "gemini", OLLAMA: "models"}
_KEYS = {"chat_model": "chat", "embedding_model": "embedding", "vision_model": "vision", "handwriting": "handwriting"}
# How often to check the backend: a local server is free to poll, an API isn't.
POLL_SECONDS = {GEMINI: 120.0, OLLAMA: 4.0}
EMBED_BATCH = {GEMINI: 100, OLLAMA: 16}

__all__ = ["GEMINI", "OLLAMA", "KEY_URL", "PROVIDER_LABELS", "best_model", "get_api_key", "load_health",
           "load_settings", "make_client", "model_fits", "save_health", "save_settings"]


def load_settings(settings, provider: str | None = None) -> dict:
    """The active provider (or the one named) and its models, from QSettings."""
    provider = provider or settings.value("models/provider", DEFAULT_PROVIDER, type=str)
    if provider not in DEFAULTS:
        provider = DEFAULT_PROVIDER
    values: dict = {"provider": provider,
                    "host": settings.value("models/host", DEFAULT_HOST, type=str) or DEFAULT_HOST,
                    "api_root": settings.value("gemini/api_root", "", type=str)}
    for field, key in _KEYS.items():
        default = DEFAULTS[provider][field]
        if field == "handwriting":
            values[field] = settings.value(f"{_PREFIX[provider]}/{key}", default, type=bool)
        else:
            stored = settings.value(f"{_PREFIX[provider]}/{key}", None)
            # vision may be saved as "" on purpose (off); chat and embedding always need a model
            values[field] = (default if stored is None else str(stored)) if field == "vision_model" \
                else (str(stored) if stored else default)
    return values


def save_settings(settings, values: dict) -> None:
    provider = values.get("provider", DEFAULT_PROVIDER)
    settings.setValue("models/provider", provider)
    if values.get("host"):
        settings.setValue("models/host", values["host"])
    for field, key in _KEYS.items():
        if field in values:
            settings.setValue(f"{_PREFIX[provider]}/{key}", values[field])


def load_health(settings) -> dict:
    """Restore the last model check's results (which Gemini models answer) into the Gemini client."""
    import json

    from .gemini import set_health

    try:
        results = json.loads(settings.value("gemini/health", "", type=str) or "{}")
    except ValueError:
        results = {}
    set_health(results if isinstance(results, dict) else {})
    return results


def save_health(settings, results: dict) -> None:
    """Merge a model check's results into the saved ones (Cortex and tools/check_gemini.py share them)."""
    import json

    merged = {**load_health(settings), **results}
    settings.setValue("gemini/health", json.dumps(merged))
    from .gemini import set_health

    set_health(merged)


def make_client(values: dict):
    """A client for the chosen provider. `values` comes from load_settings()."""
    if values.get("provider", DEFAULT_PROVIDER) == GEMINI:
        kwargs = {"api_root": values["api_root"]} if values.get("api_root") else {}
        return GeminiClient(values.get("api_key"), **kwargs)
    return OllamaClient(values.get("host") or DEFAULT_HOST)


def model_fits(name: str, provider: str) -> bool:
    """Whether a saved model name belongs to this provider (agents may still name a local model)."""
    if not name:
        return True
    gemini_name = ":" not in name and is_gemini_chat_model(name)
    return gemini_name if provider == GEMINI else not gemini_name
