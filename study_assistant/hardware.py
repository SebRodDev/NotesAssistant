"""Match local models to what this computer can actually run.

The catalogs are ordered best first. That order is a judgement call (newer and larger first), so
update it as new models ship. Sizes are the default download sizes listed on ollama.com/library.
"""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass

GB = 1_000_000_000
GIB = 2**30

CHAT_CATALOG: tuple[tuple[str, float], ...] = (
    ("qwen3.6:35b", 23), ("qwen3.6:27b", 19), ("gemma4:31b", 20), ("gemma4:26b", 18), ("qwen3:32b", 20),
    ("qwen3:30b", 19), ("gemma4:12b", 8.0), ("qwen3:14b", 9.3), ("qwen3:8b", 5.2), ("qwen3:4b", 2.5),
    ("qwen3:1.7b", 1.4), ("qwen3:0.6b", 0.52),
)
VISION_CATALOG: tuple[tuple[str, float], ...] = (
    ("qwen3.6:35b", 23), ("qwen3.6:27b", 19), ("gemma4:31b", 20), ("gemma4:26b", 18), ("gemma4:12b", 8.0),
    ("qwen3-vl:8b", 6.2), ("qwen2.5vl:7b", 6.0), ("qwen3-vl:4b", 3.3), ("gemma3:4b", 3.4), ("qwen3-vl:2b", 1.9),
)
EMBEDDING_CATALOG: tuple[tuple[str, float], ...] = (
    ("nomic-embed-text", 0.27), ("embeddinggemma", 0.62), ("mxbai-embed-large", 0.67),
    ("snowflake-arctic-embed", 0.67), ("bge-m3", 1.2),
)
CATALOGS = {"chat": CHAT_CATALOG, "vision": VISION_CATALOG, "embedding": EMBEDDING_CATALOG}

# Below this the notes, chat history, and answer no longer fit together in one prompt.
MIN_USEFUL_CONTEXT = 8192
CONTEXT_CHOICES = (16384, MIN_USEFUL_CONTEXT, 4096)
_RUNTIME_OVERHEAD = 0.7 * GB  # compute buffers on top of the weights


def is_embedding_model(name: str) -> bool:
    lowered = name.casefold()
    return any(key in lowered for key in ("embed", "bge-", "minilm", "arctic-embed"))


def _kv_bytes_per_token(model_size: float) -> int:
    """Rough KV-cache cost at f16, from the layer counts of the 4B-14B and 27B+ model classes."""
    return 150_000 if model_size < 10 * GB else 260_000


@dataclass(frozen=True)
class Hardware:
    memory_bytes: int
    apple_silicon: bool = True

    @property
    def model_budget(self) -> float:
        """Bytes a model may occupy while leaving macOS and open apps enough room.

        Metal only lets the GPU wire part of unified memory, so anything larger spills to the CPU.
        Without Apple Silicon inference runs on the CPU, where big models are too slow to be useful.
        """
        share = 0.625 if self.memory_bytes <= 36 * GIB else 0.75
        budget = self.memory_bytes * share
        return budget if self.apple_silicon else min(budget, 6 * GB)

    def _runs(self, model_size: float, context: int) -> bool:
        needed = model_size + _RUNTIME_OVERHEAD + _kv_bytes_per_token(model_size) * context
        return needed <= self.model_budget

    def fits(self, model_size: float) -> bool:
        return self._runs(model_size, MIN_USEFUL_CONTEXT)

    def context_for(self, model_size: float) -> int:
        """The largest context window this model can use here."""
        return next((c for c in CONTEXT_CHOICES if self._runs(model_size, c)), CONTEXT_CHOICES[-1])

    @property
    def summary(self) -> str:
        kind = "Apple Silicon" if self.apple_silicon else "CPU only"
        return f"{self.memory_bytes / GIB:.0f} GB memory · {kind}"


def detect_hardware() -> Hardware:
    memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return Hardware(memory, sys.platform == "darwin" and platform.machine() == "arm64")


@dataclass(frozen=True)
class ModelPlan:
    chat: str
    vision: str
    upgrades: tuple[str, ...] = ()  # models worth downloading: better than what is installed


def _recommended(catalog: Sequence[tuple[str, float]], hardware: Hardware) -> str:
    return next((name for name, size in catalog if hardware.fits(size * GB)), catalog[-1][0])


def _best_installed(catalog: Sequence[tuple[str, float]], models: list[dict], hardware: Hardware) -> str | None:
    """Prefer catalog order; models the catalog doesn't know rank after it, largest first."""
    if not models:
        return None
    ranks = {name: rank for rank, (name, _size) in enumerate(catalog)}
    fitting = [model for model in models if hardware.fits(model.get("size") or 0)]
    if not fitting:  # nothing fits comfortably, so the smallest gives the best chance of running
        return min(models, key=lambda model: model.get("size") or 0)["name"]
    return min(fitting, key=lambda m: (ranks.get(m["name"], len(ranks)), -(m.get("size") or 0)))["name"]


def plan_models(installed: Sequence[dict], hardware: Hardware, embedding_model: str) -> ModelPlan:
    """Pick the best installed chat and vision models, and name any better ones to download.

    `installed` holds {"name", "size", "capabilities"} entries. Ollama's cloud models are skipped:
    they report a tiny size because they run remotely, and notes must stay on this computer.
    """
    local = [m for m in installed if "cloud" not in m["name"].casefold()]
    chat_models = [
        m for m in local
        if not is_embedding_model(m["name"]) and "embedding" not in (m.get("capabilities") or ())
    ]
    vision_names = {name for name, _size in VISION_CATALOG}
    vision_models = [
        m for m in chat_models if m["name"] in vision_names or "vision" in (m.get("capabilities") or ())
    ]
    picks, upgrades = {}, []
    for role, catalog, models in (("chat", CHAT_CATALOG, chat_models), ("vision", VISION_CATALOG, vision_models)):
        best = _best_installed(catalog, models, hardware)
        target = _recommended(catalog, hardware)
        ranks = [name for name, _size in catalog]
        if best is None or best not in ranks or ranks.index(target) < ranks.index(best):
            upgrades.append(target)
        picks[role] = best or target
    names = {m["name"] for m in local}
    if embedding_model not in names and f"{embedding_model}:latest" not in names:
        upgrades.append(embedding_model)
    return ModelPlan(picks["chat"], picks["vision"], tuple(dict.fromkeys(upgrades)))


@dataclass(frozen=True)
class ModelChoice:
    name: str
    size_gb: float
    fits: bool


def is_installed(name: str, installed: Sequence[str]) -> bool:
    return name in installed or f"{name}:latest" in installed


def download_choices(role: str, installed: Sequence[str], hardware: Hardware) -> list[ModelChoice]:
    """Catalog models for a role that aren't installed yet.

    Ones that fit this computer come first, best first; the rest follow, closest to fitting first.
    """
    choices = [
        ModelChoice(name, size, hardware.fits(size * GB))
        for name, size in CATALOGS[role] if not is_installed(name, installed)
    ]
    return sorted(choices, key=lambda choice: (not choice.fits, 0 if choice.fits else choice.size_gb))


def known_size(name: str, installed: Sequence[dict]) -> float:
    """Bytes for a model, from what Ollama reports or else the catalog. 0 if unknown."""
    for model in installed:
        if model["name"] == name or model["name"] == f"{name}:latest":
            return float(model.get("size") or 0)
    return next((size * GB for entry, size in (*CHAT_CATALOG, *VISION_CATALOG) if entry == name), 0.0)


def notes_pages_for(context: int) -> int:
    """How many retrieved pages fit beside the chat history and a full answer."""
    return max(1, min(6, (context - 3500) // 1000))


def quiz_pages_for(context: int) -> int:
    """How many lecture pages fit in a quiz prompt beside the generated questions."""
    return max(1, min(12, (context - 2500) // 800))


def is_cloud_model(name: str) -> bool:
    """Ollama's cloud models run on remote servers, so they never qualify: notes stay on this computer."""
    lowered = name.casefold()
    return lowered.endswith("-cloud") or lowered.endswith(":cloud")


def models_by_role(models: Sequence[dict]) -> dict[str, list[str]]:
    """Split installed models into what each slot can actually use.

    `models` holds {"name", "capabilities"} entries (capabilities from Ollama's /api/show; may be empty).
    Chat lists every local model that can generate text, vision only those that can read images,
    embedding only embedding models. Cloud models are left out entirely.
    """
    vision_names = {name for name, _size in VISION_CATALOG}
    roles: dict[str, list[str]] = {"chat": [], "vision": [], "embedding": []}
    for model in sorted(models, key=lambda m: m["name"]):
        name = model["name"]
        if is_cloud_model(name):
            continue
        capabilities = set(model.get("capabilities") or ())
        if "embedding" in capabilities or (not capabilities and is_embedding_model(name)):
            roles["embedding"].append(name)
            continue
        roles["chat"].append(name)
        if "vision" in capabilities or name in vision_names or name.removesuffix(":latest") in vision_names:
            roles["vision"].append(name)
    return roles


def fetch_models_by_role(client, timeout: float = 2.0) -> dict[str, list[str]]:
    """Ask Ollama what's installed and what each model can do. Raises OllamaError when offline."""
    models = [{"name": m["name"]} for m in client.list_models(timeout=timeout)]
    for model in models:
        if not is_cloud_model(model["name"]):
            model["capabilities"] = client.capabilities(model["name"])
    return models_by_role(models)
