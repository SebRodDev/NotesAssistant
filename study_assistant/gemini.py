"""Google Gemini API backend: a drop-in for OllamaClient, so the models run on Google's servers.

Same interface as OllamaClient (list_models, capabilities, embed, chat with streaming), so
StudyModels, quizzes, handwriting, and the screen overlay work unchanged. Uses the REST API directly,
with no SDK to install. The API key comes from GEMINI_API_KEY or the macOS Keychain.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator

from .ollama import AnswerStopped, ChatResult, ChatStats, OllamaError, OllamaUnavailable

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
# Google's rolling aliases always point at the current release, so they don't go stale when models retire.
# A specific model rather than the gemini-flash-latest alias, which is often overloaded on the free tier.
DEFAULT_GEMINI_CHAT = "gemini-3.5-flash"
DEFAULT_GEMINI_EMBEDDING = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768  # plenty for page search, and a quarter the storage of the full 3072
KEY_URL = "https://aistudio.google.com/apikey"

# The same Keychain entry the earlier Gemini version of this app used, so a saved key is picked up.
KEYCHAIN_SERVICE = "Purdue Study Assistant"
KEYCHAIN_ACCOUNT = "gemini-api-key"

# Specialised models that can't do text chat.
_NOT_CHAT = ("tts", "image", "live", "veo", "imagen", "aqa", "audio", "computer-use", "robotics", "embedding",
             "transcribe", "banana")


class GeminiUnavailable(OllamaUnavailable):
    """No usable API key, or Google's API can't be reached."""


class GeminiBusy(OllamaError):
    """This model can't answer right now: overloaded (HTTP 5xx), rate limited, or too slow to respond.
    Another model usually can, so chat() falls back to a sibling when it sees this."""


class GeminiModelMissing(OllamaError):
    """Listed, but this key can't use it (HTTP 404): retired, or not enabled for the key."""


_OVERLOADED = (500, 502, 503, 504)
STICKY_SECONDS = 600.0
# model -> (backup that answered, when to stop preferring it). Module-level because the app makes a
# fresh client per request; this way the next question goes straight to a model that's working.
_STICKY: dict[str, tuple[str, float]] = {}


# ---------------------------------------------------------------------------------------------
# API key storage
# ---------------------------------------------------------------------------------------------

_cached_key: str | None = None


def get_api_key() -> str:
    """GEMINI_API_KEY wins; otherwise the macOS Keychain (read once, then cached). "" when none is saved."""
    global _cached_key
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"].strip()
    if _cached_key is not None:
        return _cached_key
    key = ""
    try:
        import keyring  # optional; the Keychain CLI below covers macOS without it

        key = (keyring.get_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT) or "").strip()
    except Exception:
        if sys.platform == "darwin":
            done = subprocess.run(
                ["/usr/bin/security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", KEYCHAIN_ACCOUNT, "-w"],
                capture_output=True, text=True, check=False,
            )
            if done.returncode == 0:
                key = done.stdout.strip()
    _cached_key = key
    return key


def save_api_key(api_key: str) -> bool:
    """Store the key in the Keychain. Returns False when only this session will remember it."""
    global _cached_key
    api_key = api_key.strip()
    _cached_key = api_key
    try:
        import keyring

        keyring.set_password(KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT, api_key)
        return True
    except Exception:
        pass
    if sys.platform == "darwin":
        done = subprocess.run(
            ["/usr/bin/security", "add-generic-password", "-U", "-s", KEYCHAIN_SERVICE, "-a", KEYCHAIN_ACCOUNT,
             "-w", api_key],
            capture_output=True, text=True, check=False,
        )
        if done.returncode == 0:
            return True
    os.environ["GEMINI_API_KEY"] = api_key
    return False


def mask_key(api_key: str) -> str:
    return f"{api_key[:4]}…{api_key[-4:]}" if len(api_key) > 12 else "saved"


# ---------------------------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------------------------

def _thinks_by_default(model: str) -> bool:
    """Gemini 2.5+ and the -latest aliases think before answering unless told otherwise."""
    name = model.casefold()
    if "pro" in name:  # Pro models can't turn thinking down far; leave them alone
        return False
    versions = re.findall(r"gemini-(\d+(?:\.\d+)?)", name)
    return "latest" in name or bool(versions and float(versions[0]) >= 2.5)


def _fast_thinking(model: str) -> dict:
    """The lowest thinking setting each generation accepts: a token budget on 2.5, a level on 3+."""
    versions = re.findall(r"gemini-(\d+(?:\.\d+)?)", model.casefold())
    if versions and float(versions[0]) < 3:
        return {"thinkingBudget": 0}
    return {"thinkingLevel": "low"}


def _retry_delay(detail: dict) -> float | None:
    """Seconds Google asks us to wait (RetryInfo.retryDelay like "23s"), or None if it didn't say."""
    for info in detail.get("details") or []:
        delay = str(info.get("retryDelay") or "")
        if delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                return None
    return None


def _model_path(model: str) -> str:
    return model if model.startswith("models/") else f"models/{model}"


def is_gemini_chat_model(name: str) -> bool:
    lowered = name.casefold()
    return lowered.startswith(("gemini", "gemma")) and not any(word in lowered for word in _NOT_CHAT)


def _version_key(name: str) -> tuple:
    """Rank model names stable first (previews get retired and have tighter limits), then newest."""
    numbers = tuple(float(n) for n in re.findall(r"\d+(?:\.\d+)?", name)[:1]) or (0.0,)
    preview = any(word in name for word in ("preview", "exp"))
    return (not preview, *numbers, "latest" in name)


def best_model(names: list[str], kind: str) -> str:
    """Pick a sensible default from what the key can use: newest stable Flash, or newest embedding."""
    if kind == "embedding":
        candidates = [n for n in names if "embedding" in n]
        fallback = DEFAULT_GEMINI_EMBEDDING
    else:
        candidates = [n for n in names if is_gemini_chat_model(n) and "flash" in n and "lite" not in n
                      and n.startswith("gemini")]
        fallback = DEFAULT_GEMINI_CHAT
    if not candidates:
        return fallback
    answering = [n for n in candidates if (_HEALTH.get(n) or {}).get("kind") == "ok"]
    if kind != "embedding" and answering:  # the last model check knows what actually works
        stable = [n for n in answering if "preview" not in n] or answering
        return min(stable, key=lambda n: _HEALTH[n]["seconds"])
    if fallback in candidates:
        return fallback
    return max(candidates, key=_version_key)


# ---------------------------------------------------------------------------------------------
# Model health: which models answered the last check (see check_models / tools/check_gemini.py)
# ---------------------------------------------------------------------------------------------

# name -> {"kind": ok | missing | busy | limited | slow | error, "seconds": float, "error": str, "at": epoch}
_HEALTH: dict[str, dict] = {}
HEALTH_FRESH_SECONDS = 6 * 3600  # busy/limited results expire; "missing" (not offered to the key) doesn't


def set_health(results: dict[str, dict]) -> None:
    _HEALTH.clear()
    _HEALTH.update(results)


def health() -> dict[str, dict]:
    return dict(_HEALTH)


def is_unavailable(name: str) -> bool:
    """The key can't use this model at all (it answered 404 in the last check)."""
    return (_HEALTH.get(name) or {}).get("kind") == "missing"


def _fresh_ok(name: str) -> float | None:
    entry = _HEALTH.get(name) or {}
    if entry.get("kind") == "ok" and time.time() - float(entry.get("at") or 0) < HEALTH_FRESH_SECONDS:
        return float(entry.get("seconds") or 0)
    return None


def classify_probe(ok: bool, message: str) -> str:
    if ok:
        return "ok"
    lowered = message.casefold()
    if "doesn't offer" in lowered:
        return "missing"
    if "rate limit" in lowered:
        return "limited"
    if "overloaded" in lowered:
        return "busy"
    if "didn't respond" in lowered:
        return "slow"
    return "error"


def check_models(models: list[str], timeout: float = 20.0, api_key: str | None = None,
                 api_root: str | None = None) -> dict[str, dict]:
    """Ping every model at once with a tiny request; return their health (also kept for fallbacks)."""
    from concurrent.futures import ThreadPoolExecutor

    def probe(name: str) -> tuple[str, bool, float, str]:
        client = GeminiClient(api_key, **({"api_root": api_root} if api_root else {}))
        return name, *client.probe(name, timeout=timeout)

    now = time.time()
    with ThreadPoolExecutor(max_workers=max(1, min(8, len(models)))) as pool:
        results = {
            name: {"kind": classify_probe(ok, message), "seconds": round(seconds, 2),
                   "error": "" if ok else message, "at": now}
            for name, ok, seconds, message in pool.map(probe, models)
        }
    _HEALTH.update(results)
    return results


def recommend(results: dict[str, dict]) -> str | None:
    """A full, stable Flash model that answered quickly; else the fastest model that answered at all.
    Lite and preview models trade quality or stability for speed."""
    working = sorted((n for n, r in results.items() if r["kind"] == "ok"), key=lambda n: results[n]["seconds"])
    solid = [n for n in working if "lite" not in n and "preview" not in n and results[n]["seconds"] < 5]
    return (solid or working or [None])[0]


def health_label(name: str) -> str:
    """Short status for dropdowns: "✓ 1.2s", "✗ busy" (empty when never checked)."""
    entry = _HEALTH.get(name)
    if not entry:
        return ""
    if entry["kind"] == "ok":
        return f"✓ {entry['seconds']:.1f}s"
    return {"missing": "✗ not available", "limited": "✗ rate limited", "busy": "✗ overloaded",
            "slow": "✗ no response"}.get(entry["kind"], "✗ error")


class GeminiClient:
    """Talks to the Gemini REST API with the same methods StudyModels uses on OllamaClient."""

    def __init__(self, api_key: str | None = None, timeout: float = 300.0, api_root: str = API_ROOT) -> None:
        self.api_key = (api_key if api_key is not None else get_api_key()).strip()
        self.timeout = timeout
        self.api_root = api_root.rstrip("/")
        self.host = "gemini"
        self._models: dict[str, dict] = {}
        self.backoff = (2.0, 5.0)  # seconds to wait before each retry when Google is overloaded
        self.stream_timeout = 30.0  # give up on a model that sends nothing for this long, and try another
        self.race_timeout = 8.0  # how long the parallel "who's answering?" check may take
        self.on_status: Callable[[str], None] | None = None  # "busy, retrying…" notices for the UI
        self.stop_event = None  # a threading.Event; set by the Stop button to end waits early
        self._no_fast_thinking: set[str] = set()  # models that rejected a low thinking setting
        self.include_unavailable = False

    # -- transport ---------------------------------------------------------------------------
    def _request(self, method: str, path: str, payload: dict | None = None, timeout: float | None = None,
                 query: str = "", retries: int = 2):
        if not self.api_key:
            raise GeminiUnavailable(
                f"Add your Gemini API key under ⬡ MODELS to use the Gemini API. Get one free at {KEY_URL}"
            )
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.api_root}/{path}{query}", data=data, method=method,
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
        )
        try:
            return urllib.request.urlopen(request, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as error:
            detail: dict = {}
            try:
                detail = json.loads(error.read().decode("utf-8")).get("error", {})
                message, status = detail.get("message", ""), detail.get("status", "")
                retry_after = _retry_delay(detail)
            except Exception:
                message, status, retry_after = "", "", None
            if error.code in (400, 401, 403) and ("API key" in message or status in ("PERMISSION_DENIED",
                                                                                     "UNAUTHENTICATED")):
                raise GeminiUnavailable(f"Gemini rejected the API key: {message or status}. Check it under ⬡ MODELS.") \
                    from error
            model = path.split(":")[0].removeprefix("models/") if ":" in path else "Gemini"
            if error.code == 429:
                # Indexing can afford to wait; a question shouldn't sit silent, so it switches model instead.
                embedding = "Embed" in path
                patience = 70 if embedding else 8
                per_day = "day" in (message + json.dumps(detail)).lower()
                if embedding and retry_after is None and not per_day:
                    retry_after = 60.0  # per-minute quota: a minute always clears it
                if retries > 0 and retry_after is not None and retry_after <= patience:
                    self._wait(retry_after + 0.5, f"Free-tier limit on {model} · continuing in")
                    return self._request(method, path, payload, timeout, query, retries - 1)
                raise GeminiBusy(
                    f"Gemini's rate limit was hit for {model} (the free tier allows a limited number of requests "
                    "per minute and per day). Wait a minute and try again, or pick another model in the header."
                ) from error
            if error.code in _OVERLOADED:
                if retries > 0:  # overload spikes are short: back off a little more each time
                    attempt = 3 - retries
                    self._wait(self.backoff[min(len(self.backoff) - 1, attempt - 1)],
                               f"{model} is busy · retry {attempt} of 2 in")
                    return self._request(method, path, payload, timeout, query, retries - 1)
                raise GeminiBusy(
                    f"Google's servers are overloaded for {model} right now (HTTP {error.code}). This usually "
                    "clears up within a few minutes; try again, or pick another model in the header."
                ) from error
            if error.code == 404:
                raise GeminiModelMissing(
                    f"Gemini doesn't offer the model “{model}” to this key. Pick another under ⬡ MODELS."
                ) from error
            raise OllamaError(f"Gemini returned HTTP {error.code}: {message or error.reason}") from error
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as error:
            reason = getattr(error, "reason", error)
            if isinstance(reason, TimeoutError) and ":" in path:  # reached Google, but the model kept quiet
                model = path.split(":")[0].removeprefix("models/")
                raise GeminiBusy(f"{model} didn't respond within {timeout or self.timeout:.0f}s.") from error
            raise GeminiUnavailable("I can't reach the Gemini API. Check your internet connection.") from error

    def _status(self, text: str) -> None:
        if self.on_status:
            self.on_status(text)

    def _wait(self, seconds: float, reason: str) -> None:
        """Sleep, counting down in the UI, and stop early if the user presses Stop."""
        deadline = time.monotonic() + seconds
        while (left := deadline - time.monotonic()) > 0:
            self._status(f"{reason} {math.ceil(left)}s")
            if self.stop_event is not None and self.stop_event.wait(min(1.0, left)):
                raise AnswerStopped
            if self.stop_event is None:
                time.sleep(min(1.0, left))

    def _json(self, method: str, path: str, payload: dict | None = None, timeout: float | None = None,
              query: str = "", retries: int = 2) -> dict:
        with self._request(method, path, payload, timeout, query, retries=retries) as response:
            return json.loads(response.read().decode("utf-8") or "{}")

    def _stream(self, path: str, payload: dict, retries: int = 2) -> Iterator[dict]:
        with self._request("POST", path, payload, query="?alt=sse", timeout=self.stream_timeout,
                           retries=retries) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if line.startswith("data:"):
                    body = line[5:].strip()
                    if body:
                        yield json.loads(body)

    # -- server info ---------------------------------------------------------------------------
    def version(self, timeout: float = 5.0) -> str:
        self.list_models(timeout)
        return "gemini"

    def list_models(self, timeout: float = 10.0) -> list[dict]:
        """Chat and embedding models. Models the last check found this key can't use are left out
        unless `include_unavailable` is set (the model check itself re-tests them)."""
        models, token = [], ""
        while True:
            query = "?pageSize=1000" + (f"&pageToken={token}" if token else "")
            page = self._json("GET", "models", timeout=timeout, query=query)
            for info in page.get("models", []):
                name = info.get("name", "").removeprefix("models/")
                self._models[name] = info
                capabilities = self.capabilities(name)
                if capabilities and (self.include_unavailable or not is_unavailable(name)):
                    # (skips speech, image-generation and other models that can't chat or embed)
                    models.append({"name": name, "model": name, "size": 0, "capabilities": sorted(capabilities)})
            token = page.get("nextPageToken", "")
            if not token:
                return models

    def running_models(self, timeout: float = 3.0) -> list[dict]:
        return []  # nothing runs on this computer

    def capabilities(self, model: str) -> set[str]:
        info = self._models.get(model.removeprefix("models/"))
        methods = set((info or {}).get("supportedGenerationMethods") or [])
        if info is None:  # not listed yet: infer from the name (aliases like gemini-flash-latest)
            if "embedding" in model:
                return {"embedding"}
            return {"completion", "vision"} if is_gemini_chat_model(model) else set()
        capabilities = set()
        if "embedContent" in methods or "batchEmbedContents" in methods:
            capabilities.add("embedding")
        if "generateContent" in methods and is_gemini_chat_model(model):
            capabilities.update({"completion", "vision"})  # Gemini chat models all accept images
            if info.get("thinking"):
                capabilities.add("thinking")
        return capabilities

    def pull(self, model: str, on_progress: Callable[[str], None] | None = None) -> None:
        raise OllamaError("Gemini models run on Google's servers, so there's nothing to download.")

    # -- inference -------------------------------------------------------------------------
    def embed(self, model: str, inputs: list[str], task: str = "document") -> list[list[float]]:
        task_type = "RETRIEVAL_QUERY" if task == "query" else "RETRIEVAL_DOCUMENT"
        vectors: list[list[float]] = []
        for offset in range(0, len(inputs), 100):  # the API takes at most 100 per batch
            batch = inputs[offset : offset + 100]
            result = self._json("POST", f"{_model_path(model)}:batchEmbedContents", {
                "requests": [{
                    "model": _model_path(model), "content": {"parts": [{"text": text}]},
                    "taskType": task_type, "outputDimensionality": EMBEDDING_DIMENSIONS,
                } for text in batch],
            }, retries=6)  # the free tier embeds ~100 pages a minute; a big library takes a few minutes
            vectors.extend(entry.get("values") or [] for entry in result.get("embeddings") or [])
        if len(vectors) != len(inputs):
            raise OllamaError(f"{model} returned {len(vectors)} embeddings for {len(inputs)} inputs.")
        cleaned = []
        for vector in vectors:
            if not vector or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector):
                raise OllamaError(f"{model} returned an invalid embedding.")
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0  # truncated vectors must be re-normalised
            cleaned.append([float(v) / norm for v in vector])
        return cleaned

    @staticmethod
    def _contents(messages: list[dict]) -> tuple[str, list[dict]]:
        """Ollama-style messages to Gemini's systemInstruction + contents (images as inline PNG data)."""
        system, contents = [], []
        for message in messages:
            if message["role"] == "system":
                system.append(message.get("content", ""))
                continue
            parts = [{"inline_data": {"mime_type": "image/png", "data": image}} for image in message.get("images") or []]
            if message.get("content"):
                parts.append({"text": message["content"]})
            role = "model" if message["role"] == "assistant" else "user"
            if contents and contents[-1]["role"] == role:  # Gemini wants turns to alternate
                contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": role, "parts": parts})
        return "\n\n".join(system), contents

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.3,
        num_ctx: int = 8192,
        response_format: dict | str | None = None,
        think: bool | None = None,
        on_token: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> ChatResult:
        system, contents = self._contents(messages)
        config: dict = {"temperature": float(temperature)}
        if response_format is not None:
            config["responseMimeType"] = "application/json"
            if isinstance(response_format, dict):
                config["responseJsonSchema"] = response_format
        if think and "thinking" in self.capabilities(model):  # other models reject thinkingConfig
            config["thinkingConfig"] = {"includeThoughts": True}
        elif not think and model not in self._no_fast_thinking and _thinks_by_default(model):
            # Flash models think silently before answering; keep it minimal unless reasoning was asked for.
            config["thinkingConfig"] = _fast_thinking(model)
        payload: dict = {"contents": contents, "generationConfig": config}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        busy: GeminiBusy | None = None
        tried: list[str] = []
        self._raced: list[str] = []
        for candidate in self._candidates(model):
            tried.append(candidate)
            candidate_config = dict(config)
            if candidate != model:
                self._status(f"{model} can't answer right now · trying {candidate}")
                candidate_config.pop("thinkingConfig", None)
                if think and "thinking" in self.capabilities(candidate):
                    candidate_config["thinkingConfig"] = {"includeThoughts": True}
                elif not think and candidate not in self._no_fast_thinking and _thinks_by_default(candidate):
                    candidate_config["thinkingConfig"] = _fast_thinking(candidate)
            candidate_payload = {**payload, "generationConfig": candidate_config}
            try:
                # Backups get no waiting retries: if one is busy too, move straight on to the next.
                result = self._run_with_schema_fallback(candidate, candidate_payload, candidate_config,
                                                        on_token, on_thinking, retries=2 if len(tried) == 1 else 0)
            except (GeminiBusy, GeminiModelMissing) as error:  # let another model answer
                busy = busy or (error if isinstance(error, GeminiBusy) else GeminiBusy(str(error)))
                if _STICKY.get(model, ("",))[0] == candidate:
                    _STICKY.pop(model, None)
                continue
            if candidate == model:
                _STICKY.pop(model, None)
            else:
                _STICKY[model] = (candidate, time.monotonic() + STICKY_SECONDS)
            return result
        if len(tried) == 1 and self._raced:
            raise GeminiBusy(
                f"{model} can't answer right now, and none of the {len(self._raced)} other Flash models answered a "
                "quick check either. Google is likely overloaded or your free-tier quota is used up; try again in a "
                "few minutes, or run `python tools/check_gemini.py` to see what's working."
            ) from busy
        if len(tried) > 1:
            raise GeminiBusy(
                f"Google's servers are overloaded right now: {', '.join(tried[:-1])} and {tried[-1]} all failed. "
                "This usually clears up within a few minutes. Try again shortly, or pick another model in the header."
            ) from busy
        raise busy

    def _candidates(self, model: str) -> Iterator[str]:
        """A backup that recently answered for this model, then the model, then (only if needed) whichever
        other models answer a quick parallel check first."""
        seen = []
        sticky, until = _STICKY.get(model, ("", 0.0))
        if sticky and time.monotonic() < until:
            seen.append(sticky)
            yield sticky
        if model not in seen:
            seen.append(model)
            yield model
        self._status(f"{model} can't answer right now · checking which models are answering")
        contenders = [n for n in self._fallbacks(model) if n not in seen]
        self._raced = contenders
        for name in self._race(contenders):
            seen.append(name)
            yield name

    def _race(self, models: list[str]) -> list[str]:
        """Ping several models at once; return the ones that answered, fastest first.

        Stops waiting once two have answered or `race_timeout` passes, so one hung model can't stall it."""
        if not models:
            return []
        from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

        answered: list[str] = []
        deadline = time.monotonic() + self.race_timeout
        pool = ThreadPoolExecutor(max_workers=len(models))
        try:
            pending = {pool.submit(self._clone().probe, name, self.race_timeout): name for name in models}
            while pending and len(answered) < 2 and (left := deadline - time.monotonic()) > 0:
                if self.stop_event is not None and self.stop_event.is_set():
                    raise AnswerStopped
                done, _rest = wait(pending, timeout=min(left, 0.5), return_when=FIRST_COMPLETED)
                for future in done:
                    name = pending.pop(future)
                    if future.result()[0]:
                        answered.append(name)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)  # a hung probe ends on its own timeout
        return sorted(answered, key=lambda name: "lite" in name)  # full Flash answers better than Lite

    def _clone(self) -> "GeminiClient":
        """A separate client for a parallel probe (shares the key and the model list)."""
        clone = GeminiClient(self.api_key, api_root=self.api_root)
        clone._models = self._models
        return clone

    def _fallbacks(self, model: str, limit: int = 6) -> list[str]:
        """Other Flash models this key can use, to race when `model` can't answer.

        Picks across generations and sizes (e.g. 3.5 Flash, a Lite model, 3 Flash) rather than the top
        few siblings, since closely related models tend to be overloaded at the same time."""
        if not self._models:
            try:
                self.list_models()
            except OllamaError:
                return []
        names = [n for n in self._models if n != model and is_gemini_chat_model(n) and "flash" in n
                 and n.startswith("gemini") and not is_unavailable(n) and "generateContent" in
                 (self._models[n].get("supportedGenerationMethods") or [])]
        # Models that answered the last check come first (fastest first), then the rest by version.
        proven = sorted((n for n in names if _fresh_ok(n) is not None), key=_fresh_ok)
        rest = sorted((n for n in names if _fresh_ok(n) is None), key=_version_key, reverse=True)
        picked, families = [], set()
        for name in proven + rest:
            family = (_version_key(name)[1], "lite" in name)
            if family not in families:
                families.add(family)
                picked.append(name)
        return picked[:limit]

    def probe(self, model: str, timeout: float = 20.0) -> tuple[bool, float, str]:
        """Send one tiny request with no retries or fallbacks: (answered, seconds, error or reply)."""
        payload = {"contents": [{"role": "user", "parts": [{"text": "Reply with the single word OK."}]}],
                   "generationConfig": {"temperature": 0.0}}
        if _thinks_by_default(model):
            payload["generationConfig"]["thinkingConfig"] = _fast_thinking(model)
        started = time.monotonic()
        saved = self.stream_timeout
        self.stream_timeout = timeout
        try:
            try:
                result = self._run(model, payload, None, None, retries=0)
            except OllamaError as error:
                if "thinkingConfig" not in payload["generationConfig"] or "HTTP 400" not in str(error):
                    raise
                del payload["generationConfig"]["thinkingConfig"]  # this model doesn't take the setting
                self._no_fast_thinking.add(model)
                result = self._run(model, payload, None, None, retries=0)
            return True, time.monotonic() - started, result.content.strip()[:40]
        except (OllamaError, AnswerStopped) as error:
            return False, time.monotonic() - started, str(error)
        finally:
            self.stream_timeout = saved

    def _run_with_schema_fallback(self, model, payload, config, on_token, on_thinking,
                                  retries: int = 2) -> ChatResult:
        for _attempt in range(3):
            try:
                return self._run(model, payload, on_token, on_thinking, retries)
            except GeminiBusy:
                raise
            except OllamaError as error:
                if "HTTP 400" not in str(error):
                    raise
                thinking = config.get("thinkingConfig") or {}
                if thinking and "includeThoughts" not in thinking:
                    # This model doesn't take the low-thinking setting: remember, and ask without it.
                    self._no_fast_thinking.add(model)
                    del config["thinkingConfig"]
                elif "responseJsonSchema" in config:
                    # Older models don't take a JSON schema; the prompt already describes the shape.
                    del config["responseJsonSchema"]
                else:
                    raise
        raise OllamaError(f"Gemini couldn't answer with {model}.")

    @staticmethod
    def _take_chunk(chunk: dict, content: list[str], thinking: list[str], on_token, on_thinking) -> None:
        """Collect one streamed chunk's text, passing it on as it arrives."""
        for candidate in chunk.get("candidates") or []:
            for part in (candidate.get("content") or {}).get("parts") or []:
                text = part.get("text")
                if not text:
                    continue
                if part.get("thought"):
                    thinking.append(text)
                    if on_thinking:
                        on_thinking(text)
                else:
                    content.append(text)
                    if on_token:
                        on_token(text)
            if candidate.get("finishReason") in ("SAFETY", "RECITATION", "PROHIBITED_CONTENT") and not content:
                raise OllamaError(f"Gemini declined to answer ({candidate['finishReason'].lower()}).")

    def _run(self, model, payload, on_token, on_thinking, retries: int = 2) -> ChatResult:
        content, thinking = [], []
        usage: dict = {}
        block_reason = ""
        started = time.monotonic()  # speed includes the network wait: what the answer feels like
        last = None
        try:
            for chunk in self._stream(f"{_model_path(model)}:streamGenerateContent", payload, retries):
                self._take_chunk(chunk, content, thinking, on_token, on_thinking)
                usage = chunk.get("usageMetadata") or usage
                block_reason = (chunk.get("promptFeedback") or {}).get("blockReason", block_reason)
                if content:
                    last = time.monotonic()
        except (TimeoutError, OSError) as error:  # silence or a dropped connection mid-stream
            if content:
                raise OllamaError("The connection to Gemini dropped partway through the answer. Try again.") \
                    from error
            raise GeminiBusy(f"{model} didn't respond within {self.stream_timeout:.0f}s.") from error
        if not content and not thinking:
            reason = f" ({block_reason.lower()})" if block_reason else ""
            raise OllamaError(f"Gemini returned an empty answer{reason}.")
        output_tokens = int(usage.get("candidatesTokenCount") or 0)
        duration = int((last - started) * 1e9) if last and last > started else 0
        stats = ChatStats(
            model=model, eval_count=output_tokens, eval_duration_ns=duration,
            prompt_eval_count=int(usage.get("promptTokenCount") or 0),
            cached_tokens=int(usage.get("cachedContentTokenCount") or 0),
        )
        return ChatResult(content="".join(content), thinking="".join(thinking), stats=stats)
