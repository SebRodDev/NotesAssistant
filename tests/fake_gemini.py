"""A tiny in-process stand-in for the Gemini REST API (v1beta), used by the tests."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from fake_ollama import fake_vector

MODELS = [
    {"name": "models/gemini-flash-latest", "supportedGenerationMethods": ["generateContent"], "thinking": True},
    {"name": "models/gemini-2.0-flash", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/gemini-embedding-001", "supportedGenerationMethods": ["embedContent", "batchEmbedContents"]},
    {"name": "models/gemini-2.5-flash-preview-tts", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/imagen-4.0-generate-001", "supportedGenerationMethods": ["predict"]},
]
KEY = "test-key-1234567890"


class FakeGemini:
    def __init__(self, answer: str = "Priority inversion is when a low-priority task holds a lock [S1]. "
                                      "Inheritance fixes it [S1][S7].", models: list[dict] | None = None,
                 token_delay: float = 0.0) -> None:
        self.answer = answer
        self.models = models if models is not None else MODELS
        self.token_delay = token_delay
        self.requests: list[dict] = []
        self.fail_next: list[tuple[int, dict]] = []  # queued (status, error body) responses
        self.busy_models: set[str] = set()  # models that always answer 503 "high demand"
        self.limited_models: set[str] = set()  # models whose quota is used up (429, long retry)
        self.silent_models: dict[str, float] = {}  # models that wait this many seconds before answering
        self.reject_thinking: set[str] = set()  # models that 400 on thinkingBudget / thinkingLevel
        self.last_system = ""
        self.missing_models: set[str] = set()
        # ("max" | "silent" | "stop" | "broken", words): cut the next answer off. "stop" claims it finished;
        # "broken" ends the stream mid-message.
        self.truncate: tuple[str, int] | None = None
        self.busy_on_continue = False  # answer 503 when asked to carry on a cut-off answer
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.api_root = f"http://127.0.0.1:{self.server.server_address[1]}/v1beta"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "FakeGemini":
        self.thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:
                pass

            def _send(self, payload: dict, status: int = 200) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _authorised(self) -> bool:
                if self.headers.get("x-goog-api-key") != KEY:
                    self._send({"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                          "message": "API key not valid. Please pass a valid API key."}}, 400)
                    return False
                if fake.fail_next:
                    status, body = fake.fail_next.pop(0)
                    self._send(body, status)
                    return False
                return True

            def do_GET(self) -> None:  # noqa: N802
                url = urlparse(self.path)
                fake.requests.append({"method": "GET", "path": url.path, "query": parse_qs(url.query)})
                if not self._authorised():
                    return
                if url.path.endswith("/models"):
                    self._send({"models": fake.models})
                else:
                    self._send({"error": {"code": 404, "message": "not found"}}, 404)

            def do_POST(self) -> None:  # noqa: N802
                url = urlparse(self.path)
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                fake.requests.append({"method": "POST", "path": url.path, "query": parse_qs(url.query), **payload})
                if not self._authorised():
                    return
                model = url.path.split("/models/")[-1].split(":")[0]
                known = {m["name"].removeprefix("models/") for m in fake.models}
                if model not in known or model in fake.missing_models:
                    self._send({"error": {"code": 404, "status": "NOT_FOUND", "message": f"models/{model} is not found"}},
                               404)
                    return
                if model in fake.limited_models:
                    self._send({"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded",
                                          "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                                                       "retryDelay": "45s"}]}}, 429)
                    return
                thinking = (payload.get("generationConfig") or {}).get("thinkingConfig") or {}
                if model in fake.reject_thinking and ("thinkingLevel" in thinking or "thinkingBudget" in thinking):
                    self._send({"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                          "message": "Thinking level is not supported for this model."}}, 400)
                    return
                if model in fake.silent_models:
                    time.sleep(fake.silent_models[model])
                wants_more = "cut off" in json.dumps((payload.get("contents") or [{}])[-1])
                if model in fake.busy_models or (fake.busy_on_continue and wants_more):
                    self._send({"error": {"code": 503, "status": "UNAVAILABLE",
                                          "message": "This model is currently experiencing high demand."}}, 503)
                    return
                if url.path.endswith(":batchEmbedContents"):
                    self._send({"embeddings": [
                        {"values": fake_vector(r["content"]["parts"][0]["text"])} for r in payload["requests"]
                    ]})
                elif url.path.endswith(":streamGenerateContent"):
                    self._stream(payload)
                else:
                    self._send({"error": {"code": 404, "message": "not found"}}, 404)

            def _stream(self, payload: dict) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                config = payload.get("generationConfig") or {}
                chunks: list[dict] = []
                if (config.get("thinkingConfig") or {}).get("includeThoughts"):
                    chunks.append({"candidates": [{"content": {"parts": [{"text": "Check slide 1.", "thought": True}]}}]})
                schema = config.get("responseJsonSchema") or {}
                if config.get("responseMimeType") == "application/json":
                    props = schema.get("properties", {})
                    if "score" in props:
                        text = json.dumps({"score": 0.9, "feedback": "Good.", "source_ids": ["S1"]})
                    elif "notes" in props:
                        text = json.dumps({"notes": "handwritten: mutex!"})
                    else:
                        text = json.dumps({"questions": [{
                            "question_type": "multiple_choice", "prompt": "Which fixes priority inversion?",
                            "options": ["Inheritance", "Polling", "Paging", "Spooling"], "correct_answer": "A",
                            "rubric": "Inheritance", "explanation": "Boosts the holder.", "topics": ["scheduling"],
                            "source_ids": ["S1"]}]})
                    chunks.append({"candidates": [{"content": {"parts": [{"text": text}], "role": "model"}}]})
                else:
                    words = fake.answer.split(" ")
                    finish = "STOP"
                    last_turn = (payload.get("contents") or [{}])[-1]
                    asked_to_continue = "cut off" in json.dumps(last_turn)
                    if fake.truncate and not asked_to_continue:
                        how, keep = fake.truncate
                        words, finish = words[:keep], {"max": "MAX_TOKENS", "stop": "STOP"}.get(how, "")
                        fake.cut_at = keep
                    elif asked_to_continue:
                        fake.truncate = None
                        words = words[fake.cut_at - 3:]  # repeats a few words, as models sometimes do
                    chunks += [{"candidates": [{"content": {"parts": [{"text": w + " "}], "role": "model"}}]}
                               for w in words]
                # Like Gemini's implicit cache: a request that starts like the previous one reuses that part.
                system = json.dumps(payload.get("systemInstruction"))
                cached = 800 if system == fake.last_system and len(system) > 200 else 0
                fake.last_system = system
                chunks[-1]["usageMetadata"] = {"promptTokenCount": 900, "candidatesTokenCount": 30,
                                               **({"cachedContentTokenCount": cached} if cached else {})}
                chunks[-1]["candidates"] = chunks[-1].get("candidates", [{}])
                if config.get("responseMimeType") == "application/json":
                    finish = "STOP"
                if finish:
                    chunks[-1]["candidates"][0]["finishReason"] = finish
                broken = finish == "" and fake.truncate and fake.truncate[0] == "broken"
                for chunk in chunks:
                    try:
                        self.wfile.write(f"data: {json.dumps(chunk)}\r\n\r\n".encode())
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    if fake.token_delay:
                        time.sleep(fake.token_delay)
                if broken:  # the connection dies halfway through a message
                    self.wfile.write(b'data: {"candidates": [{"content": {"par')
                    self.wfile.flush()
                    self.close_connection = True

        return Handler
