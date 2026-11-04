"""A tiny in-process stand-in for the Ollama REST API, used by the tests."""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CHAT_MODELS = {"qwen3:8b": ["completion", "thinking"], "qwen2.5vl:7b": ["completion", "vision"]}
EMBED_MODELS = {"nomic-embed-text:latest": ["embedding"]}
DIMS = 16


def fake_vector(text: str) -> list[float]:
    """Deterministic bag-of-words hashing embedding so related text lands nearby."""
    vector = [0.0] * DIMS
    for word in text.casefold().split():
        word = "".join(ch for ch in word if ch.isalnum())
        if len(word) < 3:
            continue
        index = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMS
        vector[index] += 1.0
    vector[0] += 0.01
    return vector


class FakeOllama:
    def __init__(self, answer: str = "Priority inversion happens when a low-priority task holds a lock [S1]. "
                                      "Priority inheritance fixes it [S1][S9].") -> None:
        self.answer = answer
        self.requests: list[dict] = []
        handler = self._handler()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.host = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "FakeOllama":
        self.thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:  # silence
                pass

            def _send(self, payload: dict | list, status: int = 200) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/api/tags":
                    names = list(CHAT_MODELS) + list(EMBED_MODELS)
                    self._send({"models": [{"name": n, "model": n} for n in names]})
                elif self.path == "/api/ps":
                    self._send({"models": [{"name": "qwen3:8b", "model": "qwen3:8b",
                                            "size": 6_000_000_000, "size_vram": 6_000_000_000}]})
                elif self.path == "/api/version":
                    self._send({"version": "0.0-test"})
                else:
                    self._send({"error": "not found"}, 404)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                fake.requests.append({"path": self.path, **payload})
                model = payload.get("model", "")
                known = {**CHAT_MODELS, **EMBED_MODELS}
                if model and model not in known and f"{model}:latest" not in known:
                    self._send({"error": f"model '{model}' not found"}, 404)
                    return
                if self.path == "/api/show":
                    self._send({"capabilities": known.get(model) or known.get(f"{model}:latest")})
                elif self.path == "/api/embed":
                    self._send({"embeddings": [fake_vector(text) for text in payload["input"]]})
                elif self.path == "/api/chat":
                    self._chat(payload)
                else:
                    self._send({"error": "not found"}, 404)

            def _chat(self, payload: dict) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                fmt = payload.get("format")
                chunks: list[dict] = []
                if payload.get("think"):
                    chunks += [{"message": {"role": "assistant", "thinking": w + " "}} for w in "Check the cited slide".split()]
                if isinstance(fmt, dict):
                    props = fmt.get("properties", {})
                    if "questions" in props:
                        content = json.dumps({"questions": [
                            {"question_type": "multiple_choice", "prompt": "Which fixes priority inversion?",
                             "options": ["Inheritance", "Polling", "Paging", "Spooling"], "correct_answer": "A",
                             "rubric": "Inheritance", "explanation": "Inheritance boosts the holder.",
                             "topics": ["scheduling"], "source_ids": ["S1"]},
                            {"question_type": "short_answer", "prompt": "Explain priority inversion.",
                             "options": [], "correct_answer": "A high task waits on a low one.",
                             "rubric": "Mentions blocking", "explanation": "From slide 1.",
                             "topics": ["scheduling"], "source_ids": ["S1"]},
                        ]})
                    elif "score" in props:
                        content = json.dumps({"score": 0.8, "feedback": "Good.", "source_ids": ["S1"]})
                    else:
                        content = json.dumps({"notes": "handwritten: mutex!"})
                    chunks.append({"message": {"role": "assistant", "content": content}})
                else:
                    chunks += [{"message": {"role": "assistant", "content": w + " "}} for w in fake.answer.split(" ")]
                chunks.append({"done": True, "message": {"role": "assistant", "content": ""},
                               "eval_count": 40, "eval_duration": 1_000_000_000, "prompt_eval_count": 800})
                for chunk in chunks:
                    self.wfile.write((json.dumps(chunk) + "\n").encode())
                    self.wfile.flush()

        return Handler
