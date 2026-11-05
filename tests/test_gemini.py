import math
import os
import unittest
from unittest.mock import patch

from fake_gemini import KEY, FakeGemini
from study_assistant import gemini
from study_assistant.gemini import GeminiClient, best_model
from study_assistant.hardware import fetch_models_by_role
from study_assistant.ollama import OllamaError, OllamaUnavailable, StudyModels


def source(source_id: str = "S1") -> dict:
    return {"source_id": source_id, "page_id": 1, "class_name": "CS 354", "lecture_name": "Scheduling",
            "file_path": "/tmp/scheduling.pdf", "page_number": 1,
            "text": "Priority inversion blocks a high-priority task.", "visual_text": ""}


class GeminiClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeGemini().__enter__()
        self.addCleanup(self.fake.__exit__)
        self.client = GeminiClient(KEY, api_root=self.fake.api_root)
        self.client.backoff = (0.0, 0.0)  # no real waiting in tests
        gemini._STICKY.clear()  # backups remembered by an earlier test shouldn't leak into this one
        self.addCleanup(gemini._STICKY.clear)
        gemini.set_health({})
        self.addCleanup(gemini.set_health, {})

    def test_lists_only_chat_and_embedding_models_with_capabilities(self) -> None:
        roles = fetch_models_by_role(self.client)
        self.assertEqual(roles["chat"], ["gemini-2.0-flash", "gemini-flash-latest"])  # no TTS, no Imagen
        self.assertEqual(roles["vision"], roles["chat"])  # Gemini chat models all read images
        self.assertEqual(roles["embedding"], ["gemini-embedding-001"])
        self.assertIn("thinking", self.client.capabilities("gemini-flash-latest"))
        self.assertNotIn("thinking", self.client.capabilities("gemini-2.0-flash"))

    def test_streams_answers_with_system_prompt_and_alternating_turns(self) -> None:
        tokens = []
        models = StudyModels(self.client, chat_model="gemini-flash-latest", embedding_model="gemini-embedding-001")
        result = models.stream_answer("What is priority inversion?", [source()], notes_mode="strict",
                                      persona="Be brief.", history=[("user", "hi"), ("assistant", "hello")],
                                      on_token=tokens.append)
        self.assertGreater(len(tokens), 5)
        self.assertIn("[1]", result["answer"])
        self.assertNotIn("S7", result["answer"])  # invented citation dropped
        request = self.fake.requests[-1]
        self.assertEqual(request["query"]["alt"], ["sse"])
        self.assertIn("Be brief.", request["systemInstruction"]["parts"][0]["text"])
        self.assertEqual([turn["role"] for turn in request["contents"]], ["user", "model", "user"])
        self.assertGreater(result["tokens_per_second"], 0)

    def test_reasoning_is_only_requested_from_thinking_models(self) -> None:
        self.client.list_models()
        self.client.chat("gemini-2.0-flash", [{"role": "user", "content": "hi"}], think=True)
        self.assertNotIn("thinkingConfig", self.fake.requests[-1]["generationConfig"])
        thoughts = []
        self.client.chat("gemini-flash-latest", [{"role": "user", "content": "hi"}], think=True,
                         on_thinking=thoughts.append)
        self.assertTrue(thoughts)

    def test_screenshots_are_sent_as_inline_images(self) -> None:
        models = StudyModels(self.client, chat_model="gemini-flash-latest", vision_model="gemini-flash-latest")
        models.stream_screen_answer("What's this?", b"\x89PNG fake")
        parts = self.fake.requests[-1]["contents"][-1]["parts"]
        self.assertEqual(parts[0]["inline_data"]["mime_type"], "image/png")
        self.assertEqual(parts[-1]["text"], "What's this?")

    def test_embeddings_use_task_types_batches_and_unit_length(self) -> None:
        vectors = self.client.embed("gemini-embedding-001", [f"page {i} scheduling" for i in range(130)])
        self.assertEqual(len(vectors), 130)
        batches = [r for r in self.fake.requests if r["path"].endswith(":batchEmbedContents")]
        self.assertEqual([len(b["requests"]) for b in batches], [100, 30])
        self.assertEqual(batches[0]["requests"][0]["taskType"], "RETRIEVAL_DOCUMENT")
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in vectors[0])), 1.0, places=6)
        self.client.embed("gemini-embedding-001", ["query"], task="query")
        self.assertEqual(self.fake.requests[-1]["requests"][0]["taskType"], "RETRIEVAL_QUERY")

    def test_structured_quiz_and_grading_use_json_schema(self) -> None:
        models = StudyModels(self.client, chat_model="gemini-flash-latest")
        questions = models.generate_quiz(1, "multiple_choice", [source()])
        self.assertEqual(questions[0]["correct_answer"], "A")
        config = self.fake.requests[-1]["generationConfig"]
        self.assertEqual(config["responseMimeType"], "application/json")
        self.assertIn("questions", config["responseJsonSchema"]["properties"])
        grade = models.grade_short_answer({**questions[0], "prompt": "q", "correct_answer": "a", "rubric": "r"},
                                          "answer", [source()])
        self.assertTrue(grade["is_correct"])

    def test_short_rate_limits_are_retried_and_long_ones_explained(self) -> None:
        retry = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded",
                           "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "0s"}]}}
        self.fake.fail_next.append((429, retry))
        self.assertEqual(len(self.client.embed("gemini-embedding-001", ["x"])), 1)  # waited, then succeeded
        daily = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded per day"}}
        self.fake.fail_next.append((429, daily))
        with self.assertRaisesRegex(OllamaError, "rate limit"):
            self.client.embed("gemini-embedding-001", ["x"])

    def test_bad_or_missing_key_gives_a_clear_message(self) -> None:
        with self.assertRaisesRegex(OllamaUnavailable, "rejected the API key"):
            GeminiClient("wrong", api_root=self.fake.api_root).list_models()
        with patch.dict(os.environ, {}, clear=True), patch.object(gemini, "_cached_key", ""):
            with self.assertRaisesRegex(OllamaUnavailable, "Add your Gemini API key"):
                GeminiClient(api_root=self.fake.api_root).list_models()

    def test_unavailable_model_falls_back_and_says_so(self) -> None:
        notices = []
        self.client.on_status = notices.append
        result = self.client.chat("gemini-9-ultra", [{"role": "user", "content": "hi"}])
        self.assertNotEqual(result.stats.model, "gemini-9-ultra")
        self.assertIn("gemini-9-ultra can't answer right now", notices[0])
        with self.assertRaisesRegex(OllamaError, "doesn't offer the model"):
            self.client._run("gemini-9-ultra", {"contents": []}, None, None)

    def test_default_model_choice_prefers_stable_flash(self) -> None:
        names = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash-lite"]
        self.assertEqual(best_model(names, "chat"), "gemini-2.5-flash")
        self.assertEqual(best_model(names + ["gemini-3.5-flash"], "chat"), "gemini-3.5-flash")  # the default
        self.assertEqual(best_model(["gemini-embedding-001"], "embedding"), "gemini-embedding-001")


    def test_brief_overload_is_retried(self) -> None:
        busy = {"error": {"code": 503, "status": "UNAVAILABLE", "message": "high demand"}}
        self.fake.fail_next += [(503, busy), (503, busy)]
        result = self.client.chat("gemini-flash-latest", [{"role": "user", "content": "hi"}])
        self.assertTrue(result.content)
        self.assertEqual(result.stats.model, "gemini-flash-latest")

    def test_persistent_overload_falls_back_to_another_flash_model(self) -> None:
        self.fake.busy_models.add("gemini-flash-latest")
        tokens = []
        result = self.client.chat("gemini-flash-latest", [{"role": "user", "content": "hi"}], think=True,
                                  on_token=tokens.append)
        self.assertEqual(result.stats.model, "gemini-2.0-flash")
        self.assertTrue(tokens)
        last = self.fake.requests[-1]
        self.assertIn("gemini-2.0-flash", last["path"])
        self.assertNotIn("thinkingConfig", last["generationConfig"])  # the fallback can't think

    def test_overload_everywhere_gives_a_clear_message(self) -> None:
        self.fake.busy_models.update({"gemini-flash-latest", "gemini-2.0-flash"})
        with self.assertRaisesRegex(OllamaError, "none of the 1 other Flash models answered"):
            self.client.chat("gemini-flash-latest", [{"role": "user", "content": "hi"}])


    def test_thinking_is_kept_minimal_unless_reasoning_is_asked_for(self) -> None:
        self.client.list_models()
        self.client.chat("gemini-flash-latest", [{"role": "user", "content": "hi"}])
        self.assertEqual(self.fake.requests[-1]["generationConfig"]["thinkingConfig"], {"thinkingLevel": "low"})
        self.client.chat("gemini-2.0-flash", [{"role": "user", "content": "hi"}])
        self.assertNotIn("thinkingConfig", self.fake.requests[-1]["generationConfig"])  # 2.0 doesn't think
        self.assertEqual(gemini._fast_thinking("gemini-2.5-flash"), {"thinkingBudget": 0})
        self.assertFalse(gemini._thinks_by_default("gemini-2.5-pro"))

    def test_models_that_reject_the_thinking_setting_are_asked_without_it_and_remembered(self) -> None:
        self.fake.reject_thinking.add("gemini-flash-latest")
        result = self.client.chat("gemini-flash-latest", [{"role": "user", "content": "hi"}])
        self.assertTrue(result.content)
        self.assertNotIn("thinkingConfig", self.fake.requests[-1]["generationConfig"])
        before = len(self.fake.requests)
        self.client.chat("gemini-flash-latest", [{"role": "user", "content": "again"}])
        self.assertEqual(len(self.fake.requests), before + 1)  # no wasted 400 the second time

    def test_long_rate_limit_switches_model_instead_of_waiting(self) -> None:
        self.fake.limited_models.add("gemini-flash-latest")
        notices = []
        self.client.on_status = notices.append
        result = self.client.chat("gemini-flash-latest", [{"role": "user", "content": "hi"}])
        self.assertEqual(result.stats.model, "gemini-2.0-flash")
        self.assertTrue(any("trying gemini-2.0-flash" in n for n in notices))

    def test_silent_model_times_out_and_another_answers(self) -> None:
        self.client.stream_timeout = 0.5
        self.fake.silent_models["gemini-flash-latest"] = 2.0
        started = __import__("time").monotonic()
        result = self.client.chat("gemini-flash-latest", [{"role": "user", "content": "hi"}])
        self.assertEqual(result.stats.model, "gemini-2.0-flash")
        self.assertLess(__import__("time").monotonic() - started, 1.8)

    def test_stop_interrupts_a_retry_wait(self) -> None:
        import threading
        import time as clock

        from study_assistant.ollama import AnswerStopped

        self.client.backoff = (10.0, 10.0)
        self.client.stop_event = threading.Event()
        notices = []
        self.client.on_status = notices.append
        busy = {"error": {"code": 503, "status": "UNAVAILABLE", "message": "high demand"}}
        self.fake.fail_next.append((503, busy))
        threading.Timer(0.3, self.client.stop_event.set).start()
        started = clock.monotonic()
        with self.assertRaises(AnswerStopped):
            self.client.chat("gemini-flash-latest", [{"role": "user", "content": "hi"}])
        self.assertLess(clock.monotonic() - started, 2)
        self.assertIn("gemini-flash-latest is busy · retry 1 of 2 in 10s", notices)


    def test_a_backup_that_answered_is_used_first_for_ten_minutes(self) -> None:
        self.fake.busy_models.add("gemini-flash-latest")
        self.client.chat("gemini-flash-latest", [{"role": "user", "content": "hi"}])
        before = len([r for r in self.fake.requests if "flash-latest:stream" in r["path"]])
        fresh = GeminiClient(KEY, api_root=self.fake.api_root)  # the app makes a new client per question
        result = fresh.chat("gemini-flash-latest", [{"role": "user", "content": "again"}])
        self.assertEqual(result.stats.model, "gemini-2.0-flash")
        after = len([r for r in self.fake.requests if "flash-latest:stream" in r["path"]])
        self.assertEqual(after, before)  # didn't knock on the overloaded model first
        self.fake.busy_models.clear()
        gemini._STICKY["gemini-flash-latest"] = ("gemini-2.0-flash", 0.0)  # ten minutes later
        self.assertEqual(fresh.chat("gemini-flash-latest", [{"role": "user", "content": "x"}]).stats.model,
                         "gemini-flash-latest")
        self.assertNotIn("gemini-flash-latest", gemini._STICKY)

    def test_backups_are_spread_across_generations_and_sizes(self) -> None:
        names = ["gemini-2.5-flash", "gemini-2.5-flash-001", "gemini-2.5-flash-lite", "gemini-2.0-flash",
                 "gemini-2.0-flash-001", "gemini-2.0-flash-lite", "gemini-2.5-pro"]
        self.client._models = {n: {"supportedGenerationMethods": ["generateContent"]} for n in names}
        self.assertEqual(self.client._fallbacks("gemini-flash-latest"),
                         ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.0-flash-lite"])

    def test_probe_reports_each_model_without_retrying(self) -> None:
        self.fake.busy_models.add("gemini-2.0-flash")
        ok, seconds, reply = self.client.probe("gemini-flash-latest")
        self.assertTrue(ok)
        busy, _seconds, error = self.client.probe("gemini-2.0-flash")
        self.assertFalse(busy)
        self.assertIn("overloaded", error)
        self.assertEqual(len([r for r in self.fake.requests if "2.0-flash:stream" in r["path"]]), 1)


    def test_backups_are_found_by_racing_not_guessing(self) -> None:
        """Mirrors a real run: the top-versioned backups hang or are overloaded, lower ones answer."""
        models = [
            {"name": "models/gemini-flash-latest", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-3.7-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-3.5-flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-3.5-flash-lite", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/gemini-3.5-transcribe", "supportedGenerationMethods": ["generateContent"]},
        ]
        self.fake.models = models
        self.fake.busy_models.update({"gemini-flash-latest", "gemini-3.7-flash"})
        self.fake.silent_models["gemini-3.8-flash"] = 3.0  # hangs
        self.client.race_timeout = 2.0
        notices = []
        self.client.on_status = notices.append
        started = __import__("time").monotonic()
        result = self.client.chat("gemini-flash-latest", [{"role": "user", "content": "hi"}])
        self.assertEqual(result.stats.model, "gemini-3.5-flash")  # full Flash beats Lite when both answer
        self.assertLess(__import__("time").monotonic() - started, 2.5)  # didn't wait out the hung model
        self.assertIn("checking which models are answering", notices[0])
        self.assertNotIn("gemini-3.5-transcribe", self.client._raced)  # not a chat model

    def test_a_model_this_key_cannot_use_is_skipped_not_fatal(self) -> None:
        self.fake.models = [*self.fake.models,
                            {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]}]
        self.fake.missing_models.add("gemini-2.5-flash")  # listed, but this key can't use it
        self.fake.busy_models.add("gemini-flash-latest")
        result = self.client.chat("gemini-flash-latest", [{"role": "user", "content": "hi"}])
        self.assertEqual(result.stats.model, "gemini-2.0-flash")  # 2.5 was skipped, not fatal


    def test_model_check_classifies_results_and_recommends_a_full_flash_model(self) -> None:
        self.fake.models = [*self.fake.models,
                            {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
                            {"name": "models/gemini-3.1-flash-lite", "supportedGenerationMethods": ["generateContent"]}]
        self.fake.missing_models.add("gemini-2.5-flash")
        self.fake.busy_models.add("gemini-flash-latest")
        results = gemini.check_models(["gemini-flash-latest", "gemini-2.0-flash", "gemini-2.5-flash",
                                       "gemini-3.1-flash-lite"], api_key=KEY, api_root=self.fake.api_root)
        kinds = {name: r["kind"] for name, r in results.items()}
        self.assertEqual(kinds, {"gemini-flash-latest": "busy", "gemini-2.0-flash": "ok",
                                 "gemini-2.5-flash": "missing", "gemini-3.1-flash-lite": "ok"})
        self.assertEqual(gemini.recommend(results), "gemini-2.0-flash")  # full Flash beats a faster Lite
        self.assertEqual(gemini.health_label("gemini-2.5-flash"), "✗ not available")
        self.assertTrue(gemini.health_label("gemini-2.0-flash").startswith("✓"))

    def test_models_the_key_cannot_use_are_hidden_until_rechecked(self) -> None:
        gemini.set_health({"gemini-2.0-flash": {"kind": "missing", "seconds": 0.2, "error": "", "at": 0}})
        self.assertNotIn("gemini-2.0-flash", fetch_models_by_role(self.client)["chat"])
        self.client.include_unavailable = True
        self.assertIn("gemini-2.0-flash", fetch_models_by_role(self.client)["chat"])
        self.assertNotIn("gemini-2.0-flash", self.client._fallbacks("gemini-flash-latest"))

    def test_backups_that_answered_the_check_are_tried_first(self) -> None:
        import time as clock

        names = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite"]
        self.client._models = {n: {"supportedGenerationMethods": ["generateContent"]} for n in names}
        now = clock.time()
        gemini.set_health({"gemini-3.1-flash-lite": {"kind": "ok", "seconds": 0.8, "error": "", "at": now},
                           "gemini-3.5-flash": {"kind": "ok", "seconds": 4.0, "error": "", "at": now},
                           "gemini-3.8-flash": {"kind": "busy", "seconds": 3.2, "error": "", "at": now}})
        self.assertEqual(self.client._fallbacks("gemini-flash-latest")[:2], ["gemini-3.1-flash-lite", "gemini-3.5-flash"])
        stale = now - gemini.HEALTH_FRESH_SECONDS - 1
        gemini.set_health({"gemini-3.1-flash-lite": {"kind": "ok", "seconds": 0.8, "error": "", "at": stale}})
        self.assertEqual(self.client._fallbacks("gemini-flash-latest")[0], "gemini-3.8-flash")  # old news ignored

    def test_health_is_saved_and_shared_through_settings(self) -> None:
        import tempfile

        from PySide6.QtCore import QSettings

        from study_assistant.providers import load_health, save_health

        folder = tempfile.mkdtemp()
        settings = QSettings(f"{folder}/s.ini", QSettings.Format.IniFormat)
        save_health(settings, {"gemini-3.5-flash": {"kind": "ok", "seconds": 1.6, "error": "", "at": 1.0}})
        save_health(settings, {"gemini-2.5-flash": {"kind": "missing", "seconds": 0.2, "error": "", "at": 1.0}})
        gemini.set_health({})
        self.assertEqual(set(load_health(QSettings(f"{folder}/s.ini", QSettings.Format.IniFormat))),
                         {"gemini-3.5-flash", "gemini-2.5-flash"})  # merged, not overwritten
        self.assertTrue(gemini.is_unavailable("gemini-2.5-flash"))


if __name__ == "__main__":
    unittest.main()
