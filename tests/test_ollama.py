import tempfile
import unittest
from pathlib import Path

import pymupdf

from fake_ollama import FakeOllama
from study_assistant.ollama import (
    NOT_FOUND_TOKEN,
    ChatResult,
    ChatStats,
    OllamaClient,
    OllamaError,
    OllamaUnavailable,
    StudyModels,
    describe_runtime,
    embedding_prefixes,
    finalize_answer,
    validate_quiz,
)


def make_source(pdf: Path, source_id: str = "S1", text: str = "Priority inversion blocks a high-priority task.") -> dict:
    return {
        "source_id": source_id, "page_id": 1, "class_name": "CS 354", "lecture_name": "Scheduling",
        "file_path": str(pdf), "page_number": 1, "text": text, "visual_text": "",
    }


class PureHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = [make_source(Path("/tmp/a.pdf"), "S1"), make_source(Path("/tmp/b.pdf"), "S2")]

    def test_citations_are_renumbered_and_invented_ids_dropped(self) -> None:
        result = ChatResult(content="Fact one [S2]. Fact two [S7][S1]. Again [S2].", stats=ChatStats("m", 10, 10**9, 90))
        answer = finalize_answer(result, self.sources, "strict", num_ctx=1000)
        self.assertEqual(answer["answer"], "Fact one [1]. Fact two [2]. Again [1].")
        self.assertEqual([s["source_id"] for s in answer["sources"]], ["S2", "S1"])
        self.assertAlmostEqual(answer["tokens_per_second"], 10.0)
        self.assertAlmostEqual(answer["context_used"], 0.1)

    def test_not_found_token_becomes_helpful_message(self) -> None:
        answer = finalize_answer(ChatResult(content=NOT_FOUND_TOKEN), self.sources, "strict")
        self.assertFalse(answer["answerable"])
        self.assertIn("couldn't find that in your notes", answer["answer"])

    def test_strict_answer_without_citations_is_flagged(self) -> None:
        answer = finalize_answer(ChatResult(content="Something uncited."), self.sources, "strict")
        self.assertIn("didn't cite", answer["answer"])

    def test_inline_think_blocks_are_removed(self) -> None:
        answer = finalize_answer(ChatResult(content="<think>hmm</think>Hello"), [], "off")
        self.assertEqual(answer["answer"], "Hello")

    def test_quiz_validation_keeps_only_well_formed_questions(self) -> None:
        parsed = {"questions": [
            {"question_type": "multiple_choice", "prompt": "P", "options": ["A) x", "B) y", "C) z", "D) w"],
             "correct_answer": "b) y", "rubric": "r", "explanation": "e", "topics": ["t"], "source_ids": ["S1"]},
            {"question_type": "multiple_choice", "prompt": "bad", "options": ["x", "x", "y", "z"],
             "correct_answer": "A", "rubric": "r", "explanation": "e", "topics": ["t"], "source_ids": ["S1"]},
            {"question_type": "short_answer", "prompt": "uncited", "options": [], "correct_answer": "abc",
             "rubric": "r", "explanation": "e", "topics": ["t"], "source_ids": ["S9"]},
        ]}
        questions = validate_quiz(parsed, 3, "mixed", self.sources)
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["options"], ["x", "y", "z", "w"])
        self.assertEqual(questions[0]["correct_answer"], "B")
        with self.assertRaises(OllamaError):
            validate_quiz({"questions": []}, 1, "mixed", self.sources)

    def test_embedding_prefixes_match_model_family(self) -> None:
        self.assertEqual(embedding_prefixes("nomic-embed-text:latest")[0], "search_query: ")
        self.assertEqual(embedding_prefixes("embeddinggemma")[1], "title: none | text: ")
        self.assertEqual(embedding_prefixes("some-other-embed"), ("", ""))

    def test_runtime_description(self) -> None:
        running = [{"name": "m", "size": 100, "size_vram": 50}]
        self.assertEqual(describe_runtime(running, "m")["where"], "GPU 50% · CPU")
        self.assertFalse(describe_runtime(running, "other")["loaded"])


class FakeServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.pdf = Path(self.temp_dir.name) / "scheduling.pdf"
        document = pymupdf.open()
        document.new_page().insert_text((72, 72), "Priority inversion.")
        document.save(self.pdf)
        document.close()
        self.fake = FakeOllama().__enter__()
        self.addCleanup(self.fake.__exit__)
        self.models = StudyModels(OllamaClient(self.fake.host))

    def test_streaming_answer_with_reasoning_and_validated_citations(self) -> None:
        tokens, thoughts = [], []
        result = self.models.stream_answer(
            "What is priority inversion?", [make_source(self.pdf)], persona="Be brief.", notes_mode="strict",
            reasoning=True, on_token=tokens.append, on_thinking=thoughts.append,
        )
        self.assertGreater(len(tokens), 5)
        self.assertTrue(thoughts)
        self.assertIn("[1]", result["answer"])
        self.assertNotIn("S9", result["answer"])
        self.assertEqual(len(result["sources"]), 1)
        request = self.fake.requests[-1]
        self.assertTrue(request["think"])
        self.assertIn("Be brief.", request["messages"][0]["content"])
        self.assertIn("[S1]", request["messages"][0]["content"])  # slides lead, right after the instructions
        self.assertEqual(request["messages"][-1]["content"], "Question: What is priority inversion?")

    def test_off_mode_sends_no_notes_and_history_is_kept(self) -> None:
        self.models.stream_answer("hi", [], notes_mode="off", history=[("user", "earlier"), ("assistant", "reply")])
        messages = self.fake.requests[-1]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[-1]["content"], "hi")

    def test_embeddings_use_task_prefixes(self) -> None:
        saved = {}
        page = {**make_source(self.pdf), "id": 7}
        self.assertEqual(self.models.index_pages([page], lambda pid, vec, model: saved.update({pid: model})), 1)
        self.assertEqual(saved, {7: "nomic-embed-text"})
        self.assertTrue(self.fake.requests[-1]["input"][0].startswith("search_document: "))
        self.assertEqual(len(self.models.embed_query("priority")), 16)
        self.assertTrue(self.fake.requests[-1]["input"][0].startswith("search_query: "))

    def test_quiz_generation_and_grading(self) -> None:
        source = make_source(self.pdf)
        questions = self.models.generate_quiz(2, "mixed", [source])
        self.assertEqual({q["question_type"] for q in questions}, {"multiple_choice", "short_answer"})
        self.assertIn("format", self.fake.requests[-1])
        grade = self.models.grade_short_answer(questions[1], "A high task waits.", [source])
        self.assertTrue(grade["is_correct"])
        self.assertIn("CS 354 · Scheduling", grade["feedback"])

    def test_handwriting_uses_vision_model_with_page_image(self) -> None:
        saved = {}
        self.assertTrue(self.models.vision_ready())
        page = {**make_source(self.pdf), "id": 3}
        self.models.read_visual_notes([page], lambda pid, text: saved.update({pid: text}))
        self.assertEqual(saved, {3: "handwritten: mutex!"})
        self.assertTrue(self.fake.requests[-1]["messages"][1]["images"])

    def test_missing_model_has_pull_instructions(self) -> None:
        with self.assertRaisesRegex(OllamaError, "ollama pull llama9"):
            StudyModels(OllamaClient(self.fake.host), chat_model="llama9").stream_answer("q", [], notes_mode="off")

    def test_pull_reports_progress(self) -> None:
        updates = []
        OllamaClient(self.fake.host).pull("qwen3:4b", updates.append)
        self.assertEqual(updates, ["Downloading qwen3:4b: 50% of 0.0 GB…"])

    def test_pull_errors_are_raised(self) -> None:
        with self.assertRaisesRegex(OllamaError, "file does not exist"):
            OllamaClient(self.fake.host).pull("no-such-model")

    def test_pull_that_ends_early_is_an_error(self) -> None:
        with self.assertRaisesRegex(OllamaError, "stopped before it finished"):
            OllamaClient(self.fake.host).pull("cut-off-model")

    def test_offline_server_is_explained(self) -> None:
        with self.assertRaisesRegex(OllamaUnavailable, "Open the Ollama app"):
            OllamaClient("http://127.0.0.1:9").list_models(timeout=1)


if __name__ == "__main__":
    unittest.main()
