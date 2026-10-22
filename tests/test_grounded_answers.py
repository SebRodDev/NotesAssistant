import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pymupdf
from google.genai import types

from study_assistant.gemini import GeminiService


class GroundedAnswerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.pdf = Path(self.temp_dir.name) / "scheduling.pdf"
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), "A high-priority task can be blocked by a low-priority task.")
        document.save(self.pdf)
        document.close()
        self.source = {
            "source_id": "S1",
            "class_name": "CS 354",
            "lecture_name": "Scheduling",
            "file_path": str(self.pdf),
            "page_number": 1,
            "text": "Priority inversion blocks a high-priority task behind a low-priority task.",
            "visual_text": "",
        }

    def make_service(self, payload: dict) -> tuple[GeminiService, dict]:
        response = SimpleNamespace(text=json.dumps(payload))
        request = {}

        def generate_content(**kwargs):
            request.update(kwargs)
            return response

        service = object.__new__(GeminiService)
        service.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        return service, request

    def test_valid_source_id_becomes_application_citation(self) -> None:
        payload = {"status": "answered", "claims": [{"text": "A high-priority task waits behind a low-priority task.", "source_ids": ["S1"]}]}
        with patch("study_assistant.gemini.render_page", return_value=b"png"):
            service, request = self.make_service(payload)
            result = service.answer_grounded("What is priority inversion?", [self.source])
        self.assertIn("CS 354 · Scheduling · scheduling.pdf · p. 1", result["answer"])
        self.assertEqual(result["used_source_ids"], ["S1"])
        self.assertIsInstance(request["config"], types.GenerateContentConfig)
        self.assertEqual(request["config"].response_mime_type, "application/json")
        self.assertEqual(request["config"].response_json_schema["type"], "object")
        self.assertIsInstance(request["contents"][1], types.Content)

    def test_model_cannot_cite_unretrieved_source(self) -> None:
        payload = {"status": "answered", "claims": [{"text": "Unsupported claim.", "source_ids": ["S9"]}]}
        with patch("study_assistant.gemini.render_page", return_value=b"png"):
            service, _request = self.make_service(payload)
            with self.assertRaisesRegex(RuntimeError, "not retrieved"):
                service.answer_grounded("What is priority inversion?", [self.source])

    def test_not_found_is_explicit_and_uncited(self) -> None:
        with patch("study_assistant.gemini.render_page", return_value=b"png"):
            service, _request = self.make_service({"status": "not_found", "claims": []})
            result = service.answer_grounded("What is priority inversion?", [self.source])
        self.assertFalse(result["answerable"])
        self.assertEqual(result["used_source_ids"], [])
        self.assertIn("couldn't find enough support", result["answer"])

    def test_visual_note_batch_validates_page_ids_and_saves_empty_results(self) -> None:
        page = {
            "id": 12, "page_number": 3, "text": "printed", "class_name": "CS 354",
            "lecture_name": "Scheduling", "file_path": str(self.pdf),
        }
        service, request = self.make_service({"pages": [{"page_id": 12, "notes": "  [unclear] note  "}]})
        saved = []
        with patch("study_assistant.gemini.render_page", return_value=b"image"):
            count = service.read_visual_notes_batch([page], lambda page_id, text: saved.append((page_id, text)))
        self.assertEqual(count, 1)
        self.assertEqual(saved, [(12, "[unclear] note")])
        self.assertEqual(request["model"], "gemini-3.5-flash-lite")
        self.assertIsInstance(request["config"], types.GenerateContentConfig)
        self.assertEqual(request["config"].response_json_schema["properties"]["pages"]["items"]["required"], ["page_id", "notes"])

        empty, _request = self.make_service({"pages": [{"page_id": 12, "notes": ""}]})
        empty_saved = []
        with patch("study_assistant.gemini.render_page", return_value=b"image"):
            empty.read_visual_notes_batch([page], lambda page_id, text: empty_saved.append((page_id, text)))
        self.assertEqual(empty_saved, [(12, "")])

        invalid, _request = self.make_service({"pages": []})
        with patch("study_assistant.gemini.render_page", return_value=b"image"):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                invalid.read_visual_notes_batch([page], lambda *_args: None)

    def test_quiz_generation_uses_sdk_valid_json_schema_config(self) -> None:
        question = {
            "question_type": "short_answer",
            "prompt": "Apply the scheduler rule to this scenario.",
            "options": [],
            "correct_answer": "The high-priority task waits.",
            "rubric": "Identify the blocking relationship.",
            "explanation": "The lecture describes priority inversion.",
            "topics": ["priority inversion"],
            "source_ids": ["S1"],
        }
        service, request = self.make_service({"questions": [question]})
        with patch("study_assistant.gemini.render_page", return_value=b"png"):
            generated = service.generate_quiz(1, "short_answer", [self.source])
        self.assertEqual(generated[0]["source_ids"], ["S1"])
        self.assertIsInstance(request["config"], types.GenerateContentConfig)
        self.assertIsNone(request["config"].temperature)

    def test_short_answer_grading_uses_sdk_valid_config_and_validates_score(self) -> None:
        service, request = self.make_service({"score": 0.5, "feedback": "Some reasoning is correct.", "source_ids": ["S1"]})
        question = {"prompt": "Explain the effect.", "correct_answer": "A task waits.", "rubric": "Name the wait."}
        with patch("study_assistant.gemini.render_page", return_value=b"png"):
            result = service.grade_short_answer(question, "It waits.", [self.source])
        self.assertEqual(result["score"], 0.5)
        self.assertIn("scheduling.pdf · p. 1", result["feedback"])
        self.assertIsInstance(request["config"], types.GenerateContentConfig)

        bad_service, _request = self.make_service({"score": True, "feedback": "Wrong type.", "source_ids": ["S1"]})
        with patch("study_assistant.gemini.render_page", return_value=b"png"):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                bad_service.grade_short_answer(question, "It waits.", [self.source])


if __name__ == "__main__":
    unittest.main()
