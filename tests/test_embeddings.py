import unittest
from types import SimpleNamespace
from unittest.mock import patch

from google.genai import types

from study_assistant.gemini import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, GeminiService


class EmbeddingTests(unittest.TestCase):
    def test_query_uses_search_and_question_answering_instructions(self) -> None:
        captured = []

        def embed_content(**kwargs):
            captured.append(kwargs)
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.25] * EMBEDDING_DIMENSIONS)])

        service = object.__new__(GeminiService)
        service.client = SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))

        self.assertEqual(len(service.embed_text("priority inversion")), EMBEDDING_DIMENSIONS)
        service.embed_text("priority inversion", task="question answering")
        self.assertEqual(captured[0]["model"], EMBEDDING_MODEL)
        self.assertEqual(captured[0]["contents"], "task: search result | query: priority inversion")
        self.assertEqual(captured[1]["contents"], "task: question answering | query: priority inversion")
        self.assertIsInstance(captured[0]["config"], types.EmbedContentConfig)

    def test_index_batches_visual_pages_and_saves_every_vector(self) -> None:
        captured = []
        saved = []

        def embed_content(**kwargs):
            captured.append(kwargs)
            return SimpleNamespace(embeddings=[
                SimpleNamespace(values=[0.1] * EMBEDDING_DIMENSIONS)
                for _ in kwargs["contents"]
            ])

        service = object.__new__(GeminiService)
        service.client = SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))
        pages = [
            {
                "id": index,
                "page_number": index,
                "file_path": "/tmp/lecture.pdf",
                "class_name": "CS 354",
                "lecture_name": "Scheduling",
                "text": "Priority inversion",
                "visual_text": "handwritten annotation",
            }
            for index in range(1, 6)
        ]
        with patch("study_assistant.gemini.render_page", return_value=b"png"):
            count = service.index_pages(pages, lambda page_id, values, model: saved.append((page_id, len(values), model)))

        self.assertEqual(count, 5)
        self.assertEqual([len(call["contents"]) for call in captured], [4, 1])
        self.assertEqual([entry[0] for entry in saved], [1, 2, 3, 4, 5])
        self.assertTrue(all(isinstance(content, types.Content) for call in captured for content in call["contents"]))
        first_content = captured[0]["contents"][0]
        self.assertIn("handwritten annotation", first_content.parts[0].text)
        self.assertEqual(first_content.parts[1].inline_data.mime_type, "image/png")
        self.assertIsInstance(captured[0]["config"], types.EmbedContentConfig)

    def test_rejects_wrong_query_embedding_dimension(self) -> None:
        service = object.__new__(GeminiService)
        service.client = SimpleNamespace(models=SimpleNamespace(
            embed_content=lambda **_kwargs: SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0])])
        ))
        with self.assertRaisesRegex(RuntimeError, "invalid query embedding"):
            service.embed_text("priority inversion")


if __name__ == "__main__":
    unittest.main()
