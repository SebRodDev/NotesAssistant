import tempfile
import unittest
import sqlite3
from pathlib import Path

from study_assistant.storage import Library


class LibraryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.pdf = self.root / "scheduling.pdf"
        self.pdf.write_bytes(b"test fixture path; PDF parsing is tested separately")
        self.library = Library(self.root / "library.sqlite3")
        self.addCleanup(self.library.close)
        self.library.import_pdf(
            "CS 354", "Scheduling", self.pdf,
            ["Priority inversion occurs when a high-priority task waits on a low-priority task."],
        )
        self.page_id = self.library.connection.execute("SELECT id FROM pages").fetchone()[0]

    def test_keyword_search_and_scope_filter(self) -> None:
        all_results = self.library.search("Which slides cover priority inversion?")
        self.assertEqual(len(all_results), 1)
        self.assertEqual(all_results[0]["page_number"], 1)
        self.assertEqual(self.library.search("priority inversion", class_name="CS 999"), [])
        self.assertEqual(self.library.search("priority inversion", lecture_name="Other"), [])

    def test_indexing_status_uses_compact_page_counts(self) -> None:
        self.assertEqual(self.library.indexing_status("test-model"), {
            "total_pages": 1, "indexed_pages": 0, "pending_visual_pages": 1,
        })
        self.library.save_visual_text(self.page_id, "")
        self.library.save_embedding(self.page_id, [1.0, 0.0], "test-model")
        self.assertEqual(self.library.indexing_status("test-model"), {
            "total_pages": 1, "indexed_pages": 1, "pending_visual_pages": 0,
        })

    def test_visual_text_is_searchable_and_invalidates_vector(self) -> None:
        self.assertEqual(len(self.library.pages_needing_visual_scan()), 1)
        self.library.save_embedding(self.page_id, [1.0, 0.0], "test-model")
        self.library.save_visual_text(self.page_id, "handwritten note: priority inheritance")
        rows = self.library.search("priority inheritance")
        self.assertEqual(rows[0]["visual_text"], "handwritten note: priority inheritance")
        self.assertEqual(self.library.pages_needing_visual_scan(), [])
        self.assertEqual(self.library.pages_to_index("test-model")[0]["id"], self.page_id)

    def test_semantic_search_and_class_history(self) -> None:
        self.library.save_embedding(self.page_id, [1.0, 0.0], "test-model")
        result = self.library.search(
            "scheduling concept", query_embedding=[1.0, 0.0], model="test-model",
        )
        self.assertEqual(result[0]["match_type"], "semantic")
        source = {key: result[0][key] for key in (
            "class_name", "lecture_name", "file_path", "page_number", "text", "visual_text", "page_id"
        )}
        self.library.save_class_chat("CS 354", "What is it?", "Grounded reply", [source])
        history = self.library.class_chat_history("CS 354")
        self.assertEqual(history[0]["question"], "What is it?")
        self.assertEqual(history[0]["sources"][0]["page_number"], 1)
        self.library.save_class_chat("CS 354", "What is the deadline?", "No deadline is listed.", [])
        self.assertEqual(self.library.class_chat_history("CS 354", limit=1)[0]["question"], "What is the deadline?")
        with self.assertRaisesRegex(ValueError, "positive"):
            self.library.class_chat_history("CS 354", limit=0)

    def test_reimport_replaces_stale_text_and_embedding(self) -> None:
        self.library.save_embedding(self.page_id, [1.0, 0.0], "test-model")
        self.library.import_pdf("CS 354", "Scheduling", self.pdf, ["A new page about mutexes."])
        self.assertEqual(self.library.search("priority inversion"), [])
        result = self.library.search("mutexes")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "A new page about mutexes.")
        self.assertEqual(self.library.pages_to_index("test-model")[0]["text"], "A new page about mutexes.")

    def test_quiz_page_selection_covers_selected_lectures(self) -> None:
        second_pdf = self.root / "memory.pdf"
        second_pdf.write_bytes(b"fixture")
        self.library.import_pdf("CS 354", "Memory", second_pdf, ["First memory page", "Last memory page"])
        self.library.import_pdf(
            "CS 354", "Scheduling", self.root / "other.pdf",
            [f"Scheduling page {number}" for number in range(1, 31)],
        )
        pages = self.library.pages_for_scopes([("CS 354", "Scheduling"), ("CS 354", "Memory")], limit=8)
        self.assertLessEqual(len(pages), 8)
        self.assertEqual({(page["class_name"], page["lecture_name"]) for page in pages}, {
            ("CS 354", "Scheduling"), ("CS 354", "Memory"),
        })
        self.assertEqual(sum(page["lecture_name"] == "Memory" for page in pages), 2)

    def test_topic_performance_deduplicates_case_variants(self) -> None:
        question_template = {
            "question_type": "short_answer",
            "prompt": "Explain the concept.",
            "options": [],
            "correct_answer": "A task is blocked.",
            "rubric": "Name the blocking relationship.",
            "explanation": "The page describes priority inversion.",
            "topics": ["Priority Inversion", "priority inversion"],
            "source_ids": ["S1"],
            "source_map": {"S1": {"id": self.page_id}},
        }
        saved = self.library.create_quiz_session(
            [("CS 354", "Scheduling")], [question_template, question_template]
        )
        self.library.record_quiz_attempt(saved[0]["id"], "Wrong answer", 0.0, False, "Review this.")
        self.library.record_quiz_attempt(saved[1]["id"], "Correct answer", 1.0, True, "Good.")
        performance = self.library.topic_performance()
        self.assertEqual(len(performance), 1)
        self.assertEqual(performance[0]["attempts"], 2)
        self.assertEqual(performance[0]["misses"], 1)
        self.assertGreater(performance[0]["accuracy"], 0)

    def test_targeted_review_can_persist_search_result_sources(self) -> None:
        source = self.library.search("priority inversion")[0]
        question = {
            "question_type": "short_answer",
            "prompt": "Apply the concept.",
            "options": [],
            "correct_answer": "A task is blocked.",
            "rubric": "Name the blocking relationship.",
            "explanation": "The lecture describes priority inversion.",
            "topics": ["priority inversion"],
            "source_ids": ["S1"],
            "source_map": {"S1": source},
        }
        saved = self.library.create_quiz_session(
            [("CS 354", "Scheduling")], [question]
        )
        self.assertEqual(saved[0]["source_page_ids"], [self.page_id])

    def test_rejects_invalid_embedding_and_quiz_scores(self) -> None:
        with self.assertRaises(ValueError):
            self.library.save_embedding(self.page_id, [float("nan")], "test-model")
        with self.assertRaises(ValueError):
            self.library.search("priority inversion", query_embedding=[float("inf")])
        with self.assertRaises(ValueError):
            self.library.search("priority inversion", limit=0)

    def test_migrates_version_one_page_index_and_visual_notes(self) -> None:
        legacy_path = self.root / "legacy.sqlite3"
        connection = sqlite3.connect(legacy_path)
        connection.executescript(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                class_name TEXT NOT NULL,
                lecture_name TEXT NOT NULL,
                file_path TEXT NOT NULL UNIQUE,
                imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE pages (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                page_number INTEGER NOT NULL,
                text TEXT NOT NULL,
                UNIQUE(document_id, page_number)
            );
            CREATE VIRTUAL TABLE pages_fts USING fts5(text, content='pages', content_rowid='id');
            CREATE TRIGGER pages_ai AFTER INSERT ON pages BEGIN
                INSERT INTO pages_fts(rowid, text) VALUES (new.id, new.text);
            END;
            INSERT INTO documents(id, class_name, lecture_name, file_path)
                VALUES (1, 'CS 354', 'Scheduling', '/legacy/slides.pdf');
            INSERT INTO pages(id, document_id, page_number, text)
                VALUES (1, 1, 1, 'Legacy priority inversion page');
            PRAGMA user_version = 1;
            """
        )
        connection.commit()
        connection.close()

        migrated = Library(legacy_path)
        self.addCleanup(migrated.close)
        self.assertEqual(migrated.connection.execute("PRAGMA user_version").fetchone()[0], 3)
        self.assertEqual(len(migrated.pages_needing_visual_scan()), 1)
        self.assertEqual(migrated.search("priority inversion")[0]["page_id"], 1)
        migrated.save_visual_text(1, "handwritten priority inheritance")
        self.assertEqual(migrated.search("priority inheritance")[0]["visual_text"], "handwritten priority inheritance")


if __name__ == "__main__":
    unittest.main()
