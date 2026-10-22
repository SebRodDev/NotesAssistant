import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pymupdf
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

import study_assistant.app as app_module
from study_assistant.app import ImportDetailsDialog, MainWindow, QuizRunDialog
from study_assistant.pdf_reader import extract_page_texts
from study_assistant.storage import Library


class DesktopWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.library = Library(root / "library.sqlite3")
        self.addCleanup(self.library.close)
        for filename, text in (("week-1.pdf", "Scheduling and priority inversion."), ("week-2.pdf", "Scheduling and mutexes.")):
            path = root / filename
            document = pymupdf.open()
            page = document.new_page()
            page.insert_text((72, 72), text)
            if filename == "week-2.pdf":
                document.new_page()
            document.save(path)
            document.close()
            self.library.import_pdf("CS 354", "Scheduling", path, extract_page_texts(path))

    def test_chat_course_scope_search_and_source_preview(self) -> None:
        with patch.object(app_module, "get_api_key", return_value=None):
            window = MainWindow(self.library)
        self.addCleanup(window.close)
        self.assertEqual(window.navigator.topLevelItemCount(), 2)
        self.assertIn("3 pages", window.library_status.text())
        self.assertIn("AI indexed: 0/3", window.library_status.text())
        self.assertIn("visual checks left: 3", window.library_status.text())
        class_item = window.navigator.topLevelItem(1)
        self.assertEqual(class_item.childCount(), 1)
        self.assertEqual(class_item.child(0).text(0), "Scheduling · 3 pages")

        window.navigator.setCurrentItem(class_item.child(0))
        self.assertEqual(window.scope_title.text(), "CS 354 · Scheduling")
        window.query.setText("scheduling")
        window.search()
        self.assertEqual(len(window.message_widgets), 3)
        results = window.message_widgets[-1]
        self.assertIn("2 relevant page(s)", results.text)
        self.assertEqual(len(results.source_buttons), 2)
        results.source_buttons[0].click()
        self.assertTrue(window.preview_toggle.isChecked())
        self.assertFalse(window.preview_panel.isHidden())
        self.assertIn("Scheduling", window.preview_title.text())

        window.query.clear()
        window.search()
        self.assertEqual(len(window.message_widgets), 3)

    def test_course_scope_is_preserved_when_navigation_refreshes(self) -> None:
        with patch.object(app_module, "get_api_key", return_value=None):
            window = MainWindow(self.library)
        self.addCleanup(window.close)
        class_item = window.navigator.topLevelItem(1)
        lecture_item = class_item.child(0)
        window.navigator.setCurrentItem(lecture_item)
        window.refresh_navigation()
        selected = window.navigator.currentItem()
        self.assertEqual(selected.data(0, app_module.Qt.ItemDataRole.UserRole), ("CS 354", "Scheduling"))
        dialog = ImportDetailsDialog("/tmp/week-3.pdf", window, window.current_scope[0])
        self.assertEqual(dialog.class_name.text(), "CS 354")
        self.assertEqual(dialog.lecture_name.text(), "week-3")

    def test_last_study_space_is_restored_after_reopening(self) -> None:
        settings_path = Path(self.temp_dir.name) / "settings.ini"
        settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
        with patch.object(app_module, "get_api_key", return_value=None):
            window = MainWindow(self.library, settings=settings)
        class_item = window.navigator.topLevelItem(1)
        window.navigator.setCurrentItem(class_item.child(0))
        self.assertEqual(window.current_scope, ("CS 354", "Scheduling"))
        window.close()

        reopened_library = Library(Path(self.temp_dir.name) / "library.sqlite3")
        self.addCleanup(reopened_library.close)
        with patch.object(app_module, "get_api_key", return_value=None):
            reopened = MainWindow(reopened_library, settings=settings)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.current_scope, ("CS 354", "Scheduling"))
        self.assertEqual(reopened.scope_title.text(), "CS 354 · Scheduling")

    def test_add_pdf_button_has_native_open_shortcut(self) -> None:
        with patch.object(app_module, "get_api_key", return_value=None):
            window = MainWindow(self.library)
        self.addCleanup(window.close)
        self.assertEqual(
            window.import_button.shortcut().toString(),
            app_module.QKeySequence(app_module.QKeySequence.StandardKey.Open).toString(),
        )

    def test_chats_do_not_leak_across_classes_and_saved_class_history_restores(self) -> None:
        source = dict(self.library.search("priority inversion")[0])
        self.library.save_class_chat("CS 354", "What is priority inversion?", "It is a blocking problem.", [source])
        other_pdf = Path(self.temp_dir.name) / "memory.pdf"
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 72), "Virtual memory and paging.")
        document.save(other_pdf)
        document.close()
        self.library.import_pdf("CS 182", "Memory", other_pdf, extract_page_texts(other_pdf))

        with patch.object(app_module, "get_api_key", return_value=None):
            window = MainWindow(self.library)
        self.addCleanup(window.close)
        class_items = {
            window.navigator.topLevelItem(index).text(0): window.navigator.topLevelItem(index)
            for index in range(1, window.navigator.topLevelItemCount())
        }
        window.navigator.setCurrentItem(class_items["CS 354"])
        self.assertEqual(window.message_widgets[0].text, "What is priority inversion?")
        self.assertIn("blocking problem", window.message_widgets[1].text)

        window.navigator.setCurrentItem(class_items["CS 182"])
        self.assertEqual(len(window.message_widgets), 1)
        self.assertIn("CS 182", window.message_widgets[0].text)
        self.assertNotIn("priority inversion", window.message_widgets[0].text.lower())

        window.navigator.setCurrentItem(class_items["CS 354"])
        self.assertEqual(window.message_widgets[0].text, "What is priority inversion?")

    def test_import_parsing_runs_in_background_and_keeps_new_lecture_selected(self) -> None:
        with patch.object(app_module, "get_api_key", return_value=None):
            window = MainWindow(self.library)
        self.addCleanup(window.close)
        path = Path(self.temp_dir.name) / "week-1.pdf"
        dialog = ImportDetailsDialog(str(path), window, "CS 354")
        dialog.lecture_name.setText("New lecture")
        with (
            patch.object(app_module.QFileDialog, "getOpenFileName", return_value=(str(path), "PDF files (*.pdf)")),
            patch.object(app_module, "ImportDetailsDialog", return_value=dialog),
            patch.object(dialog, "exec", return_value=app_module.QDialog.DialogCode.Accepted),
        ):
            window.import_pdf()
        worker = window.active_task
        self.assertIsNotNone(worker)
        self.assertFalse(window.query.isEnabled())
        worker.wait()
        self.application.processEvents()
        self.assertEqual(window.current_scope, ("CS 354", "New lecture"))
        self.assertTrue(any(row["lecture_name"] == "New lecture" for row in self.library.documents()))

    def test_composer_switches_between_ask_and_find_modes(self) -> None:
        with patch.object(app_module, "get_api_key", return_value=None):
            window = MainWindow(self.library)
        self.addCleanup(window.close)
        self.assertEqual(window.mode_combo.currentData(), "ask")
        self.assertEqual(window.send_button.text(), "Ask")
        self.assertFalse(window.semantic_search.isVisible())
        window.mode_combo.setCurrentIndex(1)
        self.assertEqual(window.send_button.text(), "Find")
        self.assertFalse(window.semantic_search.isHidden())

    def test_dark_mode_persists_and_visual_notes_are_not_a_manual_action(self) -> None:
        settings = QSettings(str(Path(self.temp_dir.name) / "settings.ini"), QSettings.Format.IniFormat)
        with patch.object(app_module, "get_api_key", return_value=None):
            window = MainWindow(self.library, settings=settings)
        self.addCleanup(window.close)
        self.assertFalse(hasattr(window, "read_notes_button"))
        window.dark_mode_toggle.setChecked(True)
        self.assertIn("#171a20", window.styleSheet())
        self.assertTrue(settings.value("appearance/dark_mode", False, type=bool))
        window.dark_mode_toggle.setChecked(False)
        self.assertIn("#f6f7f9", window.styleSheet())

    def test_background_task_callback_runs_after_worker_stops(self) -> None:
        with patch.object(app_module, "get_api_key", return_value=None):
            window = MainWindow(self.library)
        self.addCleanup(window.close)
        results = []
        window._run_task(lambda _progress: "completed", results.append)
        worker = window.active_task
        self.assertIsNotNone(worker)
        self.assertFalse(window.query.isEnabled())
        worker.wait()
        self.application.processEvents()
        self.assertIsNone(window.active_task)
        self.assertEqual(results, ["completed"])
        self.assertTrue(window.query.isEnabled())

    def test_background_error_runs_recovery_callback_and_reenables_controls(self) -> None:
        with patch.object(app_module, "get_api_key", return_value=None):
            window = MainWindow(self.library)
        self.addCleanup(window.close)
        failures = []

        def fail(_progress):
            raise RuntimeError("simulated network failure")

        window._run_task(fail, lambda _result: self.fail("unexpected task success"), failures.append)
        worker = window.active_task
        worker.wait()
        self.application.processEvents()
        self.assertEqual(failures, ["simulated network failure"])
        self.assertTrue(window.query.isEnabled())
        self.assertIsNone(window.active_task)

    def test_quiz_dialog_allows_retry_after_grading_failure(self) -> None:
        question = {
            "question_type": "short_answer",
            "prompt": "Explain the concept.",
            "options": [],
            "correct_answer": "A task waits.",
            "rubric": "Name the blocking relationship.",
            "explanation": "The source describes a task waiting.",
            "topics": ["scheduling"],
            "source_ids": ["S1"],
            "source_map": {
                "S1": {
                    "class_name": "CS 354",
                    "lecture_name": "Scheduling",
                    "file_path": "/tmp/lecture.pdf",
                    "page_number": 1,
                }
            },
        }
        dialog = QuizRunDialog([question])
        dialog.show()
        self.application.processEvents()
        dialog.set_grading(True)
        dialog.reject()
        self.assertTrue(dialog.isVisible())
        dialog.grading_failed()
        self.assertTrue(dialog.submit_button.isEnabled())
        self.assertTrue(dialog.cancel_button.isEnabled())
        self.assertFalse(dialog.grading)
        dialog.reject()
        self.assertFalse(dialog.isVisible())


if __name__ == "__main__":
    unittest.main()
