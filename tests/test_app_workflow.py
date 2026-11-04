import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pymupdf
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

import study_assistant.app as app_module
from fake_ollama import FakeOllama
from study_assistant.app import MainWindow
from study_assistant.dialogs import AgentEditorDialog, BatchImportDetailsDialog, ImportDetailsDialog, QuizRunDialog
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
            document.new_page().insert_text((72, 72), text)
            if filename == "week-2.pdf":
                document.new_page()
            document.save(path)
            document.close()
            self.library.import_pdf("CS 354", "Scheduling", path, extract_page_texts(path))
        self.settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)

    def window(self, host: str | None = None) -> MainWindow:
        if host:
            self.settings.setValue("models/host", host)
        window = MainWindow(self.library, settings=self.settings, start_monitor=False)
        self.addCleanup(window.close)
        return window

    def wait(self, window: MainWindow) -> None:
        deadline = time.time() + 20
        while window.active_task and time.time() < deadline:
            self.application.processEvents()
            time.sleep(0.01)
        self.application.processEvents()
        self.assertIsNone(window.active_task)

    def fake_server(self, **kwargs) -> FakeOllama:
        fake = FakeOllama(**kwargs).__enter__()
        self.addCleanup(fake.__exit__)
        return fake

    def test_layout_has_agents_courses_and_neural_link_panel(self) -> None:
        window = self.window()
        self.assertEqual(len(window.agent_cards), 3)
        self.assertEqual(window.agent_title.text(), "TUTOR")
        self.assertEqual(window.panel_name.text(), "TUTOR")
        self.assertEqual(window.navigator.topLevelItemCount(), 2)
        self.assertIn("3 PAGES", window.library_status.text())
        self.assertIn("0/3 INDEXED", window.library_status.text())
        self.assertEqual(window.message_widgets[0].kind, "intro")
        window.panel_button.setChecked(False)
        self.assertTrue(window.right_panel.isHidden())

    def test_find_page_and_source_opens_in_right_panel(self) -> None:
        window = self.window()
        class_item = window.navigator.topLevelItem(1)
        window.navigator.setCurrentItem(class_item.child(0))
        self.assertEqual(window.scope_title.text(), "CS 354 · SCHEDULING")
        window.set_mode("find")
        window.semantic_search.setChecked(False)
        self.assertEqual(window.send_button.text(), "FIND ▸")
        window.query.setText("scheduling")
        window.send_message()
        results = window.message_widgets[-1]
        self.assertIn("**2** page(s)", results.text)
        self.assertEqual(len(results.source_buttons), 2)
        results.source_buttons[0].click()
        self.assertEqual(window.right_stack.currentIndex(), 1)
        self.assertIn("SCHEDULING", window.preview_title.text())

    def test_ask_streams_answer_with_citations_and_stats(self) -> None:
        fake = self.fake_server()
        window = self.window(fake.host)
        window.navigator.setCurrentItem(window.navigator.topLevelItem(1))
        window.query.setText("What is priority inversion?")
        window.send_message()
        self.assertIsNotNone(window.active_task)
        self.assertFalse(window.query.isEnabled())
        self.wait(window)
        answer = window.message_widgets[-1]
        self.assertIn("[1]", answer.text)
        self.assertTrue(answer.sources)
        self.assertEqual(window.stat_speed.text(), "40 t/s")
        self.assertIn("of 16k", window.stat_context.text())
        self.assertIn("SEARCHED NOTES", answer.retrieval_label.text())
        history = self.library.class_chat_history("CS 354", agent_id=window.current_agent["id"])
        self.assertEqual(history[-1]["question"], "What is priority inversion?")
        # conversation history is sent on the next turn
        window.query.setText("And how is it fixed?")
        window.send_message()
        self.wait(window)
        roles = [m["role"] for m in fake.requests[-1]["messages"]]
        self.assertEqual(roles[:3], ["system", "user", "assistant"])

    def test_regenerate_replaces_last_answer(self) -> None:
        fake = self.fake_server()
        window = self.window(fake.host)
        window.query.setText("What is priority inversion?")
        window.send_message()
        self.wait(window)
        count = len(window.message_widgets)
        window.regenerate(window.message_widgets[-1])
        self.wait(window)
        self.assertEqual(len(window.message_widgets), count)
        self.assertEqual(window.message_widgets[-2].text, "What is priority inversion?")

    def test_offline_ollama_shows_readable_error_and_reenables_controls(self) -> None:
        window = self.window("http://127.0.0.1:9")
        window.query.setText("What is priority inversion?")
        window.send_message()
        self.wait(window)
        self.assertIn("Ollama", window.message_widgets[-1].text)
        self.assertEqual(window.message_widgets[-1].kind, "error")
        self.assertTrue(window.query.isEnabled())

    def test_index_embeds_pages_and_reads_handwriting_locally(self) -> None:
        fake = self.fake_server()
        window = self.window(fake.host)
        window.index_library()
        self.wait(window)
        self.assertIn("3/3 INDEXED", window.library_status.text())
        status = self.library.indexing_status("nomic-embed-text")
        self.assertEqual(status["pending_visual_pages"], 0)
        results = self.library.search("mutex")
        self.assertIn("mutex", results[0]["visual_text"])

    def test_agents_switch_conversations_and_can_be_created(self) -> None:
        window = self.window()
        window.navigator.setCurrentItem(window.navigator.topLevelItem(1))
        tutor_intro = window.message_widgets[0].text
        explainer = window.library.agents()[1]
        window.select_agent(explainer["id"])
        self.assertEqual(window.agent_title.text(), "EXPLAINER")
        self.assertNotEqual(window.message_widgets[0].text, tutor_intro)
        self.assertEqual(self.settings.value("agents/current", type=int), explainer["id"])

        dialog = AgentEditorDialog(None, ["qwen3:8b", "nomic-embed-text:latest"], "qwen3:8b", window)
        self.addCleanup(dialog.close)
        self.assertEqual(dialog.model_combo.count(), 2)  # default + qwen3 (embedding models are hidden)
        dialog.name_input.setText("OS TA")
        dialog.creativity.setValue(90)
        self.assertIn("CREATIVE", dialog.creativity_value.text())
        with (
            patch.object(app_module, "AgentEditorDialog", return_value=dialog),
            patch.object(dialog, "exec", return_value=app_module.QDialog.DialogCode.Accepted),
        ):
            window.create_agent()
        self.assertEqual(window.current_agent["name"], "OS TA")
        self.assertEqual(len(window.agent_cards), 4)

    def test_last_study_space_and_agent_are_restored(self) -> None:
        window = self.window()
        window.navigator.setCurrentItem(window.navigator.topLevelItem(1).child(0))
        window.select_agent(window.library.agents()[2]["id"])
        window.close()
        reopened_library = Library(Path(self.temp_dir.name) / "library.sqlite3")
        self.addCleanup(reopened_library.close)
        reopened = MainWindow(reopened_library, settings=self.settings, start_monitor=False)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.current_scope, ("CS 354", "Scheduling"))
        self.assertEqual(reopened.current_agent["name"], "Free chat")

    def test_chats_do_not_leak_across_classes_and_saved_history_restores(self) -> None:
        agent_id = self.library.agents()[0]["id"]
        source = dict(self.library.search("priority inversion")[0])
        self.library.save_class_chat("CS 354", "What is priority inversion?", "It is a blocking problem.", [source], agent_id)
        other_pdf = Path(self.temp_dir.name) / "memory.pdf"
        document = pymupdf.open()
        document.new_page().insert_text((72, 72), "Virtual memory and paging.")
        document.save(other_pdf)
        document.close()
        self.library.import_pdf("CS 182", "Memory", other_pdf, extract_page_texts(other_pdf))
        window = self.window()
        items = {window.navigator.topLevelItem(i).text(0): window.navigator.topLevelItem(i)
                 for i in range(1, window.navigator.topLevelItemCount())}
        window.navigator.setCurrentItem(items["CS 354"])
        self.assertEqual(window.message_widgets[0].text, "What is priority inversion?")
        window.navigator.setCurrentItem(items["CS 182"])
        self.assertEqual(len(window.message_widgets), 1)
        self.assertIn("CS 182", window.message_widgets[0].text)
        window.navigator.setCurrentItem(items["CS 354"])
        self.assertEqual(window.message_widgets[0].text, "What is priority inversion?")

    def test_import_runs_in_background_and_selects_new_lecture(self) -> None:
        window = self.window()
        path = Path(self.temp_dir.name) / "week-1.pdf"
        dialog = ImportDetailsDialog(str(path), window, "CS 354")
        dialog.lecture_name.setText("New lecture")
        with (
            patch.object(app_module.QFileDialog, "getOpenFileNames", return_value=([str(path)], "PDF files (*.pdf)")),
            patch.object(app_module, "ImportDetailsDialog", return_value=dialog),
            patch.object(dialog, "exec", return_value=app_module.QDialog.DialogCode.Accepted),
        ):
            window.import_pdf()
        self.assertIsNotNone(window.active_task)
        self.wait(window)
        self.assertEqual(window.current_scope, ("CS 354", "New lecture"))

    def test_batch_import_dialog_keeps_per_pdf_labels(self) -> None:
        files = ["/tmp/week-3.pdf", "/tmp/week-4.pdf"]
        dialog = BatchImportDetailsDialog(files, default_class_name="CS 354")
        self.addCleanup(dialog.close)
        dialog.table.item(1, 1).setText("CS 182")
        self.assertEqual(dialog.import_specs()[1], (Path(files[1]), "CS 182", "week-4"))

    def test_add_pdf_button_has_native_open_shortcut(self) -> None:
        window = self.window()
        self.assertEqual(
            window.import_button.shortcut().toString(),
            app_module.QKeySequence(app_module.QKeySequence.StandardKey.Open).toString(),
        )

    def test_background_error_runs_recovery_callback(self) -> None:
        window = self.window()
        failures = []

        def fail(_progress):
            raise RuntimeError("simulated failure")

        window._run_task(fail, lambda _result: self.fail("unexpected"), failures.append)
        self.wait(window)
        self.assertEqual(failures, ["simulated failure"])
        self.assertTrue(window.query.isEnabled())

    def test_quiz_flow_with_local_model(self) -> None:
        fake = self.fake_server()
        window = self.window(fake.host)
        shown = []
        with patch.object(window, "_show_quiz", side_effect=shown.append):
            window.generate_quiz([("CS 354", "Scheduling")], 2, "mixed")
            self.wait(window)
        self.assertEqual(len(shown[0]["questions"]), 2)
        self.assertTrue(all(q["id"] for q in shown[0]["questions"]))

    def test_quiz_dialog_allows_retry_after_grading_failure(self) -> None:
        question = {
            "question_type": "short_answer", "prompt": "Explain the concept.", "options": [],
            "correct_answer": "A task waits.", "rubric": "Name the blocking relationship.",
            "explanation": "The source describes a task waiting.", "topics": ["scheduling"], "source_ids": ["S1"],
            "source_map": {"S1": {"class_name": "CS 354", "lecture_name": "Scheduling",
                                  "file_path": "/tmp/lecture.pdf", "page_number": 1}},
        }
        dialog = QuizRunDialog([question])
        dialog.show()
        self.application.processEvents()
        dialog.set_grading(True)
        dialog.reject()
        self.assertTrue(dialog.isVisible())
        dialog.grading_failed()
        self.assertTrue(dialog.submit_button.isEnabled())
        dialog.close()


if __name__ == "__main__":
    unittest.main()
