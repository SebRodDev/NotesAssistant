import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pymupdf
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication

import study_assistant.app as app_module
from fake_ollama import FakeOllama
from study_assistant.widgets import ChatMessageWidget
from study_assistant.app import MainWindow
from study_assistant.dialogs import (
    AgentEditorDialog,
    BatchImportDetailsDialog,
    ImportDetailsDialog,
    ModelField,
    ModelSettingsDialog,
    QuizRunDialog,
)
from study_assistant.pdf_reader import extract_page_texts
from study_assistant.hardware import models_by_role
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
        self.settings.setValue("models/provider", "ollama")  # most tests drive the fake Ollama server

    def window(self, host: str | None = None) -> MainWindow:
        if host:
            self.settings.setValue("models/host", host)
        window = MainWindow(self.library, settings=self.settings, start_monitor=False)
        window._auto_indexed = True  # tests start indexing explicitly; test_auto_index covers the automatic pass
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
        self.assertIn("0 indexed for search", window.library_status.toolTip())
        self.assertEqual(window.message_widgets[0].kind, "intro")
        self.assertTrue(window.right_panel.isHidden())  # only opens to preview a cited page
        window.show_link_card()  # the header badge opens the agent card
        self.assertTrue(window.link_card.isVisible())
        self.assertEqual(window.side_tabs.tabBar().elideMode(), Qt.TextElideMode.ElideNone)

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
        self.assertFalse(results.sources_panel.isHidden())  # search results start open
        results.source_buttons[0].click()
        self.assertEqual(window.right_stack.currentIndex(), 1)
        self.assertFalse(window.right_panel.isHidden())
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
        self.assertEqual(answer.retrieval_label.text(), "◆ READ ALL 2 PAGES OF CS 354")  # blank page skipped
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
        self.assertIn("3 indexed for search", window.library_status.toolTip())
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
        combo = dialog.model_field.combo
        self.assertLess(combo.findData("nomic-embed-text:latest"), 0)  # embedding models can't chat
        self.assertGreaterEqual(combo.findData("qwen3:8b"), 0)
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

    def test_model_dropdown_lists_only_installed_models_and_cannot_be_typed_in(self) -> None:
        field = ModelField("chat", "http://127.0.0.1:9")
        field.set_models(["qwen3:8b", "gemma3:4b"], "", "Default (qwen3:8b)")
        self.assertFalse(field.combo.isEditable())
        self.assertEqual([field.combo.itemText(i) for i in range(field.combo.count())],
                         ["Default (qwen3:8b)", "qwen3:8b", "gemma3:4b"])
        self.assertEqual(field.model_name(), "")
        self.assertLess(field.combo.findData("qwen3:4b"), 0)  # downloadable, but not installed
        self.assertTrue(field.download_button.isHidden())
        menu = [action.text() for action in field.more_menu.actions()]
        self.assertIn("qwen3:4b  ·  2.5 GB", menu)
        self.assertNotIn("qwen3:8b  ·  5.2 GB", menu)

    def test_saved_model_that_is_not_installed_is_flagged(self) -> None:
        field = ModelField("chat", "http://127.0.0.1:9")
        field.set_models(["qwen3:8b"], "qwen3:4b")
        self.assertIn("NOT INSTALLED", field.combo.currentText())
        self.assertEqual(field.model_name(), "qwen3:4b")
        self.assertFalse(field.download_button.isHidden())
        self.assertEqual(field.download_button.text(), "DOWNLOAD 2.5 GB")
        self.assertIn("isn't installed", field.status.text())
        field.combo.setCurrentIndex(field.combo.findData("qwen3:8b"))
        self.assertTrue(field.download_button.isHidden())

    def test_empty_install_shows_a_placeholder_that_cannot_be_chosen(self) -> None:
        field = ModelField("vision", "http://127.0.0.1:9")
        field.set_models([], "")
        self.assertFalse(field.combo.model().item(0).isEnabled())
        self.assertEqual(field.model_name(), "")

    def test_roles_filter_by_capability_and_skip_cloud_models(self) -> None:
        roles = models_by_role([
            {"name": "qwen3:8b", "capabilities": ["completion", "thinking"]},
            {"name": "gemma3:4b", "capabilities": ["completion", "vision"]},
            {"name": "nomic-embed-text:latest", "capabilities": ["embedding"]},
            {"name": "gpt-oss:120b-cloud", "capabilities": ["completion"]},
        ])
        self.assertEqual(roles, {"chat": ["gemma3:4b", "qwen3:8b"], "vision": ["gemma3:4b"],
                                 "embedding": ["nomic-embed-text:latest"]})

    def test_header_model_picker_switches_the_agents_model(self) -> None:
        fake = self.fake_server()
        window = self.window(fake.host)
        window._on_ollama_status({"online": True, "running": [], "models": [
            {"name": "qwen3:8b", "capabilities": ["completion"]},
            {"name": "qwen2.5vl:7b", "capabilities": ["completion", "vision"]},
            {"name": "nomic-embed-text:latest", "capabilities": ["embedding"]},
        ]})
        picker = window.model_picker
        self.assertFalse(picker.isEditable())
        self.assertEqual([picker.itemData(i) for i in range(picker.count())], ["", "qwen2.5vl:7b", "qwen3:8b"])
        index = picker.findData("qwen2.5vl:7b")
        picker.setCurrentIndex(index)
        picker.activated.emit(index)
        self.assertEqual(window.current_agent["model"], "qwen2.5vl:7b")
        self.assertEqual(self.library.agent(window.current_agent["id"])["model"], "qwen2.5vl:7b")
        self.assertEqual(window.agent_cards[0].findChild(app_module.QLabel, "agentModel").text(), "qwen2.5vl:7b")
        window.query.setText("What is priority inversion?")
        window.send_message()
        self.wait(window)
        self.assertEqual(fake.requests[-1]["model"], "qwen2.5vl:7b")

    def wait_for_download(self, field: ModelField) -> None:
        deadline = time.time() + 10
        while time.time() < deadline and not field.download_button.isEnabled():
            self.application.processEvents()
            time.sleep(0.01)
        self.application.processEvents()

    def test_download_installs_the_chosen_model(self) -> None:
        fake = self.fake_server()
        field = ModelField("chat", fake.host)
        field.set_models(["qwen3:8b"], "qwen3:4b")
        field.download_button.click()
        self.wait_for_download(field)
        self.assertEqual([r["model"] for r in fake.requests if r["path"] == "/api/pull"], ["qwen3:4b"])
        self.assertIn("qwen3:4b", field.installed)
        self.assertEqual(field.model_name(), "qwen3:4b")
        self.assertTrue(field.download_button.isHidden())

    def test_get_more_menu_downloads_and_selects_the_model(self) -> None:
        fake = self.fake_server()
        field = ModelField("chat", fake.host)
        field.set_models(["qwen3:8b"], "qwen3:8b")
        action = next(a for a in field.more_menu.actions() if a.text().startswith("qwen3:4b"))
        action.trigger()
        deadline = time.time() + 10
        while time.time() < deadline and "qwen3:4b" not in field.installed:
            self.application.processEvents()
            time.sleep(0.01)
        self.assertEqual(field.model_name(), "qwen3:4b")
        self.assertNotIn("NOT INSTALLED", field.combo.currentText())

    def test_failed_download_shows_the_error_and_can_retry(self) -> None:
        field = ModelField("chat", self.fake_server().host)
        field.set_models([], "no-such-model")
        field.download_button.click()
        self.wait_for_download(field)
        self.assertIn("file does not exist", field.status.text())
        self.assertFalse(field.download_button.isHidden())
        self.assertTrue(field.download_button.isEnabled())

    def test_model_settings_keep_saved_names_exact(self) -> None:
        fake = self.fake_server()
        dialog = ModelSettingsDialog({"provider": "ollama", "host": fake.host, "chat_model": "qwen3:8b",
                                      "embedding_model": "nomic-embed-text", "vision_model": ""})
        self.addCleanup(dialog.close)
        values = dialog.values()
        self.assertEqual(values["embedding_model"], "nomic-embed-text")  # not ":latest", which would re-index
        self.assertEqual(values["vision_model"], "")
        self.assertTrue(dialog.embed_field.download_button.isHidden())
        self.assertLess(dialog.vision_field.combo.findData("qwen3:8b"), 0)  # can't read images
        self.assertLess(dialog.chat_field.combo.findData("nomic-embed-text:latest"), 0)
        self.assertEqual(dialog.embed_field.combo.count(), 1)
        dialog.vision_field.combo.setCurrentIndex(dialog.vision_field.combo.findData("qwen2.5vl:7b"))
        dialog.refresh_models()
        self.assertEqual(dialog.values()["vision_model"], "qwen2.5vl:7b")

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

    def test_lecture_and_course_keep_separate_conversations(self) -> None:
        window, fake = self.gemini_window()
        self.select_lecture(window)
        self.ask(window, "What is priority inversion in this lecture?")
        agent_id = window.current_agent["id"]
        saved = self.library.class_chat_history("CS 354", agent_id=agent_id, lecture_name="Scheduling")
        self.assertEqual([h["question"] for h in saved], ["What is priority inversion in this lecture?"])
        window.navigator.setCurrentItem(window.navigator.topLevelItem(1))  # the whole course
        texts = [m.text for m in window.message_widgets]
        self.assertNotIn("What is priority inversion in this lecture?", texts)  # not merged in
        self.ask(window, "Summarise the course")
        window.navigator.setCurrentItem(window.navigator.topLevelItem(0))  # all materials
        self.ask(window, "What's in my library?")
        # A restart restores each conversation under its own selection.
        reopened = MainWindow(self.library, settings=self.settings, start_monitor=False)
        self.addCleanup(reopened.close)
        reopened.navigator.setCurrentItem(reopened.navigator.topLevelItem(1).child(0))
        texts = [m.text for m in reopened.message_widgets]
        self.assertIn("What is priority inversion in this lecture?", texts)
        self.assertNotIn("Summarise the course", texts)
        reopened.navigator.setCurrentItem(reopened.navigator.topLevelItem(1))
        texts = [m.text for m in reopened.message_widgets]
        self.assertIn("Summarise the course", texts)
        self.assertNotIn("What is priority inversion in this lecture?", texts)
        reopened.navigator.setCurrentItem(reopened.navigator.topLevelItem(0))
        self.assertIn("What's in my library?", [m.text for m in reopened.message_widgets])

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


    def _png(self) -> bytes:
        image = app_module.QImage(2400, 1500, app_module.QImage.Format.Format_RGB32)
        image.fill(0x202020)
        buffer = app_module.QBuffer()
        buffer.open(app_module.QIODevice.OpenModeFlag.WriteOnly)
        image.save(buffer, "PNG")
        return bytes(buffer.data())

    def wait_for_screen(self, window: MainWindow) -> None:
        deadline = time.time() + 20
        self.application.processEvents()
        while (window.screen_task is not None or window.overlay.busy) and time.time() < deadline:
            self.application.processEvents()
            time.sleep(0.01)
        self.application.processEvents()

    def test_overlay_reads_the_screen_with_the_vision_model(self) -> None:
        fake = self.fake_server()
        window = self.window(fake.host)
        window.toggle_overlay()
        self.assertTrue(window.overlay.isVisible())
        self.assertTrue(window.overlay.testAttribute(app_module.Qt.WidgetAttribute.WA_TranslucentBackground))
        captured = []

        def capture(x, y, w, h):
            captured.append(window.overlay.isVisible())  # the overlay hides itself first
            return self._png()

        with (
            patch.object(app_module, "screen_capture_allowed", return_value=True),
            patch.object(app_module, "capture_region", side_effect=capture),
        ):
            window.overlay.input.setText("How does priority inversion apply here?")
            window.overlay.read_button.click()
            self.wait_for_screen(window)
        self.assertEqual(captured, [False])
        self.assertTrue(window.overlay.isVisible())
        user, answer = window.overlay.messages[-2:]
        self.assertEqual(user.text, "How does priority inversion apply here?")
        self.assertIn("Priority inversion", answer.text)
        request = fake.requests[-1]
        self.assertEqual(request["model"], "qwen2.5vl:7b")
        self.assertTrue(request["messages"][-1]["images"])
        sent = app_module.QImage.fromData(__import__("base64").b64decode(request["messages"][-1]["images"][0]))
        self.assertLessEqual(max(sent.width(), sent.height()), 1600)  # Retina captures are downscaled
        self.assertNotIn("lecture-note excerpts", request["messages"][-1]["content"].casefold())  # screen only
        self.assertNotIn("scheduling and priority inversion", str(request["messages"]).casefold())
        # follow-ups carry the overlay's own history
        with (
            patch.object(app_module, "screen_capture_allowed", return_value=True),
            patch.object(app_module, "capture_region", return_value=self._png()),
        ):
            window.overlay.read_button.click()
            self.wait_for_screen(window)
        self.assertEqual(window.overlay.messages[-2].text, "◉ Solve what's on my screen")
        roles = [m["role"] for m in fake.requests[-1]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])
        window.overlay.clear()
        self.assertEqual(window.overlay.messages, [])

    def test_overlay_asks_for_screen_recording_permission(self) -> None:
        window = self.window()
        with (
            patch.object(app_module, "screen_capture_allowed", return_value=False),
            patch.object(app_module, "request_screen_capture") as request,
            patch.object(app_module.QDesktopServices, "openUrl") as open_url,
        ):
            window.read_screen("")
        request.assert_called_once()
        open_url.assert_called_once()
        self.assertIn("Screen Recording", window.overlay.messages[-1].text)
        self.assertIsNone(window.screen_task)

    def test_overlay_reports_capture_failures(self) -> None:
        window = self.window()
        with (
            patch.object(app_module, "screen_capture_allowed", return_value=True),
            patch.object(app_module, "capture_region", side_effect=app_module.ScreenCaptureError("denied")),
        ):
            window.read_screen("")
            self.wait_for_screen(window)
        self.assertIn("denied", window.overlay.messages[-1].text)
        self.assertFalse(window.overlay.busy)
        self.assertTrue(window.overlay.read_button.isEnabled())

    def test_overlay_without_a_vision_model_explains_what_to_install(self) -> None:
        fake = self.fake_server()
        self.settings.setValue("models/vision", "")
        window = self.window(fake.host)
        with (
            patch.object(app_module, "screen_capture_allowed", return_value=True),
            patch.object(app_module, "capture_region", return_value=self._png()),
        ):
            window.read_screen("")
            self.wait_for_screen(window)
        self.assertIn("ollama pull", window.overlay.messages[-1].text)
        self.assertEqual(window.overlay.messages[-1].kind, "error")


    # -- overlay: shortcuts, modes, area picking, stop, persistence ---------------------------
    def test_shortcuts_parse_label_and_reject_plain_keys(self) -> None:
        shortcut = app_module.Shortcut.parse("cmd+shift+space")
        self.assertEqual(shortcut.label, "⇧⌘Space")
        self.assertEqual(app_module.Shortcut.parse("Command+Opt+\\").text, "cmd+option+\\")
        self.assertIsNone(app_module.Shortcut.parse("off"))
        with self.assertRaises(ValueError):
            app_module.Shortcut.parse("cmd+banana")
        with self.assertRaises(ValueError):
            app_module.Shortcut("k", frozenset({"shift"})).validate()  # would block typing a capital K
        self.assertEqual(app_module.shortcut_label("cmd+shift+return"), "⇧⌘↩")

    def test_recorded_keys_become_mac_shortcuts(self) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeySequence

        from study_assistant.dialogs import shortcut_from_sequence

        cmd, ctrl = Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.MetaModifier  # Qt's macOS mapping
        shift, opt = Qt.KeyboardModifier.ShiftModifier, Qt.KeyboardModifier.AltModifier
        self.assertEqual(shortcut_from_sequence(QKeySequence(cmd | shift | Qt.Key.Key_Space), mac=True).text,
                         "cmd+shift+space")
        self.assertEqual(shortcut_from_sequence(QKeySequence(cmd | shift | Qt.Key.Key_Question), mac=True).text,
                         "cmd+shift+/")  # ⇧/ reports "?" but binds the / key
        self.assertEqual(shortcut_from_sequence(QKeySequence(ctrl | opt | Qt.Key.Key_Return), mac=True).text,
                         "control+option+return")
        self.assertIsNone(shortcut_from_sequence(QKeySequence()))

    def test_shortcuts_are_configurable_and_conflicts_are_reported(self) -> None:
        window = self.window()
        self.assertEqual(window.shortcut_settings(), {"toggle": "cmd+shift+space", "read": "cmd+shift+return"})
        self.assertIn("⇧⌘Space", window.overlay.hint.text())

        class FakeHotkeys:
            def __init__(self):
                self.registered = []

            def clear(self):
                self.registered.clear()

            def register(self, shortcut, action):
                self.registered.append(shortcut.text)
                return shortcut.text != "cmd+shift+return"  # pretend another app owns this one

            def close(self):
                pass

        window.hotkeys = FakeHotkeys()
        with patch.object(app_module.sys, "platform", "darwin"):
            window.apply_shortcuts()
        self.assertEqual(window.failed_shortcuts, ["⇧⌘↩"])
        dialog = app_module.ShortcutsDialog(window.shortcut_settings(), window.failed_shortcuts)
        self.addCleanup(dialog.close)
        dialog.toggle_field.set_value("option+k")
        dialog.read_field.set_value("off")
        with (
            patch.object(app_module, "ShortcutsDialog", return_value=dialog),
            patch.object(dialog, "exec", return_value=app_module.QDialog.DialogCode.Accepted),
            patch.object(app_module.sys, "platform", "darwin"),
        ):
            window.configure_shortcuts()
        self.assertEqual(window.shortcut_settings(), {"toggle": "option+k", "read": "off"})
        self.assertEqual(window.hotkeys.registered, ["option+k"])
        self.assertEqual(window.failed_shortcuts, [])
        self.assertIn("⌥K", window.overlay.hint.text())
        self.assertNotIn("reads the screen straight away", window.overlay.hint.text())

    def test_hint_mode_asks_for_a_hint_not_a_solution(self) -> None:
        fake = self.fake_server()
        window = self.window(fake.host)
        window.toggle_overlay()
        window.overlay.mode_buttons["hint"].setChecked(True)
        with (
            patch.object(app_module, "screen_capture_allowed", return_value=True),
            patch.object(app_module, "capture_region", return_value=self._png()),
        ):
            window.overlay.read_button.click()
            self.wait_for_screen(window)
        self.assertIn("without the full solution", fake.requests[-1]["messages"][-1]["content"])
        self.assertEqual(window.overlay.messages[0].text, "◉ Hint what's on my screen")
        self.assertEqual(self.settings.value("overlay/mode"), "hint")

    def test_area_picker_reads_only_the_chosen_region_and_esc_cancels(self) -> None:
        fake = self.fake_server()
        window = self.window(fake.host)
        window.toggle_overlay()
        with (
            patch.object(app_module, "screen_capture_allowed", return_value=True),
            patch.object(app_module, "capture_interactive", return_value=None),
            patch.object(app_module, "capture_region") as full_screen,
        ):
            window.overlay.area_button.click()
            self.wait_for_screen(window)
        full_screen.assert_not_called()
        self.assertEqual(window.overlay.messages, [])  # Esc: nothing asked
        self.assertTrue(window.overlay.isVisible())
        self.assertFalse(window.overlay.busy)
        with (
            patch.object(app_module, "screen_capture_allowed", return_value=True),
            patch.object(app_module, "capture_interactive", return_value=self._png()),
        ):
            window.overlay.area_button.click()
            self.wait_for_screen(window)
        self.assertIn("Priority inversion", window.overlay.messages[-1].text)
        self.assertTrue(fake.requests[-1]["messages"][-1]["images"])

    def test_stop_ends_an_overlay_answer_and_keeps_what_was_written(self) -> None:
        fake = self.fake_server(token_delay=0.05)
        window = self.window(fake.host)
        window.toggle_overlay()
        with (
            patch.object(app_module, "screen_capture_allowed", return_value=True),
            patch.object(app_module, "capture_region", return_value=self._png()),
        ):
            window.read_screen("", "solve", False)
            deadline = time.time() + 10
            while time.time() < deadline and not (window.overlay.messages
                                                  and window.overlay.messages[-1].text.strip()):
                self.application.processEvents()
                time.sleep(0.01)
            self.assertEqual(window.overlay.read_button.text(), "■ STOP")
            started = time.time()
            window.overlay.read_button.click()
            self.wait_for_screen(window)
        self.assertLess(time.time() - started, 2)
        answer = window.overlay.messages[-1]
        self.assertTrue(answer.text.endswith("*Stopped.*"))
        self.assertNotIn("[S9]", answer.text)
        self.assertEqual(window.overlay.read_button.text(), "◉ READ SCREEN")

    def test_stop_ends_a_chat_answer(self) -> None:
        fake = self.fake_server(token_delay=0.05)
        window = self.window(fake.host)
        window.query.setText("What is priority inversion?")
        window.send_message()
        deadline = time.time() + 10
        while time.time() < deadline and not window.message_widgets[-1].text.strip():
            self.application.processEvents()
            time.sleep(0.01)
        self.assertEqual(window.send_button.text(), "■ STOP")
        self.assertTrue(window.send_button.isEnabled())
        window.send_message()
        self.wait(window)
        self.assertTrue(window.message_widgets[-1].text.endswith("*Stopped.*"))
        self.assertEqual(window.send_button.text(), "SEND ▸")

    def test_overlay_remembers_position_size_and_mode(self) -> None:
        window = self.window()
        window.toggle_overlay()
        window.overlay.setGeometry(120, 140, 500, 600)
        window.overlay.mode_buttons["explain"].setChecked(True)
        window.overlay.hide()
        reopened = app_module.ScreenOverlay(settings=self.settings)
        self.addCleanup(reopened.close)
        self.assertEqual((reopened.width(), reopened.height()), (500, 600))
        self.assertEqual(reopened.mode(), "explain")

    def test_overlay_has_no_title_or_transparency_slider(self) -> None:
        from PySide6.QtWidgets import QSlider

        overlay = app_module.ScreenOverlay()
        self.addCleanup(overlay.close)
        overlay.set_agent("OS Notes")
        self.assertFalse(overlay.findChildren(QSlider))
        self.assertFalse([label for label in overlay.findChildren(app_module.QLabel) if "SCREEN LINK" in label.text()])
        self.assertAlmostEqual(overlay._panel_alpha, 0.82)

    def test_escape_hides_the_overlay(self) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtTest import QTest

        window = self.window()
        window.toggle_overlay()
        QTest.keyClick(window.overlay, Qt.Key.Key_Escape)
        self.assertFalse(window.overlay.isVisible())


    def test_offline_ollama_is_not_reported_as_missing_models(self) -> None:
        dialog = app_module.ModelSettingsDialog({"provider": "ollama", "host": "http://127.0.0.1:9", "chat_model": "qwen3:8b",
                                                 "embedding_model": "nomic-embed-text", "vision_model": ""})
        self.addCleanup(dialog.close)
        self.assertIn("OLLAMA OFFLINE", dialog.chat_field.combo.currentText())
        self.assertTrue(dialog.chat_field.download_button.isHidden())
        self.assertEqual(dialog.values()["chat_model"], "qwen3:8b")


    # -- Gemini API (the default provider) ----------------------------------------------------
    def gemini_window(self, **kwargs) -> tuple[MainWindow, object]:
        from fake_gemini import KEY, MODELS, FakeGemini

        default = {"name": "models/gemini-3.5-flash", "supportedGenerationMethods": ["generateContent"], "thinking": True}
        kwargs.setdefault("models", [default, *MODELS])

        fake = FakeGemini(**kwargs).__enter__()
        self.addCleanup(fake.__exit__)
        from study_assistant import gemini

        gemini._STICKY.clear()
        self.addCleanup(gemini._STICKY.clear)
        gemini.set_health({})
        self.addCleanup(gemini.set_health, {})
        environment = patch.dict(os.environ, {"GEMINI_API_KEY": KEY})
        environment.start()
        self.addCleanup(environment.stop)
        self.settings.remove("models/provider")  # the default: Gemini
        self.settings.setValue("gemini/api_root", fake.api_root)
        return self.window(), fake

    def test_gemini_is_the_default_and_answers_with_citations(self) -> None:
        window, fake = self.gemini_window()
        self.assertEqual(window.provider, "gemini")
        self.assertEqual(window.footer_label.text(), "◆ CLOUD · GEMINI API")
        self.assertEqual(window.stat_runtime.text(), "CLOUD")
        window.navigator.setCurrentItem(window.navigator.topLevelItem(1))
        window.query.setText("What is priority inversion?")
        window.send_message()
        self.wait(window)
        answer = window.message_widgets[-1]
        self.assertIn("[1]", answer.text)
        self.assertNotIn("S7", answer.text)
        chats = [r for r in fake.requests if r["path"].endswith(":streamGenerateContent")]
        self.assertEqual(chats[-1]["path"], "/v1beta/models/gemini-3.5-flash:streamGenerateContent")

    def test_agents_saved_with_a_local_model_fall_back_to_the_gemini_default(self) -> None:
        tutor = self.library.agents()[0]
        self.library.save_agent({**tutor, "model": "qwen3:8b"})
        window, _fake = self.gemini_window()
        self.assertEqual(window.agent_model(), "gemini-3.5-flash")
        self.assertEqual(window.model_picker.currentData(), "")
        self.settings.setValue("models/provider", "ollama")
        self.assertEqual(window.agent_model(), "qwen3:8b")  # still there when switching back

    def test_gemini_status_lists_models_and_settles_unavailable_defaults(self) -> None:
        from fake_gemini import MODELS

        window, fake = self.gemini_window(models=[m for m in MODELS if "latest" not in m["name"]])  # no 3.5 either
        client = app_module.make_client(window.model_settings())
        models = client.list_models()
        window._on_ollama_status({"online": True, "models": models, "running": []})
        self.assertEqual(window.chat_models, ["gemini-2.0-flash"])
        self.assertEqual(window.model_settings()["chat_model"], "gemini-2.0-flash")  # alias not offered here
        self.assertIn("GEMINI ONLINE · 1 MODEL", window.online_label.text())

    def test_missing_gemini_key_points_to_the_models_window(self) -> None:
        window, _fake = self.gemini_window()
        with patch.dict(os.environ, {}, clear=True), patch.object(app_module, "get_api_key", return_value=""):
            window._on_ollama_status({"online": False, "models": [], "running": []})
            self.assertIn("ADD A GEMINI API KEY", window.online_label.text())

    def test_gemini_indexing_skips_handwriting_by_default_and_batches_embeddings(self) -> None:
        window, fake = self.gemini_window()
        window.index_library()
        self.wait(window)
        self.assertIn("3 indexed for search", window.library_status.toolTip())
        paths = [r["path"] for r in fake.requests if r["method"] == "POST"]
        self.assertEqual(paths, ["/v1beta/models/gemini-embedding-001:batchEmbedContents"])  # one request
        results = self.library.search("priority", query_embedding=window._models().embed_query("priority"),
                                      model="gemini-embedding-001")
        self.assertTrue(results)

    def test_models_window_switches_between_gemini_and_ollama(self) -> None:
        window, _fake = self.gemini_window()
        ollama = self.fake_server()
        self.settings.setValue("models/host", ollama.host)
        dialog = app_module.ModelSettingsDialog(window.model_settings(), window, settings=self.settings)
        self.addCleanup(dialog.close)
        self.assertFalse(dialog.key_row.isHidden())
        self.assertTrue(dialog.host_row.isHidden())
        self.assertTrue(dialog.chat_field.more_button.isHidden())  # nothing to download from Google
        self.assertFalse(dialog.handwriting.isChecked())
        self.assertGreaterEqual(dialog.vision_field.combo.findData("gemini-flash-latest"), 0)
        dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("ollama"))
        self.assertTrue(dialog.key_row.isHidden())
        self.assertFalse(dialog.chat_field.more_button.isHidden())
        self.assertGreaterEqual(dialog.chat_field.combo.findData("qwen3:8b"), 0)
        with (
            patch.object(app_module, "ModelSettingsDialog", return_value=dialog),
            patch.object(dialog, "exec", return_value=app_module.QDialog.DialogCode.Accepted),
        ):
            window.configure_models()
        self.assertEqual(window.provider, "ollama")
        self.assertEqual(window.footer_label.text(), "◆ 100% LOCAL · NO CLOUD")
        self.assertEqual(self.settings.value("gemini/chat"), None)  # the Gemini picks were left alone

    def test_new_api_key_is_saved_when_the_models_window_is_saved(self) -> None:
        window, _fake = self.gemini_window()
        dialog = app_module.ModelSettingsDialog(window.model_settings(), window, settings=self.settings)
        self.addCleanup(dialog.close)
        dialog.key_input.setText("new-key-abcdefghijkl")
        with (
            patch.object(app_module, "ModelSettingsDialog", return_value=dialog),
            patch.object(dialog, "exec", return_value=app_module.QDialog.DialogCode.Accepted),
            patch.object(app_module, "save_api_key", return_value=True) as save,
        ):
            window.configure_models()
        save.assert_called_once_with("new-key-abcdefghijkl")
        self.assertIsNone(self.settings.value("api_key"))  # never stored in plain settings


    def test_chat_shows_what_it_is_waiting_on_while_gemini_retries(self) -> None:
        window, fake = self.gemini_window()
        window.select_agent(self.library.agents()[2]["id"])
        fake.fail_next.append((503, {"error": {"code": 503, "status": "UNAVAILABLE", "message": "high demand"}}))
        window.query.setText("hello")
        window.send_message()
        message = window.message_widgets[-1]
        seen = set()
        deadline = time.time() + 15
        while window.active_task and time.time() < deadline:
            self.application.processEvents()
            seen.add(message.body.text())
            time.sleep(0.02)
        self.wait(window)
        self.assertTrue(any("Reading your notes" in text for text in seen))
        self.assertTrue(any("gemini-3.5-flash is busy · retry 1 of 2 in" in text for text in seen))
        self.assertIn("Priority inversion", message.text)  # then the answer arrived


    def test_models_window_tests_models_marks_them_and_uses_the_recommendation(self) -> None:
        window, fake = self.gemini_window()
        fake.busy_models.add("gemini-3.5-flash")
        fake.missing_models.add("gemini-flash-latest")
        dialog = app_module.ModelSettingsDialog(window.model_settings(), window, settings=self.settings)
        self.addCleanup(dialog.close)
        self.assertFalse(dialog.test_row.isHidden())
        dialog.test_button.click()
        deadline = time.time() + 20
        while dialog._tester is not None and time.time() < deadline:
            self.application.processEvents()
            time.sleep(0.02)
        self.application.processEvents()
        self.assertIn("recommended: gemini-2.0-flash", dialog.test_status.text())
        texts = [dialog.chat_field.combo.itemText(i) for i in range(dialog.chat_field.combo.count())]
        self.assertTrue(any(t.startswith("gemini-2.0-flash  ·  ✓") for t in texts))
        self.assertTrue(any(t.startswith("gemini-3.5-flash  ·  ✗ overloaded") for t in texts))
        self.assertLess(dialog.chat_field.combo.findData("gemini-flash-latest"), 0)  # key can't use it: hidden
        dialog.use_best_button.click()
        self.assertEqual(dialog.values()["chat_model"], "gemini-2.0-flash")
        self.assertEqual(dialog.values()["vision_model"], "gemini-2.0-flash")
        self.assertIn("gemini-flash-latest", self.settings.value("gemini/health"))  # shared with the tool


    # -- answering from the selected lecture, and explaining empty answers ---------------------
    def select_lecture(self, window: MainWindow) -> None:
        window.navigator.setCurrentItem(window.navigator.topLevelItem(1).child(0))

    def ask(self, window: MainWindow, question: str) -> str:
        window.query.setText(question)
        window.send_message()
        self.wait(window)
        return window.message_widgets[-1].text

    def test_selected_lecture_is_read_in_full_for_overview_questions(self) -> None:
        window, fake = self.gemini_window()
        self.select_lecture(window)
        answer = self.ask(window, "Explain to me what was taught in this lecture")
        self.assertIn("Priority inversion", answer)
        self.assertEqual(window.message_widgets[-1].retrieval_label.text(), "◆ READ ALL 2 PAGES OF SCHEDULING")
        chat = [r for r in fake.requests if r["path"].endswith(":streamGenerateContent")][-1]
        prompt = chat["systemInstruction"]["parts"][0]["text"]  # the slides lead the request (cache-friendly)
        self.assertIn("[S1]", prompt)
        self.assertIn("[S2]", prompt)  # every readable page, not a keyword search that found nothing
        self.assertNotIn("[S3] CS 354", prompt)  # the blank page isn't sent
        self.assertNotIn(":batchEmbedContents", " ".join(r["path"] for r in fake.requests))  # no search needed

    def test_empty_library_says_so(self) -> None:
        empty = Library(Path(self.temp_dir.name) / "empty.sqlite3")
        self.addCleanup(empty.close)
        window = MainWindow(empty, settings=self.settings, start_monitor=False)
        window._auto_indexed = True
        self.addCleanup(window.close)
        self.assertEqual(window.library_status.text(), "NO PDFS YET · ADD A LECTURE TO START")
        self.assertIn("Your library is empty", self.ask(window, "What is a semaphore?"))

    def test_selected_course_reads_every_lecture_in_it(self) -> None:
        path = Path(self.temp_dir.name) / "semaphores.pdf"
        document = pymupdf.open()
        document.new_page().insert_text((72, 72), "Semaphores: wait and signal.")
        document.save(path)
        document.close()
        self.library.import_pdf("CS 354", "Semaphores", path, extract_page_texts(path))
        window, fake = self.gemini_window()
        window.navigator.setCurrentItem(window.navigator.topLevelItem(1))  # the course folder
        self.ask(window, "Summarize the course so far")
        self.assertEqual(window.message_widgets[-1].retrieval_label.text(), "◆ READ ALL 3 PAGES OF CS 354 · 2 LECTURES")
        prompt = [r for r in fake.requests if r["path"].endswith(":streamGenerateContent")][-1]["systemInstruction"]["parts"][0]["text"]
        self.assertLess(prompt.index("priority inversion"), prompt.index("Semaphores: wait"))  # teaching order

    def test_courses_too_big_to_send_whole_use_the_most_relevant_pages(self) -> None:
        window, fake = self.gemini_window()
        window.SCOPE_CHARS = {"gemini": 150}  # pretend only one page fits
        window.navigator.setCurrentItem(window.navigator.topLevelItem(1))
        self.ask(window, "mutexes")
        self.assertTrue(window.message_widgets[-1].retrieval_label.text().startswith(
            "◆ READ 1 MOST RELEVANT OF 2 PAGES IN CS 354"))
        prompt = [r for r in fake.requests if r["path"].endswith(":streamGenerateContent")][-1]["systemInstruction"]["parts"][0]["text"]
        self.assertIn("mutexes", prompt)  # the page that matches, not just the first one
        self.assertNotIn("priority inversion", prompt)

    def test_scanned_pdf_is_explained(self) -> None:
        path = Path(self.temp_dir.name) / "scan.pdf"
        document = pymupdf.open()
        document.new_page()
        document.new_page()
        document.save(path)
        document.close()
        self.library.import_pdf("MA 416", "Week 1", path, extract_page_texts(path))
        window = self.window()
        items = {window.navigator.topLevelItem(i).text(0): window.navigator.topLevelItem(i)
                 for i in range(1, window.navigator.topLevelItemCount())}
        window.navigator.setCurrentItem(items["MA 416"].child(0))
        self.assertIn("none of them has selectable text", self.ask(window, "Summarize this lecture"))

    def test_pages_that_do_not_cover_the_question_are_counted(self) -> None:
        window, fake = self.gemini_window(answer="NOT_IN_NOTES")
        self.select_lecture(window)
        answer = self.ask(window, "What did the lecture say about TCP congestion control?")
        self.assertIn("I read 2 pages from **CS 354 · Scheduling**, but they don't cover that", answer)

    def test_sidebar_shows_what_is_saved_and_what_needs_indexing(self) -> None:
        window = self.window()
        self.assertTrue(window.library_status.text().startswith(
            "CONTEXT: WHOLE LIBRARY\n3 PAGES\nALL SENT WITH EACH QUESTION"))
        self.assertEqual(window.index_button.text(), "⟳ INDEX 3 NEW PAGES")
        self.select_lecture(window)  # the summary follows the selection
        self.assertTrue(window.library_status.text().startswith("CONTEXT: SCHEDULING\n3 PAGES"))

    def test_answer_shows_first_source_and_folds_the_rest(self) -> None:
        pages = [{"id": i, "class_name": "CS 354", "lecture_name": "Scheduling", "page_number": i,
                  "file_path": "/x.pdf"} for i in range(1, 6)]
        answer = ChatMessageWidget("assistant", "Inversion [1][3].", pages)
        self.assertFalse(answer.source_buttons[0].isHidden())  # the first source always shows
        self.assertTrue(answer.sources_panel.isHidden())
        self.assertEqual(answer.sources_toggle.text(), "▸  4 MORE SOURCES")
        answer.sources_toggle.click()
        self.assertFalse(answer.sources_panel.isHidden())
        self.assertTrue(answer.sources_toggle.text().startswith("▾"))
        single = ChatMessageWidget("assistant", "One [1].", pages[:1])
        self.assertTrue(single.sources_toggle.isHidden())  # nothing to fold

    def test_sidebar_mentions_search_only_for_selections_too_big_to_send(self) -> None:
        window = self.window()
        window.SCOPE_CHARS_DEFAULT = 100  # pretend the model can only take a sliver of the notes
        window.update_library_status()
        self.assertIn("TOO BIG TO SEND WHOLE", window.library_status.text())
        self.assertIn("SEARCH READY FOR 0/3", window.library_status.text())

    def test_new_pages_are_indexed_automatically_and_questions_wait_for_it(self) -> None:
        window, fake = self.gemini_window()
        window._auto_indexed = False
        fake.token_delay = 0.0
        window._on_ollama_status({"online": True, "running": [],
                                  "models": app_module.make_client(window.model_settings()).list_models()})
        self.assertTrue(window._auto_index_running)
        self.assertTrue(window.query.isEnabled())  # you can keep typing
        window.query.setText("What is priority inversion?")
        window.send_message()  # queued until indexing finishes
        deadline = time.time() + 15
        while time.time() < deadline and not any(r["path"].endswith(":streamGenerateContent") for r in fake.requests):
            self.application.processEvents()
            time.sleep(0.02)
        self.wait(window)
        self.assertIn("3 indexed for search", window.library_status.toolTip())
        self.assertIn("Priority inversion", window.message_widgets[-1].text)
        window._on_ollama_status({"online": True, "running": [], "models": []})
        self.assertIsNone(window.active_task)  # once per launch

    def test_follow_up_questions_search_with_the_previous_question(self) -> None:
        window, fake = self.gemini_window()
        window.SCOPE_CHARS = {"gemini": 100}  # too big to send whole, so it searches
        window.index_library()
        self.wait(window)
        window.navigator.setCurrentItem(window.navigator.topLevelItem(1))
        self.ask(window, "What is priority inversion?")
        self.ask(window, "And how is it fixed?")
        query = [r for r in fake.requests if r["path"].endswith(":batchEmbedContents")][-1]
        self.assertEqual(query["requests"][0]["content"]["parts"][0]["text"],
                         "What is priority inversion? And how is it fixed?")


    def test_follow_ups_reuse_the_same_opening_so_gemini_can_cache_it(self) -> None:
        window, fake = self.gemini_window()
        self.select_lecture(window)
        self.ask(window, "What is priority inversion?")
        self.assertNotIn("CACHED", window.stat_context.text())  # nothing to reuse on the first question
        self.ask(window, "And how is it fixed?")
        chats = [r for r in fake.requests if r["path"].endswith(":streamGenerateContent")]
        self.assertEqual(chats[-1]["systemInstruction"], chats[-2]["systemInstruction"])  # identical opening
        self.assertIn("Scheduling and priority inversion", chats[-1]["systemInstruction"]["parts"][0]["text"])
        self.assertEqual([turn["role"] for turn in chats[-1]["contents"]], ["user", "model", "user"])
        self.assertEqual(chats[-1]["contents"][-1]["parts"][-1]["text"], "Question: And how is it fixed?")
        self.assertIn("89% CACHED", window.stat_context.text())  # 800 of 900 prompt tokens reused


if __name__ == "__main__":
    unittest.main()
