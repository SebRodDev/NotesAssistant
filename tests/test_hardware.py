import unittest

from study_assistant.hardware import GIB, Hardware, download_choices


class DownloadChoicesTests(unittest.TestCase):
    def test_lists_models_that_fit_before_ones_that_do_not(self) -> None:
        choices = download_choices("chat", [], Hardware(8 * GIB))
        fits = [choice.fits for choice in choices]
        self.assertEqual(fits, sorted(fits, reverse=True))
        self.assertIn("qwen3:4b", [choice.name for choice in choices if choice.fits])
        self.assertIn("qwen3.6:35b", [choice.name for choice in choices if not choice.fits])

    def test_skips_installed_models_including_latest_tags(self) -> None:
        names = [c.name for c in download_choices("embedding", ["nomic-embed-text:latest"], Hardware(8 * GIB))]
        self.assertNotIn("nomic-embed-text", names)
        self.assertIn("mxbai-embed-large", names)


if __name__ == "__main__":
    unittest.main()
