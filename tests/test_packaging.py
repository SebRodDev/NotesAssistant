import unittest
from unittest.mock import patch

import build_macos_app


class MacPackagingTests(unittest.TestCase):
    def test_build_arguments_include_desktop_runtime_and_keychain_support(self) -> None:
        arguments = build_macos_app.build_arguments()
        self.assertEqual(arguments[0], str(build_macos_app.ROOT / "macos_launcher.py"))
        self.assertIn("--windowed", arguments)
        self.assertIn("--osx-bundle-identifier", arguments)
        self.assertIn("edu.purdue.studyassistant", arguments)
        self.assertIn("pymupdf", arguments)
        self.assertIn("google.genai", arguments)
        self.assertIn("keyring.backends.macOS", arguments)

    def test_build_script_explains_that_it_only_runs_on_macos(self) -> None:
        with patch.object(build_macos_app.sys, "platform", "linux"):
            with self.assertRaisesRegex(SystemExit, "Build the macOS application on macOS"):
                build_macos_app.main()


if __name__ == "__main__":
    unittest.main()
