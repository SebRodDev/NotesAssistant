"""Build a standalone macOS application bundle with PyInstaller."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
APP_NAME = "Cortex"
APP_IDENTIFIER = "edu.purdue.studyassistant"


def build_arguments() -> list[str]:
    """Return the reproducible PyInstaller options used for the Finder app."""
    return [
        str(ROOT / "macos_launcher.py"),
        "--noconfirm",
        "--windowed",
        "--name",
        APP_NAME,
        "--osx-bundle-identifier",
        APP_IDENTIFIER,
        "--distpath",
        str(ROOT / "dist"),
        "--workpath",
        str(ROOT / "build" / "pyinstaller"),
        "--specpath",
        str(ROOT / "build"),
        "--collect-all",
        "pymupdf",
    ]


def main() -> int:
    if sys.platform != "darwin":
        raise SystemExit("Build the macOS application on macOS with the Python environment you intend to bundle.")
    os.environ.setdefault("PYINSTALLER_CONFIG_DIR", str(ROOT / "build" / "pyinstaller-config"))
    try:
        import PyInstaller.__main__
    except ImportError as error:
        raise SystemExit(
            "PyInstaller is not installed in this environment. Run: "
            "python -m pip install -e '.[macos]'"
        ) from error

    PyInstaller.__main__.run(build_arguments())
    bundle = ROOT / "dist" / f"{APP_NAME}.app"
    if not bundle.is_dir():
        raise SystemExit(f"PyInstaller finished without creating the expected app bundle: {bundle}")
    print(f"Built {bundle}")
    print("Open it in Finder, or copy it to /Applications to keep it in your Applications folder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
