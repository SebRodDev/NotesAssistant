"""Check which Gemini models are answering right now, and how fast.

    python tools/check_gemini.py              # probe every chat model your key can use
    python tools/check_gemini.py --use-best   # ...and make the recommended one Cortex's default

Each model gets one tiny request ("reply with OK") with no retries, all at the same time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study_assistant.gemini import GeminiClient, check_models, get_api_key, recommend  # noqa: E402
from study_assistant.hardware import fetch_models_by_role  # noqa: E402
from study_assistant.ollama import OllamaError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--use-best", action="store_true", help="make the recommended model Cortex's default")
    parser.add_argument("--timeout", type=float, default=20.0, help="seconds to wait for each model (default 20)")
    args = parser.parse_args()

    if not get_api_key():
        print("No Gemini API key found. Add one in Cortex under ⬡ MODELS (or set GEMINI_API_KEY).")
        return 1
    from PySide6.QtCore import QSettings

    from study_assistant.providers import load_health, save_health

    settings = QSettings("Purdue", "StudyAssistant")
    load_health(settings)
    client = GeminiClient()
    client.include_unavailable = True  # re-test models an earlier check found unusable
    try:
        models = fetch_models_by_role(client, timeout=15)["chat"]
    except OllamaError as error:
        print(f"Couldn't list models: {error}")
        return 1
    models = [m for m in models if m.startswith("gemini")]
    print(f"Probing {len(models)} models (up to {args.timeout:.0f}s each, in parallel)…\n")
    results = check_models(models, timeout=args.timeout)
    save_health(settings, results)  # Cortex reads these: hides unusable models, prefers working backups
    settings.sync()

    working = sorted((n for n in results if results[n]["kind"] == "ok"), key=lambda n: results[n]["seconds"])
    failing = [n for n in results if results[n]["kind"] != "ok"]
    for name in working:
        print(f"  ✓ {name:<38} {results[name]['seconds']:5.1f}s")
    for name in failing:
        print(f"  ✗ {name:<38} {results[name]['seconds']:5.1f}s  {results[name]['error'][:70]}")

    best = recommend(results)
    if best is None:
        print("\nNo model answered. Google may be having a wider outage, or the key has hit its daily limit.")
        return 1
    fastest = working[0]
    print(f"\nRecommended: {best}" + (f"  (fastest overall: {fastest})" if fastest != best else ""))
    if args.use_best:
        settings.setValue("gemini/chat", best)
        settings.setValue("gemini/vision", best)
        settings.sync()
        print("Set as Cortex's default chat and vision model. Restart Cortex to use it.")
    else:
        print("Run with --use-best to make it Cortex's default, or pick it from the MODEL dropdown in the header.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
