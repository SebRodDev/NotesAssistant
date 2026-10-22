# Purdue Study Assistant

A local-first macOS desktop app for searching lecture PDFs. It imports searchable PDFs, stores extracted text page by page, supports keyword and semantic search, and previews the matching source page.

## Run on macOS

### Launch the app

A standalone macOS app is already built at `dist/Purdue Study Assistant.app`. In Finder, open this project folder, then double-click **dist → Purdue Study Assistant.app**. If macOS asks you to confirm opening an app downloaded or built outside the App Store, choose **Open**.

To launch it from Terminal instead, run:

```bash
open "dist/Purdue Study Assistant.app"
```

You can drag **Purdue Study Assistant.app** from `dist` into your **Applications** folder and launch it from there. The app is packaged with its own Python runtime, so you do not need to activate a virtual environment or keep Terminal open. This build is intended for Apple silicon Macs and is not signed or notarized, so macOS security prompts can vary. Your PDFs, library database, and Gemini API key are not bundled with the app. The first time you use AI features, open **AI settings…** in the app to add your Gemini API key.

### Run from source (optional)

Python 3.11 or newer is required. On macOS, use a recent Python install (for example, `python3.12`) rather than the system `python3`, which may be older. From this project folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
study-assistant
```

### Rebuild the standalone app (optional)

Build on the Mac where you intend to run the app:

```bash
source .venv/bin/activate
python -m pip install -e '.[macos]'
python build_macos_app.py
open "dist/Purdue Study Assistant.app"
```

Run the local regression tests with `python -m unittest discover -s tests -v`.

Choose **Add lecture PDFs** in the left sidebar (or press **Command+O**) and enter each PDF's class and lecture labels. The class is prefilled from your current study space, and the lecture name defaults to the PDF filename. Your last selected class or lecture is restored the next time you open the app. Page extraction runs in the background so a long PDF does not freeze the window. The sidebar shows how many pages are imported, AI-indexed, and awaiting visual checks. Keyword search works locally right away. To enable AI answers, visual indexing, meaning-based search, and quizzes, open **AI settings…** and enter a Gemini API key; the app stores it in macOS Keychain. Then choose **Index PDFs for AI search**. During indexing, the app automatically checks page images for handwritten notes and meaningful visual labels, then creates searchable text-and-image embeddings. Page text and rendered page images are sent to Gemini; transcribed notes and embeddings are stored locally. The first indexing run may take longer because it checks every imported page. Later runs only process pages not yet checked or indexed.

Use the **Dark mode** checkbox in the sidebar to switch appearance; your choice is saved between launches. Handwriting is included automatically during indexing, and cited page previews show any saved visual-note transcription for verification.

Use the **Classes and lectures** list in the sidebar to choose a class or lecture as your current study space. Each class or lecture keeps a separate conversation while the app is open, so switching study spaces won't mix their messages. New questions and page searches are restricted to that selection; **All materials** searches the whole library. Class-scoped Q&A is saved and restored between launches (the chat shows recent exchanges; **Question history** opens the longer archive).

Open a cited page in the preview panel to compare the original page with any saved visual-note transcription. Machine readings are clearly labeled and should be verified against the page. Re-importing a PDF removes its old page records and vectors so stale content does not remain searchable.

Use the **Ask from my PDFs** composer mode for a concise answer grounded in indexed pages. The model returns claims with source IDs; the app checks every ID against the retrieved pages and attaches the real class, lecture, file, and PDF page. Unsupported questions produce an explicit not-found answer. Each source appears as a clickable citation; selecting it opens the PDF page preview. Select text in an answer and choose **Explain selected text** for a deeper cited explanation, or use **Copy cited prompt** to put a prompt and evidence excerpt on the clipboard. Choose **Find a page** to search by keywords locally; the optional **Search by meaning** checkbox also uses page embeddings.

Choose **Practice quiz…** to select one or more lectures and generate short-answer, multiple-choice, or mixed practice. Questions use new scenarios rather than copying slide wording. Multiple choice is graded locally; short answers are graded against a source-grounded rubric. **Weak spots** ranks topics by recency-weighted accuracy and can create a targeted review quiz for the weakest topic.

The library database, visual-note readings, and vectors are stored at `~/Library/Application Support/Purdue Study Assistant/study_assistant.sqlite3`. Original PDFs remain where they are; the app stores their paths rather than copying them. Keep those files in place so page previews and visual indexing continue to work.

## First milestone limits

Gemini-backed features require a key and network access. The app uses Gemini Embedding 2 for visual/semantic indexing and Gemini 3.5 Flash-Lite / 3.8 Flash for visual reading, grounded answers, and quiz generation. Google currently lists these models as free on its API free tier, with request quotas that can change. Use a key from a project with billing disabled to honor the no-cost requirement; a key tied to paid billing can incur charges. Google's free-tier terms say submitted content may be used to improve its products. Semantic search sends your query to Gemini; indexing (including automatic visual-note reading), grounded Q&A, and quiz creation/grading also send relevant PDF page text and/or rendered page images. Keyword search and stored vectors remain local.
