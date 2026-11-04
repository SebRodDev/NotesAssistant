# Cortex

My own offline AI chat interface for studying. It runs local models through [Ollama](https://ollama.com), so nothing leaves my Mac, and I can build and configure custom agents that answer questions from my lecture notes, with citations to the exact PDF pages. Clean, cyberpunk-inspired design. Still a work in progress!

## How it "knows" my notes

The model isn't fine-tuned. Cortex uses **retrieval-augmented generation (RAG)**, which works better for course notes:

1. **Import.** Lecture PDFs are split into pages and stored in a local SQLite library (full-text indexed).
2. **Index.** Every page is embedded with a local embedding model (`nomic-embed-text`). If a vision model is installed, handwriting and diagram labels are read from the page images first.
3. **Ask.** Your question is matched against the pages by keywords *and* meaning. The best pages go to the chat model with instructions to answer from them and cite `[S1]`-style IDs.
4. **Verify.** The app checks every citation against the pages it actually retrieved, drops invented ones, and turns the rest into clickable `[1]`, `[2]` links that open the page.

New notes become searchable as soon as you import and index them. Nothing needs retraining, and answers stay traceable to a slide.

## Setup

1. Install Ollama from https://ollama.com/download and open it. It runs in the menu bar.
2. Pull the models (in Terminal):

   ```bash
   ollama pull qwen3:8b            # chat, quizzes, grading
   ollama pull nomic-embed-text    # search by meaning
   ollama pull qwen2.5vl:7b        # optional: reads handwriting on slides
   ```

   On an 8 GB Mac, use `qwen3:4b` and `gemma3:4b` instead and choose them under **⬡ MODELS**.

3. Run from source (Python 3.11+):

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install -e .
   cortex
   ```

To build a double-clickable app: `python -m pip install -e '.[macos]' && python build_macos_app.py`, then open `dist/Cortex.app`.

## Using it

- **Agents** (left sidebar): each agent has its own model, creativity (temperature), notes access, color, and system prompt. Choose **+ NEW** to make one, or **⚙ CONFIG** to edit the active one. It comes with three agents:
  - **Tutor**: notes only, strict citations, says so when your notes don't cover something.
  - **Explainer**: starts from your notes, adds general knowledge marked *(general knowledge)*, and shows its reasoning on thinking models.
  - **Free chat**: a normal local chatbot that doesn't read your notes.
- **Courses** tab: **+ ADD LECTURE PDFS** (⌘O), then **⟳ INDEX FOR SEARCH**. Select a class or lecture to scope the conversation. Each agent keeps a separate saved history per class.
- **Composer**: **ASK** chats with the agent; **FIND PAGE** lists matching pages. Enter sends, Shift+Enter adds a new line, Ctrl+K focuses the composer.
- **Header** shows the live model, creativity, generation speed (tokens/s), where the model runs (GPU/CPU and memory, from `ollama ps`), and how much of the 16k context the last answer used.
- **Neural link panel** (right): the agent's generated portrait and system prompt. Clicking a citation opens the PDF page and its extracted text there.
- **Study** tab: practice quizzes (multiple choice and short answer, graded locally), weak-spot tracking, and question history.
- **⬡ MODELS**: Ollama server address and default chat, embedding, and vision models. Changing the embedding model re-indexes pages the next time you index.

Data lives at `~/Library/Application Support/Purdue Study Assistant/study_assistant.sqlite3`. Original PDFs stay where they are; keep them in place so page previews work.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests run against a small fake Ollama server (`tests/fake_ollama.py`), so they don't need real models.
