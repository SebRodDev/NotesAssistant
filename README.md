# Cortex

My own AI chat interface for studying. I can build and configure custom agents that answer questions from my lecture notes, with citations to the exact PDF pages. By default the models run on the **Gemini API**, so my laptop stays fast; it can also run fully offline on **Ollama** if a machine has the memory for it. Clean, cyberpunk-inspired design. Still a work in progress!

## How it "knows" my notes

The model isn't fine-tuned. Cortex uses **retrieval-augmented generation (RAG)**, which works better for course notes:

1. **Import.** Lecture PDFs are split into pages and stored in a local SQLite library (full-text indexed).
2. **Index.** Every page is embedded (`gemini-embedding-001` on Gemini, `nomic-embed-text` on Ollama). Optionally, handwriting and diagram labels are read from the page images first.
3. **Ask.** Your question is matched against the pages by keywords *and* meaning. The best pages go to the chat model with instructions to answer from them and cite `[S1]`-style IDs.
4. **Verify.** The app checks every citation against the pages it actually retrieved, drops invented ones, and turns the rest into clickable `[1]`, `[2]` links that open the page.

New notes become searchable as soon as you import and index them. Nothing needs retraining, and answers stay traceable to a slide.

## Setup

1. Get a free Gemini API key at https://aistudio.google.com/apikey.
2. Run from source (Python 3.11+):

   ```bash
   python3.12 -m venv .venv-cortex
   source .venv-cortex/bin/activate
   python -m pip install -e .
   cortex
   ```

3. Open **⬡ MODELS**, paste the key, and press **Save**. It's stored in the macOS Keychain, never in a file. (Setting `GEMINI_API_KEY` in your shell also works.)
4. Add lecture PDFs and press **⟳ INDEX FOR SEARCH**.

After that, `source .venv-cortex/bin/activate && cortex` starts it. To build a double-clickable app: `python -m pip install -e '.[macos]' && python build_macos_app.py`, then open `dist/Cortex.app`.

### Privacy and limits on Gemini

Each question sends the relevant lecture pages to Google. On the free tier, Google may use that content to improve its products, and there are per-minute and per-day request limits. Short rate limits are retried automatically, and a billed key removes both concerns. Your library, PDFs, and chat history stay on your Mac either way.

### Running offline with Ollama instead

On a Mac with enough memory (16 GB or more is comfortable), install [Ollama](https://ollama.com/download), choose **Ollama · runs on this Mac** under **⬡ MODELS**, and download models from the dropdowns (or `ollama pull qwen3:8b nomic-embed-text`). Nothing leaves the Mac in this mode. Each provider keeps its own model picks, so you can switch back and forth.

## Using it

- **Agents** (left sidebar): each agent has its own model, creativity (temperature), notes access, color, and system prompt. Choose **+ NEW** to make one, or **⚙ CONFIG** to edit the active one. It comes with three agents:
  - **Tutor**: notes only, strict citations, says so when your notes don't cover something.
  - **Explainer**: starts from your notes, adds general knowledge marked *(general knowledge)*, and shows its reasoning on thinking models.
  - **Free chat**: a normal chatbot that doesn't read your notes.
- **Courses** tab: **+ ADD LECTURE PDFS** (⌘O), then **⟳ INDEX FOR SEARCH**. Select a class or lecture to scope the conversation. Each agent keeps a separate saved history per class.
- **Composer**: **ASK** chats with the agent; **FIND PAGE** lists matching pages. Enter sends, Shift+Enter adds a new line, and **■ STOP** ends an answer early.
- **Header**: switch the agent's model from the dropdown, and see creativity, generation speed, where the model runs, and context use.
- **Neural link panel** (right): the agent's generated portrait and system prompt. Clicking a citation opens the PDF page and its extracted text there.
- **Study** tab: practice quizzes (multiple choice and short answer, graded by the model against your notes), weak-spot tracking, and question history.
- **⬡ MODELS**: where the models run (Gemini or Ollama), the API key, and the default chat, embedding, and vision models. Dropdowns list only models you can actually use. Changing the embedding model re-indexes pages the next time you index.

Data lives at `~/Library/Application Support/Purdue Study Assistant/study_assistant.sqlite3`. Original PDFs stay where they are; keep them in place so page previews work.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests run against small fake Gemini and Ollama servers (`tests/fake_gemini.py`, `tests/fake_ollama.py`), so they need no API key or models.
