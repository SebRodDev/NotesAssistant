"""SQLite storage, full-text search, and local vector ranking."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import struct
from pathlib import Path


def database_path() -> Path:
    """Return the per-user application database location on macOS."""
    data_dir = Path.home() / "Library" / "Application Support" / "Purdue Study Assistant"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "study_assistant.sqlite3"


DEFAULT_ACCENT = "#00e5ff"

DEFAULT_AGENTS = [
    {
        "name": "Tutor",
        "temperature": 0.2,
        "notes_mode": "strict",
        "reasoning": False,
        "accent": DEFAULT_ACCENT,
        "system_prompt": (
            "You are Tutor, a precise study assistant for a Purdue CS student. Answer clearly and directly "
            "from the student's lecture notes, keep answers short unless asked for detail, and cite the pages "
            "you used."
        ),
    },
    {
        "name": "Explainer",
        "temperature": 0.6,
        "notes_mode": "blend",
        "reasoning": True,
        "accent": "#fcee0a",
        "system_prompt": (
            "You are Explainer, a patient TA. Start from what the lecture notes say, then build intuition with "
            "a small example or analogy. Walk through steps for algorithms and proofs. Flag anything you add "
            "beyond the notes."
        ),
    },
    {
        "name": "Free chat",
        "temperature": 0.8,
        "notes_mode": "blend",
        "reasoning": False,
        "accent": "#e8e8ee",
        "system_prompt": "You are a helpful, friendly assistant. Answer clearly and directly, and keep answers short unless asked for detail.",
    },
]


_SEARCH_STOP_WORDS = {
    "a", "an", "and", "are", "about", "can", "could", "cover", "covered", "does", "do",
    "explain", "find", "for", "give", "how", "in", "is", "it", "me", "of", "on", "page",
    "pages", "please", "show", "slide", "slides", "tell", "the", "this", "to", "what", "where",
    "which", "why", "with",
}


MAX_LEARNED_MEMORIES = 200


def clean_model_name(text: str) -> str:
    """Reduce a model dropdown label to the name Ollama accepts.

    "Default (qwen3:4b)" and "qwen3:4b  ·  2.5 GB" are display text. Ollama rejects them with
    HTTP 400 "invalid model name". Plain names pass through unchanged.
    """
    text = text.strip()
    default_label = re.fullmatch(r"default \((.+)\)", text, re.IGNORECASE)
    if default_label:
        text = default_label[1]
    return text.split("·")[0].strip()


class Library:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or database_path()
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                class_name TEXT NOT NULL,
                lecture_name TEXT NOT NULL,
                file_path TEXT NOT NULL UNIQUE,
                imported_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                page_number INTEGER NOT NULL,
                text TEXT NOT NULL,
                visual_text TEXT NOT NULL DEFAULT '',
                visual_checked INTEGER NOT NULL DEFAULT 0,
                UNIQUE(document_id, page_number)
            );
            CREATE TABLE IF NOT EXISTS page_embeddings (
                page_id INTEGER PRIMARY KEY REFERENCES pages(id) ON DELETE CASCADE,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector BLOB NOT NULL,
                indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS quiz_sessions (
                id INTEGER PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                scope_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS quiz_questions (
                id INTEGER PRIMARY KEY,
                session_id INTEGER NOT NULL REFERENCES quiz_sessions(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                question_type TEXT NOT NULL,
                prompt TEXT NOT NULL,
                options_json TEXT NOT NULL,
                correct_answer TEXT NOT NULL,
                rubric TEXT NOT NULL,
                explanation TEXT NOT NULL,
                topics_json TEXT NOT NULL,
                source_page_ids_json TEXT NOT NULL,
                UNIQUE(session_id, position)
            );
            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id INTEGER PRIMARY KEY,
                question_id INTEGER NOT NULL REFERENCES quiz_questions(id) ON DELETE CASCADE,
                answer TEXT NOT NULL,
                score REAL NOT NULL CHECK(score >= 0 AND score <= 1),
                is_correct INTEGER NOT NULL CHECK(is_correct IN (0, 1)),
                feedback TEXT NOT NULL,
                attempted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS class_chat_messages (
                id INTEGER PRIMARY KEY,
                class_name TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(pages)")}
        if "visual_text" not in columns:
            self.connection.execute("ALTER TABLE pages ADD COLUMN visual_text TEXT NOT NULL DEFAULT ''")
        if "visual_checked" not in columns:
            self.connection.execute("ALTER TABLE pages ADD COLUMN visual_checked INTEGER NOT NULL DEFAULT 0")

        # Rebuild the external-content FTS table once so previous libraries also index visual text.
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version < 2:
            self.connection.executescript(
                """
                DROP TRIGGER IF EXISTS pages_ai;
                DROP TRIGGER IF EXISTS pages_ad;
                DROP TRIGGER IF EXISTS pages_au;
                DROP TABLE IF EXISTS pages_fts;
                CREATE VIRTUAL TABLE pages_fts USING fts5(
                    text, visual_text, content='pages', content_rowid='id'
                );
                INSERT INTO pages_fts(pages_fts) VALUES ('rebuild');
                CREATE TRIGGER pages_ai AFTER INSERT ON pages BEGIN
                    INSERT INTO pages_fts(rowid, text, visual_text)
                    VALUES (new.id, new.text, new.visual_text);
                END;
                CREATE TRIGGER pages_ad AFTER DELETE ON pages BEGIN
                    INSERT INTO pages_fts(pages_fts, rowid, text, visual_text)
                    VALUES ('delete', old.id, old.text, old.visual_text);
                END;
                CREATE TRIGGER pages_au AFTER UPDATE ON pages BEGIN
                    INSERT INTO pages_fts(pages_fts, rowid, text, visual_text)
                    VALUES ('delete', old.id, old.text, old.visual_text);
                    INSERT INTO pages_fts(rowid, text, visual_text)
                    VALUES (new.id, new.text, new.visual_text);
                END;
                PRAGMA user_version = 2;
                """
            )
        if version < 3:
            self.connection.execute("PRAGMA user_version = 3")
        if version < 4:
            # v4: local agents + per-agent chat history (Ollama rewrite).
            chat_columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(class_chat_messages)")}
            if "agent_id" not in chat_columns:
                self.connection.execute("ALTER TABLE class_chat_messages ADD COLUMN agent_id INTEGER")
            self.connection.execute("PRAGMA user_version = 4")
        if version < 5:
            # v5: every conversation is saved, including lecture and all-materials ('' class) scopes.
            chat_columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(class_chat_messages)")}
            if "lecture_name" not in chat_columns:
                self.connection.execute(
                    "ALTER TABLE class_chat_messages ADD COLUMN lecture_name TEXT NOT NULL DEFAULT ''"
                )
            self.connection.execute("PRAGMA user_version = 5")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS agents (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                model TEXT NOT NULL DEFAULT '',
                temperature REAL NOT NULL DEFAULT 0.3,
                notes_mode TEXT NOT NULL DEFAULT 'strict',
                reasoning INTEGER NOT NULL DEFAULT 0,
                accent TEXT NOT NULL DEFAULT '#00e5ff',
                system_prompt TEXT NOT NULL DEFAULT '',
                position INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL UNIQUE COLLATE NOCASE,
                pinned INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        if not self.connection.execute("SELECT 1 FROM agents LIMIT 1").fetchone():
            for position, agent in enumerate(DEFAULT_AGENTS):
                self.connection.execute(
                    """INSERT INTO agents(name, model, temperature, notes_mode, reasoning, accent, system_prompt, position)
                       VALUES (?, '', ?, ?, ?, ?, ?, ?)""",
                    (agent["name"], agent["temperature"], agent["notes_mode"], int(agent["reasoning"]),
                     agent["accent"], agent["system_prompt"], position),
                )
        if version < 6:
            # v6: pink/purple theme. Retired swatches move to their replacements, and the built-in agents
            # get their new colors unless they were recolored. Other chosen colors are left alone.
            for old, new in (("#9b87f5", DEFAULT_ACCENT), ("#ff6ac1", "#ff2e97")):
                self.connection.execute("UPDATE agents SET accent=? WHERE accent=?", (new, old))
            for name, old, new in (("Explainer", "#2ee6d6", "#ff2e97"), ("Free chat", "#e8e8ee", "#f06bff")):
                self.connection.execute("UPDATE agents SET accent=? WHERE name=? AND accent=?", (new, name, old))
            self.connection.execute("PRAGMA user_version = 6")
        if version < 7:
            # v7: dropdown labels were sometimes saved as an agent's model, which Ollama rejects. Keep the
            # model the label named rather than clearing it, since the default may not be installed.
            for agent_id, model in self.connection.execute("SELECT id, model FROM agents").fetchall():
                if clean_model_name(model) != model:
                    self.connection.execute(
                        "UPDATE agents SET model=? WHERE id=?", (clean_model_name(model), agent_id)
                    )
            self.connection.execute("PRAGMA user_version = 7")
        if version < 8:
            # v8: cyberpunk palette (cyan-led). Built-in agents still on their v6 colors move to the new
            # ones; any agent the user recolored keeps its color.
            for name, old, new in (("Tutor", "#b14cff", DEFAULT_ACCENT), ("Explainer", "#ff2e97", "#fcee0a"),
                                   ("Free chat", "#f06bff", "#e8e8ee")):
                self.connection.execute("UPDATE agents SET accent=? WHERE name=? AND accent=?", (new, name, old))
            self.connection.execute("PRAGMA user_version = 8")
        if version < 9:
            # v9: the main chat always answers from the selected notes, so "no notes" agents use notes plus
            # general knowledge instead (the screen overlay is where notes are left out).
            self.connection.execute("UPDATE agents SET notes_mode='blend' WHERE notes_mode='off'")
            self.connection.execute("PRAGMA user_version = 9")
        self.connection.commit()

    # -- agents ------------------------------------------------------------------------------
    def agents(self) -> list[dict]:
        return [dict(row) for row in self.connection.execute("SELECT * FROM agents ORDER BY position, id")]

    def agent(self, agent_id: int) -> dict | None:
        row = self.connection.execute("SELECT * FROM agents WHERE id=?", (agent_id,)).fetchone()
        return dict(row) if row else None

    def save_agent(self, agent: dict) -> int:
        name = str(agent.get("name", "")).strip()
        if not name:
            raise ValueError("An agent needs a name.")
        if agent.get("notes_mode") not in {"strict", "blend"}:
            raise ValueError("Unknown notes mode.")
        temperature = float(agent.get("temperature", 0.3))
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("Creativity must be between 0 and 2.")
        values = (
            name, str(agent.get("model") or "").strip(), temperature, agent["notes_mode"],
            int(bool(agent.get("reasoning"))), str(agent.get("accent") or DEFAULT_ACCENT),
            str(agent.get("system_prompt") or "").strip(),
        )
        with self.connection:
            if agent.get("id"):
                self.connection.execute(
                    """UPDATE agents SET name=?, model=?, temperature=?, notes_mode=?, reasoning=?, accent=?,
                       system_prompt=? WHERE id=?""",
                    (*values, agent["id"]),
                )
                return int(agent["id"])
            position = self.connection.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM agents").fetchone()[0]
            cursor = self.connection.execute(
                """INSERT INTO agents(name, model, temperature, notes_mode, reasoning, accent, system_prompt, position)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (*values, position),
            )
            return int(cursor.lastrowid)

    def delete_agent(self, agent_id: int) -> None:
        if self.connection.execute("SELECT COUNT(*) FROM agents").fetchone()[0] <= 1:
            raise ValueError("Keep at least one agent.")
        with self.connection:
            self.connection.execute("DELETE FROM agents WHERE id=?", (agent_id,))
            self.connection.execute("DELETE FROM class_chat_messages WHERE agent_id=?", (agent_id,))

    def import_pdf(self, class_name: str, lecture_name: str, file_path: Path, page_texts: list[str]) -> None:
        """Replace an existing import at this path and save one record per page."""
        class_name = class_name.strip()
        lecture_name = lecture_name.strip()
        if not class_name or not lecture_name:
            raise ValueError("Class and lecture names cannot be blank.")
        if not page_texts:
            raise ValueError("A PDF must contain at least one page.")
        resolved_path = str(file_path.expanduser().resolve())
        with self.connection:
            self.connection.execute("DELETE FROM documents WHERE file_path = ?", (resolved_path,))
            cursor = self.connection.execute(
                "INSERT INTO documents(class_name, lecture_name, file_path) VALUES (?, ?, ?)",
                (class_name, lecture_name, resolved_path),
            )
            document_id = cursor.lastrowid
            self.connection.executemany(
                "INSERT INTO pages(document_id, page_number, text) VALUES (?, ?, ?)",
                [(document_id, number, text) for number, text in enumerate(page_texts, start=1)],
            )

    def documents(self) -> list[dict]:
        return [dict(row) for row in self.connection.execute(
            """SELECT d.*, COUNT(p.id) AS page_count,
                      SUM(CASE WHEN e.page_id IS NOT NULL THEN 1 ELSE 0 END) AS indexed_pages
               FROM documents d LEFT JOIN pages p ON p.document_id=d.id
               LEFT JOIN page_embeddings e ON e.page_id=p.id
               GROUP BY d.id ORDER BY d.class_name, d.lecture_name"""
        )]

    def pages_to_index(self, model: str) -> list[sqlite3.Row]:
        return list(self.connection.execute(
            """SELECT p.id, p.page_number, p.text, p.visual_text,
                      d.class_name, d.lecture_name, d.file_path
               FROM pages p JOIN documents d ON d.id=p.document_id
               LEFT JOIN page_embeddings e ON e.page_id=p.id AND e.model=?
               WHERE e.page_id IS NULL ORDER BY d.class_name, d.lecture_name, p.page_number""",
            (model,),
        ))

    def pages_needing_visual_scan(self) -> list[sqlite3.Row]:
        """Return pages not yet checked for handwriting or meaningful visual labels."""
        return list(self.connection.execute(
            """SELECT p.id, p.page_number, p.text, p.visual_text,
                      d.class_name, d.lecture_name, d.file_path
               FROM pages p JOIN documents d ON d.id=p.document_id
               WHERE p.visual_checked=0
               ORDER BY d.class_name, d.lecture_name, p.page_number"""
        ))

    def indexing_status(self, model: str) -> dict[str, int]:
        """Return compact counts for library readiness without loading page content."""
        row = self.connection.execute(
            """SELECT COUNT(*) AS total_pages,
                      SUM(CASE WHEN e.page_id IS NOT NULL THEN 1 ELSE 0 END) AS indexed_pages,
                      SUM(CASE WHEN p.visual_checked=0 THEN 1 ELSE 0 END) AS pending_visual_pages
               FROM pages p
               LEFT JOIN page_embeddings e ON e.page_id=p.id AND e.model=?""",
            (model,),
        ).fetchone()
        return {
            "total_pages": row["total_pages"] or 0,
            "indexed_pages": row["indexed_pages"] or 0,
            "pending_visual_pages": row["pending_visual_pages"] or 0,
        }

    def lecture_pages(self, class_name: str, lecture_name: str) -> list[dict]:
        """Every page of one lecture, in order (a lecture can span several PDFs)."""
        return self.scope_pages(class_name, lecture_name)

    def scope_pages(self, class_name: str | None, lecture_name: str | None = None) -> list[dict]:
        """Every page in a lecture, a whole course (lecture None), or the library (both None), in the
        order the material was taught: course, then lecture in import order, then page."""
        where, parameters = [], []
        if class_name is not None:
            where.append("d.class_name=?")
            parameters.append(class_name)
        if lecture_name is not None:
            where.append("d.lecture_name=?")
            parameters.append(lecture_name)
        return [dict(row) for row in self.connection.execute(
            f"""SELECT p.id, p.page_number, p.text, p.visual_text, d.class_name, d.lecture_name, d.file_path
                FROM pages p JOIN documents d ON d.id=p.document_id
                JOIN (SELECT class_name, lecture_name, MIN(imported_at || printf('%010d', id)) AS taught
                      FROM documents GROUP BY class_name, lecture_name) l
                  ON l.class_name=d.class_name AND l.lecture_name=d.lecture_name
                {"WHERE " + " AND ".join(where) if where else ""}
                ORDER BY d.class_name, l.taught, d.imported_at, d.id, p.page_number""",
            parameters,
        )]

    def scope_status(self, class_name: str | None, lecture_name: str | None, model: str) -> dict[str, int]:
        """What's stored for a course, a lecture, or everything (None): PDFs, lectures, pages, pages with
        any text, and pages indexed for meaning search with `model`. Used to explain empty answers."""
        where, parameters = [], [model]
        if class_name is not None:
            where.append("d.class_name=?")
            parameters.append(class_name)
        if lecture_name is not None:
            where.append("d.lecture_name=?")
            parameters.append(lecture_name)
        row = self.connection.execute(
            f"""SELECT COUNT(DISTINCT d.id) AS documents,
                       COUNT(DISTINCT d.class_name || char(31) || d.lecture_name) AS lectures,
                       COUNT(p.id) AS pages,
                       SUM(CASE WHEN TRIM(COALESCE(p.text, '')) != '' OR TRIM(COALESCE(p.visual_text, '')) != ''
                           THEN 1 ELSE 0 END) AS text_pages,
                       SUM(CASE WHEN e.page_id IS NOT NULL THEN 1 ELSE 0 END) AS indexed
                FROM documents d LEFT JOIN pages p ON p.document_id=d.id
                LEFT JOIN page_embeddings e ON e.page_id=p.id AND e.model=?
                {"WHERE " + " AND ".join(where) if where else ""}""",
            parameters,
        ).fetchone()
        return {key: int(row[key] or 0) for key in ("documents", "lectures", "pages", "text_pages", "indexed")}

    def pages_for_scopes(self, scopes: list[tuple[str, str]], limit: int = 16) -> list[dict]:
        """Select pages spread across chosen lectures for quiz generation."""
        if limit < 1:
            raise ValueError("The page selection limit must be positive.")
        if not scopes:
            return []
        normalized_scopes = list(dict.fromkeys(
            (class_name.strip(), lecture_name.strip())
            for class_name, lecture_name in scopes
            if class_name.strip() and lecture_name.strip()
        ))
        if not normalized_scopes:
            return []
        clauses = " OR ".join("(d.class_name=? AND d.lecture_name=?)" for _ in normalized_scopes)
        parameters = [value for scope in normalized_scopes for value in scope]
        rows = [dict(row) for row in self.connection.execute(
            f"""SELECT p.id, p.page_number, p.text, p.visual_text,
                       d.class_name, d.lecture_name, d.file_path
                FROM pages p JOIN documents d ON d.id=p.document_id
                WHERE {clauses} ORDER BY d.class_name, d.lecture_name, p.page_number""",
            parameters,
        )]
        grouped: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            grouped.setdefault((row["class_name"], row["lecture_name"]), []).append(row)
        groups = list(grouped.values())
        if len(groups) > limit:
            if limit == 1:
                groups = [groups[len(groups) // 2]]
            else:
                indexes = {round(index * (len(groups) - 1) / (limit - 1)) for index in range(limit)}
                groups = [groups[index] for index in sorted(indexes)]

        # Give each selected lecture one page, then distribute the remaining budget evenly.
        budgets = [1] * len(groups)
        remaining = min(limit, len(rows)) - len(groups)
        while remaining:
            changed = False
            for index, group in enumerate(groups):
                if budgets[index] < len(group):
                    budgets[index] += 1
                    remaining -= 1
                    changed = True
                    if not remaining:
                        break
            if not changed:
                break

        selected = []
        for group, budget in zip(groups, budgets, strict=True):
            if budget == len(group):
                selected.extend(group)
            elif budget == 1:
                selected.append(group[len(group) // 2])
            else:
                indexes = {round(index * (len(group) - 1) / (budget - 1)) for index in range(budget)}
                selected.extend(group[index] for index in sorted(indexes))
        return selected

    def pages_by_ids(self, page_ids: list[int]) -> list[dict]:
        if not page_ids:
            return []
        placeholders = ",".join("?" for _ in page_ids)
        return [dict(row) for row in self.connection.execute(
            f"""SELECT p.id AS page_id, p.page_number, p.text, p.visual_text,
                       d.class_name, d.lecture_name, d.file_path
                FROM pages p JOIN documents d ON d.id=p.document_id
                WHERE p.id IN ({placeholders}) ORDER BY d.class_name, d.lecture_name, p.page_number""",
            page_ids,
        )]

    def create_quiz_session(self, scopes: list[tuple[str, str]], questions: list[dict]) -> list[dict]:
        """Persist a generated quiz and its answer keys before showing questions."""
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO quiz_sessions(scope_json) VALUES (?)",
                (json.dumps(scopes),),
            )
            session_id = cursor.lastrowid
            saved_questions = []
            for position, question in enumerate(questions, start=1):
                page_ids = []
                for source_id in question["source_ids"]:
                    source = question["source_map"][source_id]
                    page_id = source.get("page_id", source.get("id"))
                    if not isinstance(page_id, int) or isinstance(page_id, bool):
                        raise ValueError("Quiz sources must include a valid library page ID.")
                    page_ids.append(page_id)
                question_cursor = self.connection.execute(
                    """INSERT INTO quiz_questions(
                        session_id, position, question_type, prompt, options_json,
                        correct_answer, rubric, explanation, topics_json, source_page_ids_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id, position, question["question_type"], question["prompt"],
                        json.dumps(question["options"]), question["correct_answer"],
                        question["rubric"], question["explanation"], json.dumps(question["topics"]),
                        json.dumps(page_ids),
                    ),
                )
                saved_questions.append({
                    **question,
                    "id": question_cursor.lastrowid,
                    "source_page_ids": page_ids,
                    "session_id": session_id,
                })
        return saved_questions

    def record_quiz_attempt(
        self, question_id: int, answer: str, score: float, is_correct: bool, feedback: str
    ) -> None:
        if isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Quiz score must be a finite number between 0 and 1.")
        if not isinstance(is_correct, bool) or is_correct != (score >= 0.75):
            raise ValueError("Quiz correctness must match the 75% mastery threshold.")
        with self.connection:
            self.connection.execute(
                """INSERT INTO quiz_attempts(question_id, answer, score, is_correct, feedback)
                   VALUES (?, ?, ?, ?, ?)""",
                (question_id, answer, score, int(is_correct), feedback),
            )

    def topic_performance(self) -> list[dict]:
        """Calculate recency-weighted topic accuracy from saved quiz attempts."""
        rows = self.connection.execute(
            """SELECT a.score, a.is_correct, a.attempted_at, q.topics_json
               FROM quiz_attempts a JOIN quiz_questions q ON q.id=a.question_id
               ORDER BY a.attempted_at, a.id"""
        ).fetchall()
        by_topic: dict[str, list[sqlite3.Row]] = {}
        topic_labels: dict[str, str] = {}
        for row in rows:
            unique_topics = {
                topic.strip().casefold(): topic.strip()
                for topic in json.loads(row["topics_json"])
                if isinstance(topic, str) and topic.strip()
            }
            for key, label in unique_topics.items():
                topic_labels.setdefault(key, label)
                by_topic.setdefault(key, []).append(row)
        summaries = []
        for key, attempts in by_topic.items():
            topic = topic_labels[key]
            weights = [0.9 ** (len(attempts) - index - 1) for index in range(len(attempts))]
            total_weight = sum(weights)
            accuracy = sum(row["score"] * weight for row, weight in zip(attempts, weights, strict=True)) / total_weight
            summaries.append({
                "topic": topic,
                "accuracy": accuracy,
                "attempts": len(attempts),
                "misses": sum(not row["is_correct"] for row in attempts),
                "last_attempt": attempts[-1]["attempted_at"],
            })
        return sorted(summaries, key=lambda item: (item["accuracy"], -item["attempts"], item["topic"].casefold()))

    def save_embedding(self, page_id: int, values: list[float], model: str) -> None:
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError("An embedding must contain finite numeric values.")
        if not model.strip():
            raise ValueError("An embedding model name cannot be blank.")
        vector = struct.pack(f"<{len(values)}f", *values)
        with self.connection:
            self.connection.execute(
                """INSERT OR REPLACE INTO page_embeddings(page_id, model, dimensions, vector)
                   VALUES (?, ?, ?, ?)""",
                (page_id, model, len(values), vector),
            )

    def save_visual_text(self, page_id: int, visual_text: str) -> None:
        """Store a clearly labeled visual transcription and invalidate its old vector."""
        with self.connection:
            self.connection.execute(
                "UPDATE pages SET visual_text=?, visual_checked=1 WHERE id=?",
                (visual_text.strip(), page_id),
            )
            self.connection.execute("DELETE FROM page_embeddings WHERE page_id=?", (page_id,))

    def search(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        limit: int = 100,
        model: str | None = None,
        class_name: str | None = None,
        lecture_name: str | None = None,
        match_any: bool = False,
    ) -> list[dict]:
        """Combine keyword and vector rankings with reciprocal-rank fusion.

        match_any ranks pages that contain any query term (BM25 still favours pages with more
        of them), which suits natural-language questions; the default requires every term.
        """
        if limit < 1:
            raise ValueError("Search result limit must be positive.")
        if query_embedding is not None and any(not math.isfinite(value) for value in query_embedding):
            raise ValueError("Query embedding values must be finite numbers.")
        raw_terms = re.findall(r"[\w'-]+", query.casefold())
        terms = [term for term in raw_terms if term not in _SEARCH_STOP_WORDS] or raw_terms
        if not terms:
            return []
        match_query = (" OR " if match_any else " AND ").join(f'"{term}"' for term in terms)
        scope_clauses = []
        scope_parameters = []
        if class_name:
            scope_clauses.append("d.class_name=?")
            scope_parameters.append(class_name)
        if lecture_name:
            scope_clauses.append("d.lecture_name=?")
            scope_parameters.append(lecture_name)
        scope_sql = " AND " + " AND ".join(scope_clauses) if scope_clauses else ""
        lexical = list(self.connection.execute(
            f"""SELECT d.class_name, d.lecture_name, d.file_path, p.page_number,
                      p.text, p.visual_text, p.id AS page_id, bm25(pages_fts) AS rank
               FROM pages_fts JOIN pages p ON p.id=pages_fts.rowid
               JOIN documents d ON d.id=p.document_id
               WHERE pages_fts MATCH ?{scope_sql} ORDER BY rank LIMIT ?""",
            [match_query, *scope_parameters, limit],
        ))
        ranked: dict[int, dict] = {}
        for position, row in enumerate(lexical, start=1):
            item = dict(row)
            item["score"] = 1 / (60 + position)
            item["match_type"] = "keyword"
            ranked[item["page_id"]] = item

        if query_embedding:
            expected_bytes = len(query_embedding) * 4
            vectors = self.connection.execute(
                f"""SELECT p.id AS page_id, p.page_number, p.text, p.visual_text,
                          d.class_name, d.lecture_name, d.file_path, e.dimensions, e.vector
                   FROM page_embeddings e JOIN pages p ON p.id=e.page_id
                   JOIN documents d ON d.id=p.document_id
                   WHERE e.dimensions=? AND (? IS NULL OR e.model=?){scope_sql}""",
                [len(query_embedding), model, model, *scope_parameters],
            )
            similarities = []
            query_norm = math.sqrt(sum(value * value for value in query_embedding))
            if query_norm:
                for row in vectors:
                    if len(row["vector"]) != expected_bytes:
                        continue
                    values = struct.unpack(f"<{row['dimensions']}f", row["vector"])
                    norm = math.sqrt(sum(value * value for value in values))
                    if not norm:
                        continue
                    similarity = sum(a * b for a, b in zip(query_embedding, values, strict=True)) / (query_norm * norm)
                    similarities.append((similarity, row))
            similarities.sort(key=lambda item: item[0], reverse=True)
            for position, (similarity, row) in enumerate(similarities[:limit], start=1):
                item = dict(row)
                prior = ranked.get(item["page_id"])
                item["score"] = 1 / (60 + position) + (prior["score"] if prior else 0)
                item["match_type"] = "hybrid" if prior else "semantic"
                item["similarity"] = similarity
                ranked[item["page_id"]] = item

        return sorted(ranked.values(), key=lambda item: item["score"], reverse=True)[:limit]

    # -- long-term memory ----------------------------------------------------------------------
    def memories(self) -> list[dict]:
        return [dict(row) for row in self.connection.execute("SELECT * FROM memories ORDER BY id DESC")]

    def add_memory(self, content: str, pinned: bool = False) -> None:
        """Save a fact. Pinned facts came from the student and survive the cap on learned ones."""
        content = " ".join(content.split())
        if not content:
            raise ValueError("A memory can't be blank.")
        with self.connection:
            self.connection.execute(
                """INSERT INTO memories(content, pinned) VALUES (?, ?)
                   ON CONFLICT(content) DO UPDATE SET pinned=MAX(pinned, excluded.pinned)""",
                (content, int(pinned)),
            )
            self.connection.execute(
                """DELETE FROM memories WHERE pinned=0 AND id NOT IN
                   (SELECT id FROM memories WHERE pinned=0 ORDER BY id DESC LIMIT ?)""",
                (MAX_LEARNED_MEMORIES,),
            )

    def delete_memory(self, memory_id: int) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM memories WHERE id=?", (memory_id,))

    def clear_memories(self) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM memories")

    def relevant_memories(self, query: str, limit: int = 8) -> list[str]:
        """The facts sharing the most words with the query, newest first on ties."""
        terms = set(re.findall(r"[\w'-]+", query.casefold())) - _SEARCH_STOP_WORDS
        rows = self.memories()
        rows.sort(key=lambda row: len(terms & set(re.findall(r"[\w'-]+", row["content"].casefold()))), reverse=True)
        return [row["content"] for row in rows[:limit]]

    # -- conversations ---------------------------------------------------------------------------
    def save_class_chat(
        self, class_name: str, question: str, answer: str, sources: list[dict], agent_id: int | None = None,
        lecture_name: str = "",
    ) -> int:
        safe_sources = [
            {key: source.get(key) for key in (
                "class_name", "lecture_name", "file_path", "page_number", "text", "visual_text", "page_id"
            )}
            for source in sources
        ]
        with self.connection:
            cursor = self.connection.execute(
                """INSERT INTO class_chat_messages(class_name, lecture_name, question, answer, sources_json, agent_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (class_name, lecture_name, question, answer, json.dumps(safe_sources), agent_id),
            )
            return int(cursor.lastrowid)

    @staticmethod
    def _chat_filter(class_name: str, agent_id: int | None, lecture_name: str | None) -> tuple[str, list]:
        clause, parameters = "class_name=?", [class_name]
        if agent_id is not None:
            clause += " AND (agent_id=? OR agent_id IS NULL)"
            parameters.append(agent_id)
        if lecture_name is not None:
            clause += " AND lecture_name=?"
            parameters.append(lecture_name)
        return clause, parameters

    def class_chat_history(
        self, class_name: str, limit: int = 100, agent_id: int | None = None, lecture_name: str | None = None,
    ) -> list[dict]:
        """Saved Q&A for a class ('' is all materials), optionally for one agent or lecture.

        With agent_id, chats from before agents existed are included too. lecture_name='' means
        the class as a whole, while None means every lecture in it.
        """
        if limit < 1:
            raise ValueError("Chat history limit must be positive.")
        clause, parameters = self._chat_filter(class_name, agent_id, lecture_name)
        rows = self.connection.execute(
            f"""SELECT id, question, answer, sources_json, created_at
                FROM class_chat_messages WHERE {clause} ORDER BY id DESC LIMIT ?""",
            [*parameters, limit],
        ).fetchall()
        history = []
        for row in reversed(rows):
            history.append({
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "sources": json.loads(row["sources_json"]),
                "created_at": row["created_at"],
            })
        return history

    def clear_class_chat(
        self, class_name: str, agent_id: int | None = None, lecture_name: str | None = None,
    ) -> None:
        clause, parameters = self._chat_filter(class_name, agent_id, lecture_name)
        with self.connection:
            self.connection.execute(f"DELETE FROM class_chat_messages WHERE {clause}", parameters)

    def delete_chats(self, chat_ids: list[int]) -> None:
        with self.connection:
            self.connection.executemany("DELETE FROM class_chat_messages WHERE id=?", [(i,) for i in chat_ids])

    def close(self) -> None:
        self.connection.close()
