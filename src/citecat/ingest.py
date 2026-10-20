"""Text ingestion: fetch HAL records and store passages in SQLite."""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import struct
import threading
from datetime import UTC, datetime

from citecat.config import Config
from citecat.models import ExtractedPage, Passage, PDFExtractionResult, Record

logger = logging.getLogger(__name__)


def serialize_vector(vector: list[float]) -> bytes:
    """Pack a list of float numbers into an IEEE 754 32-bit binary blob."""
    return struct.pack(f"{len(vector)}f", *vector)


def deserialize_vector(blob: bytes, expected_dim: int | None = None) -> list[float] | None:
    """Unpack binary blob to list of floats, or return None if corrupt or dimension mismatch."""
    if len(blob) % 4 != 0:
        logger.warning("Corrupted embedding blob: byte length %d is not divisible by 4", len(blob))
        return None
    num_floats = len(blob) // 4
    if expected_dim is not None and num_floats != expected_dim:
        logger.warning(
            "Embedding dimension mismatch: expected %d, got %d", expected_dim, num_floats
        )
        return None
    try:
        return list(struct.unpack(f"{num_floats}f", blob))
    except struct.error as exc:
        logger.warning("Failed to unpack embedding vector: %s", exc)
        return None


def compute_text_hash(text: str) -> str:
    """Compute SHA-256 hash of text for staleness detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _init_db(conn: sqlite3.Connection) -> None:
    """Create the retrieval and embedding tables if they don't exist and run safe migrations."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS records (
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            source_url  TEXT NOT NULL,
            license     TEXT NOT NULL,
            full_text   TEXT NOT NULL DEFAULT '',
            retrieved_at TEXT NOT NULL,
            pdf_url     TEXT DEFAULT NULL,
            pdf_license TEXT DEFAULT NULL,
            source_type TEXT NOT NULL DEFAULT 'abstract'
        );
        CREATE TABLE IF NOT EXISTS passages (
            rowid       INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id   TEXT NOT NULL REFERENCES records(id),
            text        TEXT NOT NULL,
            start_char  INTEGER NOT NULL,
            end_char    INTEGER NOT NULL,
            word_count  INTEGER NOT NULL,
            page_number INTEGER DEFAULT NULL,
            source_type TEXT NOT NULL DEFAULT 'abstract',
            pdf_url     TEXT DEFAULT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_passages_record ON passages(record_id);

        CREATE TABLE IF NOT EXISTS embeddings (
            record_id   TEXT NOT NULL,
            start_char  INTEGER NOT NULL,
            end_char    INTEGER NOT NULL,
            text_hash   TEXT NOT NULL,
            model_name  TEXT NOT NULL,
            dimension   INTEGER NOT NULL,
            vector      BLOB NOT NULL,
            created_at  TEXT NOT NULL,
            PRIMARY KEY (record_id, start_char, end_char, model_name),
            FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_name);
        """
    )

    # Safe migrations for existing databases
    p_cols = {row[1] for row in conn.execute("PRAGMA table_info(passages)")}
    if "page_number" not in p_cols:
        conn.execute("ALTER TABLE passages ADD COLUMN page_number INTEGER DEFAULT NULL")
    if "source_type" not in p_cols:
        conn.execute("ALTER TABLE passages ADD COLUMN source_type TEXT NOT NULL DEFAULT 'abstract'")
    if "pdf_url" not in p_cols:
        conn.execute("ALTER TABLE passages ADD COLUMN pdf_url TEXT DEFAULT NULL")

    r_cols = {row[1] for row in conn.execute("PRAGMA table_info(records)")}
    if "pdf_url" not in r_cols:
        conn.execute("ALTER TABLE records ADD COLUMN pdf_url TEXT DEFAULT NULL")
    if "pdf_license" not in r_cols:
        conn.execute("ALTER TABLE records ADD COLUMN pdf_license TEXT DEFAULT NULL")
    if "source_type" not in r_cols:
        conn.execute("ALTER TABLE records ADD COLUMN source_type TEXT NOT NULL DEFAULT 'abstract'")


def split_into_passages(
    text: str,
    min_words: int = 200,
    max_words: int = 300,
) -> list[tuple[str, int, int, int]]:
    """Split *text* into passages of approximately *min_words*–*max_words* words.

    Each passage is returned as ``(passage_text, start_char, end_char, word_count)``.
    Splitting breaks on sentence boundaries when possible.

    Preserves exact source text and character offsets so that
    ``text[start_char:end_char] == passage_text`` always holds.
    """
    if not text or not text.strip():
        return []

    word_matches = list(re.finditer(r"\S+", text))
    if not word_matches:
        return []

    passages: list[tuple[str, int, int, int]] = []
    current_idx = 0
    total_words = len(word_matches)

    while current_idx < total_words:
        end_idx = min(current_idx + max_words, total_words)

        # Look for sentence break if we exceed min_words and haven't reached the end
        if (end_idx - current_idx) > min_words and end_idx < total_words:
            best_break = None
            for i in range(end_idx - 1, current_idx + min_words - 1, -1):
                w = word_matches[i].group()
                if w.endswith((".", "?", "!", '."', '?"', '!"', ".)", "?)", "!)")):
                    best_break = i + 1
                    break
            if best_break is not None:
                end_idx = best_break

        start_char = word_matches[current_idx].start()
        end_char = word_matches[end_idx - 1].end()
        passage_text = text[start_char:end_char]
        word_count = end_idx - current_idx

        passages.append((passage_text, start_char, end_char, word_count))
        current_idx = end_idx

    return passages


class IngestStore:
    """Manages the SQLite store for ingested records and passages."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Config()
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._config.retrieval_db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            _init_db(self._conn)
        return self._conn

    def ingest_record(self, record: Record, full_text: str | None = None) -> int:
        """Store a record and its passages. Returns number of passages created."""
        with self._lock:
            conn = self._get_conn()
            text = full_text or record.snippet or ""

            conn.execute(
                """
                INSERT OR REPLACE INTO records
                (id, title, source_url, license, full_text, retrieved_at,
                 pdf_url, pdf_license, source_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'abstract')
                """,
                (
                    record.id,
                    record.title,
                    record.source_url,
                    record.license,
                    text,
                    record.retrieved_at.isoformat(),
                    record.pdf_url,
                    record.pdf_license,
                ),
            )
            # Remove old passages and embeddings for this record
            conn.execute("DELETE FROM passages WHERE record_id = ?", (record.id,))
            conn.execute("DELETE FROM embeddings WHERE record_id = ?", (record.id,))

            passages = split_into_passages(
                text,
                min_words=self._config.passage_min_words,
                max_words=self._config.passage_max_words,
            )
            for p_text, start, end, wc in passages:
                conn.execute(
                    """
                    INSERT INTO passages
                    (record_id, text, start_char, end_char, word_count,
                     page_number, source_type, pdf_url)
                    VALUES (?, ?, ?, ?, ?, NULL, 'abstract', NULL)
                    """,
                    (record.id, p_text, start, end, wc),
                )
            conn.commit()
            return len(passages)

    def ingest_pdf(self, record: Record, extraction: PDFExtractionResult) -> int:
        """Store an extracted PDF document and its page-aware passages.

        Builds combined full_text while tracking exact character offsets per page.
        Returns number of passages created.
        """
        if extraction.status != "success" or extraction.ocr_required:
            logger.warning(
                "Skipping PDF ingestion for record %s: extraction status '%s' (ocr_required=%s)",
                record.id,
                extraction.status,
                extraction.ocr_required,
            )
            return 0

        with self._lock:
            conn = self._get_conn()

            # Build combined full text while tracking exact character offset of each page
            full_text_parts: list[str] = []
            page_offsets: list[tuple[ExtractedPage, int]] = []
            current_offset = 0

            for page in extraction.pages:
                if not page.text.strip():
                    continue
                if full_text_parts:
                    sep = "\n\n"
                    current_offset += len(sep)
                    full_text_parts.append(sep)
                page_offsets.append((page, current_offset))
                full_text_parts.append(page.text)
                current_offset += len(page.text)

            full_text = "".join(full_text_parts)

            # Store or replace record
            conn.execute(
                """
                INSERT OR REPLACE INTO records
                (id, title, source_url, license, full_text, retrieved_at,
                 pdf_url, pdf_license, source_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pdf')
                """,
                (
                    record.id,
                    record.title,
                    record.source_url,
                    record.license,
                    full_text,
                    record.retrieved_at.isoformat(),
                    extraction.pdf_url,
                    record.pdf_license,
                ),
            )

            # Remove previous passages and embeddings for this record
            conn.execute("DELETE FROM passages WHERE record_id = ?", (record.id,))
            conn.execute("DELETE FROM embeddings WHERE record_id = ?", (record.id,))

            passages_count = 0
            for page, page_start in page_offsets:
                splits = split_into_passages(
                    page.text,
                    min_words=self._config.passage_min_words,
                    max_words=self._config.passage_max_words,
                )
                for p_text, p_start, p_end, wc in splits:
                    global_start = page_start + p_start
                    global_end = page_start + p_end
                    conn.execute(
                        """
                        INSERT INTO passages
                        (record_id, text, start_char, end_char, word_count,
                         page_number, source_type, pdf_url)
                        VALUES (?, ?, ?, ?, ?, ?, 'pdf', ?)
                        """,
                        (
                            record.id,
                            p_text,
                            global_start,
                            global_end,
                            wc,
                            page.page_number,
                            extraction.pdf_url,
                        ),
                    )
                    passages_count += 1

            conn.commit()
            return passages_count

    def get_all_passages(self) -> list[Passage]:
        """Return all stored passages."""
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                """
                SELECT record_id, text, start_char, end_char, word_count,
                       page_number, source_type, pdf_url
                FROM passages
                """
            ).fetchall()
            return [
                Passage(
                    record_id=r[0],
                    text=r[1],
                    start_char=r[2],
                    end_char=r[3],
                    word_count=r[4],
                    page_number=r[5],
                    source_type=r[6] or "abstract",
                    pdf_url=r[7],
                )
                for r in rows
            ]

    def get_record_text(self, record_id: str) -> str | None:
        """Return the full text for a record."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT full_text FROM records WHERE id = ?", (record_id,)
            ).fetchone()
            return row[0] if row else None

    def get_record_metadata(self, record_id: str) -> dict | None:
        """Return record metadata dict."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                """
                SELECT id, title, source_url, license, retrieved_at,
                       pdf_url, pdf_license, source_type
                FROM records WHERE id = ?
                """,
                (record_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "id": row[0],
                "title": row[1],
                "source_url": row[2],
                "license": row[3],
                "retrieved_at": row[4],
                "pdf_url": row[5],
                "pdf_license": row[6],
                "source_type": row[7] or "abstract",
            }

    def store_embedding(
        self,
        record_id: str,
        start_char: int,
        end_char: int,
        text: str,
        model_name: str,
        vector: list[float],
    ) -> None:
        """Store an embedding vector for a passage with text hash for staleness detection."""
        blob = serialize_vector(vector)
        text_hash = compute_text_hash(text)
        now = datetime.now(UTC).isoformat()
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings
                (record_id, start_char, end_char, text_hash,
                 model_name, dimension, vector, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    start_char,
                    end_char,
                    text_hash,
                    model_name,
                    len(vector),
                    blob,
                    now,
                ),
            )
            conn.commit()

    def store_embeddings_batch(
        self,
        items: list[tuple[str, int, int, str, list[float]]],
        model_name: str,
    ) -> None:
        """Store a batch of embeddings: [(record_id, start_char, end_char, text, vector)]."""
        if not items:
            return
        now = datetime.now(UTC).isoformat()
        rows = [
            (
                rec_id,
                start,
                end,
                compute_text_hash(text),
                model_name,
                len(vec),
                serialize_vector(vec),
                now,
            )
            for rec_id, start, end, text, vec in items
        ]
        with self._lock:
            conn = self._get_conn()
            conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings
                (record_id, start_char, end_char, text_hash,
                 model_name, dimension, vector, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()

    def get_embedding(
        self,
        record_id: str,
        start_char: int,
        end_char: int,
        text: str,
        model_name: str,
        expected_dim: int | None = None,
    ) -> list[float] | None:
        """Retrieve the embedding for a passage, verifying it is not stale."""
        expected_hash = compute_text_hash(text)
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                """
                SELECT text_hash, dimension, vector FROM embeddings
                WHERE record_id = ? AND start_char = ? AND end_char = ? AND model_name = ?
                """,
                (record_id, start_char, end_char, model_name),
            ).fetchone()
            if row is None:
                return None
            stored_hash, dim, blob = row
            if stored_hash != expected_hash:
                logger.debug(
                    "Stale embedding detected for %s [%d:%d]", record_id, start_char, end_char
                )
                return None
            return deserialize_vector(blob, expected_dim=expected_dim or dim)

    def get_embeddings_for_passages(
        self,
        passages: list[Passage],
        model_name: str,
        expected_dim: int | None = None,
    ) -> dict[tuple[str, int, int], list[float]]:
        """Retrieve cached embeddings for a list of passages, skipping missing/stale ones."""
        if not passages:
            return {}
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                """
                SELECT record_id, start_char, end_char, text_hash, dimension, vector
                FROM embeddings
                WHERE model_name = ?
                """,
                (model_name,),
            ).fetchall()

        # Index stored rows by (record_id, start_char, end_char)
        stored_map = {(r[0], r[1], r[2]): (r[3], r[4], r[5]) for r in rows}

        result: dict[tuple[str, int, int], list[float]] = {}
        for p in passages:
            key = (p.record_id, p.start_char, p.end_char)
            if key in stored_map:
                stored_hash, dim, blob = stored_map[key]
                if stored_hash == compute_text_hash(p.text):
                    vec = deserialize_vector(blob, expected_dim=expected_dim or dim)
                    if vec is not None:
                        result[key] = vec
        return result

    def count_embeddings(self, model_name: str | None = None) -> int:
        """Count total stored embeddings, optionally filtered by model."""
        with self._lock:
            conn = self._get_conn()
            if model_name:
                row = conn.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE model_name = ?", (model_name,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
            return row[0] if row else 0

    def delete_stale_embeddings(self, model_name: str | None = None) -> int:
        """Delete embeddings whose associated passage no longer exists or whose text changed."""
        with self._lock:
            conn = self._get_conn()
            # 1. Delete orphaned embeddings whose passage no longer exists
            cursor = conn.execute(
                """
                DELETE FROM embeddings
                WHERE NOT EXISTS (
                    SELECT 1 FROM passages p
                    WHERE p.record_id = embeddings.record_id
                      AND p.start_char = embeddings.start_char
                      AND p.end_char = embeddings.end_char
                )
                """
            )
            deleted_count = cursor.rowcount

            # 2. Check remaining embeddings for text hash mismatch
            rows = conn.execute(
                """
                SELECT e.record_id, e.start_char, e.end_char, e.model_name, e.text_hash, p.text
                FROM embeddings e
                JOIN passages p ON e.record_id = p.record_id
                               AND e.start_char = p.start_char
                               AND e.end_char = p.end_char
                """
            ).fetchall()

            for rec_id, start, end, m_name, text_hash, p_text in rows:
                if text_hash != compute_text_hash(p_text):
                    conn.execute(
                        """
                        DELETE FROM embeddings
                        WHERE record_id = ? AND start_char = ? AND end_char = ? AND model_name = ?
                        """,
                        (rec_id, start, end, m_name),
                    )
                    deleted_count += 1

            conn.commit()
            return deleted_count

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
