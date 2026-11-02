"""
metadata_db.py — SQLite database for mapping FAISS vector IDs to image regions.

Schema
------
parent_images
    id        INTEGER PRIMARY KEY AUTOINCREMENT
    path      TEXT UNIQUE NOT NULL          -- absolute path to the image file
    file_hash TEXT NOT NULL                 -- SHA-256 hex digest (dedup key)

region_vectors
    vector_id       INTEGER PRIMARY KEY     -- FAISS row index (0-based)
    parent_image_id INTEGER NOT NULL        -- FK → parent_images.id
    parent_path     TEXT    NOT NULL        -- denormalised for fast lookup
    is_whole_image  INTEGER NOT NULL        -- 1 if whole image, 0 if crop
    x1 REAL, y1 REAL, x2 REAL, y2 REAL    -- bbox (NULL for whole image)
    class_label     TEXT                    -- YOLO class or "whole"/"grid_r_c"
    confidence      REAL                    -- YOLO conf (0.0 for whole/grid)
    crop_type       TEXT                    -- "whole" | "yolo" | "grid"

Usage
-----
>>> db = RegionMetadataDB(path)
>>> with db:
...     img_id = db.add_parent_image(path, sha256)
...     db.add_region_vector(vector_id, img_id, parent_path, box)
...     row = db.get_vector(vector_id)
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from visionsearch_web_ui.region.region_splitter import RegionBox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    """Return the hex-encoded SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------


class RegionMetadataDB:
    """Thin SQLite wrapper for region vector metadata.

    Supports use as a context manager (opens/closes the connection) and
    also keeps an optional persistent connection for long-lived processes.
    """

    _DDL = """
    CREATE TABLE IF NOT EXISTS parent_images (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        path      TEXT UNIQUE NOT NULL,
        file_hash TEXT NOT NULL,
        image_width INTEGER,
        image_height INTEGER,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    );

    CREATE TABLE IF NOT EXISTS region_vectors (
        vector_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        parent_image_id INTEGER NOT NULL REFERENCES parent_images(id),
        parent_path     TEXT    NOT NULL,
        is_whole_image  INTEGER NOT NULL,
        index_type TEXT NOT NULL DEFAULT 'region',
        faiss_id INTEGER,
        file_hash TEXT,
        image_width INTEGER,
        image_height INTEGER,
        x1   REAL,
        y1   REAL,
        x2   REAL,
        y2   REAL,
        class_label TEXT,
        confidence  REAL,
        crop_type   TEXT,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
    );

    CREATE INDEX IF NOT EXISTS idx_rv_parent
        ON region_vectors(parent_image_id);
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "RegionMetadataDB":
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        if _ and _[0] is not None:
            if self._conn is not None:
                self._conn.rollback()
                self._conn.close()
                self._conn = None
            return
        self.close()

    def open(self) -> None:
        """Open (and migrate) the database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.executescript(self._DDL)
            self._migrate_schema()
            self._conn.commit()

    def _migrate_schema(self) -> None:
        """Add metadata/index mapping columns to databases from earlier builds."""
        conn = self.conn
        parent_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(parent_images)")
        }
        for name, sql_type in (
            ("image_width", "INTEGER"),
            ("image_height", "INTEGER"),
            ("created_at", "TEXT"),
        ):
            if name not in parent_columns:
                conn.execute(f"ALTER TABLE parent_images ADD COLUMN {name} {sql_type}")
        vector_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(region_vectors)")
        }
        for name, sql_type in (
            ("index_type", "TEXT NOT NULL DEFAULT 'region'"),
            ("faiss_id", "INTEGER"),
            ("file_hash", "TEXT"),
            ("image_width", "INTEGER"),
            ("image_height", "INTEGER"),
            ("created_at", "TEXT"),
        ):
            if name not in vector_columns:
                conn.execute(f"ALTER TABLE region_vectors ADD COLUMN {name} {sql_type}")
        conn.execute(
            """UPDATE region_vectors
               SET index_type = CASE WHEN is_whole_image = 1 THEN 'whole' ELSE 'region' END
               WHERE faiss_id IS NULL"""
        )
        conn.execute("UPDATE region_vectors SET faiss_id = vector_id WHERE faiss_id IS NULL")
        conn.execute(
            """UPDATE region_vectors
               SET file_hash = (
                   SELECT file_hash FROM parent_images
                   WHERE parent_images.id = region_vectors.parent_image_id
               )
               WHERE file_hash IS NULL"""
        )
        conn.execute(
            """UPDATE region_vectors
               SET image_width = (
                       SELECT image_width FROM parent_images
                       WHERE parent_images.id = region_vectors.parent_image_id
                   ),
                   image_height = (
                       SELECT image_height FROM parent_images
                       WHERE parent_images.id = region_vectors.parent_image_id
                   )
               WHERE image_width IS NULL OR image_height IS NULL"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rv_faiss_lookup ON region_vectors(index_type, faiss_id)"
        )

    def close(self) -> None:
        """Commit and close the database connection."""
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not open. Call open() or use as context manager.")
        return self._conn

    # ------------------------------------------------------------------
    # Parent images
    # ------------------------------------------------------------------

    def add_parent_image(
        self,
        path: str | Path,
        file_hash: str,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> int:
        """Insert a parent image row and return its integer id.

        If a row with the same *path* already exists, return its existing id
        without modifying the row.
        """
        path_str = str(Path(path).resolve())
        self.conn.execute(
            """INSERT OR IGNORE INTO parent_images
               (path, file_hash, image_width, image_height)
               VALUES (?, ?, ?, ?)""",
            (path_str, file_hash, image_width, image_height),
        )
        row = self.conn.execute(
            "SELECT id FROM parent_images WHERE path = ?", (path_str,)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Could not retrieve metadata for indexed image: {path_str}")
        if image_width is not None and image_height is not None:
            self.conn.execute(
                "UPDATE parent_images SET image_width = ?, image_height = ? WHERE id = ?",
                (image_width, image_height, int(row["id"])),
            )
        return int(row["id"])

    def get_indexed_hashes(self) -> set[str]:
        """Return the set of SHA-256 digests for already-indexed images."""
        rows = self.conn.execute("SELECT file_hash FROM parent_images").fetchall()
        return {row["file_hash"] for row in rows}

    def get_yolo_labels(self) -> set[str]:
        """Return distinct labels from YOLO crop metadata, excluding grid tiles."""
        rows = self.conn.execute(
            """SELECT DISTINCT class_label
               FROM region_vectors
               WHERE index_type = 'region'
                 AND crop_type = 'yolo'
                 AND class_label IS NOT NULL
                 AND TRIM(class_label) != ''"""
        ).fetchall()
        return {str(row["class_label"]) for row in rows}

    def get_next_vector_id(self) -> int:
        """Return the next globally unique SQLite vector ID."""
        row = self.conn.execute(
            "SELECT COALESCE(MAX(vector_id), -1) + 1 FROM region_vectors"
        ).fetchone()
        return int(row[0])

    def get_next_faiss_id(self, index_type: str) -> int:
        """Return the next row ID for a named FAISS index."""
        if index_type not in ("whole", "region"):
            raise ValueError("index_type must be 'whole' or 'region'")
        row = self.conn.execute(
            "SELECT COALESCE(MAX(faiss_id), -1) + 1 FROM region_vectors WHERE index_type = ?",
            (index_type,),
        ).fetchone()
        return int(row[0])

    def get_parent_by_path(self, path: str | Path) -> dict | None:
        """Return the parent_images row for *path*, or None."""
        path_str = str(Path(path).resolve())
        row = self.conn.execute(
            "SELECT * FROM parent_images WHERE path = ?", (path_str,)
        ).fetchone()
        return dict(row) if row else None

    def get_vector_by_faiss_id(self, index_type: str, faiss_id: int) -> dict | None:
        """Resolve a local FAISS row ID to its stable SQLite vector metadata."""
        row = self.conn.execute(
            "SELECT * FROM region_vectors WHERE index_type = ? AND faiss_id = ?",
            (index_type, faiss_id),
        ).fetchone()
        return dict(row) if row else None

    def get_vector_counts(self) -> dict[str, int]:
        """Return counts in metadata grouped by their corresponding FAISS index."""
        rows = self.conn.execute(
            "SELECT index_type, COUNT(*) AS count FROM region_vectors GROUP BY index_type"
        ).fetchall()
        counts = {"whole": 0, "region": 0}
        counts.update({row["index_type"]: int(row["count"]) for row in rows})
        return counts

    # ------------------------------------------------------------------
    # Region vectors
    # ------------------------------------------------------------------

    def add_region_vector(
        self,
        vector_id: int | None,
        parent_image_id: int,
        parent_path: str | Path,
        box: RegionBox | None,
        *,
        is_whole_image: bool = False,
        index_type: str | None = None,
        faiss_id: int | None = None,
        file_hash: str | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> None:
        """Insert a region vector row.

        Parameters
        ----------
        vector_id : int
            FAISS row index (0-based, must be unique).
        parent_image_id : int
            Row id in ``parent_images``.
        parent_path : str | Path
            Absolute path to the parent image file.
        box : RegionBox | None
            Region geometry; ``None`` for whole-image vectors.
        is_whole_image : bool
            ``True`` for whole-image vectors (``box`` should be None).
        """
        kind = index_type or ("whole" if is_whole_image else "region")
        if kind not in ("whole", "region"):
            raise ValueError("index_type must be 'whole' or 'region'")
        if is_whole_image != (kind == "whole"):
            raise ValueError("is_whole_image must agree with index_type")
        if vector_id is None:
            vector_id = self.get_next_vector_id()
        if faiss_id is None:
            faiss_id = self.get_next_faiss_id(kind)
        if box is None and not is_whole_image:
            raise ValueError("region vectors require a bounding box")
        if box is not None and is_whole_image:
            raise ValueError("whole-image vectors cannot have a bounding box")
        self.conn.execute(
            """INSERT INTO region_vectors
               (vector_id, parent_image_id, parent_path, is_whole_image,
                index_type, faiss_id, file_hash, image_width, image_height,
                x1, y1, x2, y2, class_label, confidence, crop_type)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                vector_id,
                parent_image_id,
                str(Path(parent_path).resolve()),
                int(is_whole_image),
                kind,
                faiss_id,
                file_hash,
                image_width,
                image_height,
                None if box is None else box.x1,
                None if box is None else box.y1,
                None if box is None else box.x2,
                None if box is None else box.y2,
                "whole" if box is None else box.class_label,
                0.0 if box is None else box.confidence,
                "whole" if box is None else box.crop_type,
            ),
        )

    def get_vector(self, vector_id: int) -> dict | None:
        """Return the metadata dict for a given FAISS vector id, or None."""
        row = self.conn.execute(
            "SELECT * FROM region_vectors WHERE vector_id = ?", (vector_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_vectors_for_parent(self, parent_image_id: int) -> list[dict]:
        """Return all region vector rows for a given parent image."""
        rows = self.conn.execute(
            "SELECT * FROM region_vectors WHERE parent_image_id = ?",
            (parent_image_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return a summary dict for the /index/stats endpoint."""
        n_parents = self.conn.execute(
            "SELECT COUNT(*) FROM parent_images"
        ).fetchone()[0]
        n_region_vectors = self.conn.execute(
            "SELECT COUNT(*) FROM region_vectors WHERE index_type = 'region'"
        ).fetchone()[0]
        n_vectors = self.conn.execute(
            "SELECT COUNT(*) FROM region_vectors"
        ).fetchone()[0]
        n_whole = self.conn.execute(
            "SELECT COUNT(*) FROM region_vectors WHERE is_whole_image = 1"
        ).fetchone()[0]
        n_yolo = self.conn.execute(
            "SELECT COUNT(*) FROM region_vectors WHERE crop_type = 'yolo'"
        ).fetchone()[0]
        n_grid = self.conn.execute(
            "SELECT COUNT(*) FROM region_vectors WHERE crop_type = 'grid'"
        ).fetchone()[0]
        db_mb = self._path.stat().st_size / (1024 * 1024) if self._path.exists() else 0.0
        return {
            "indexed_images": n_parents,
            "total_vectors": n_vectors,
            "region_vectors": n_region_vectors,
            "whole_image_vectors": n_whole,
            "yolo_region_vectors": n_yolo,
            "grid_region_vectors": n_grid,
            "db_size_mb": round(db_mb, 3),
        }
