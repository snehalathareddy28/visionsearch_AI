"""SQLite persistence for encrypted Silver Jubilee information records."""

from __future__ import annotations

import base64
import binascii
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from visionsearch_web_ui.secure_info.encryption import encrypt_information

DEFAULT_DATABASE_PATH = Path(__file__).resolve().with_name("silver_jubilee.db")
_IMAGE_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAX_IMAGE_PATH_LENGTH = 4096
_MAX_TITLE_LENGTH = 200
_MAX_LIST_LIMIT = 1000


class SecureInfoDatabaseError(RuntimeError):
    """Raised when a database operation fails."""


class DuplicatePhotographError(ValueError):
    """Raised when a photograph path or image hash is already registered."""


class RecordNotFoundError(LookupError):
    """Raised when an update targets a record that does not exist."""


@dataclass(frozen=True)
class SilverJubileeRecord:
    """A stored record; information remains encrypted in this representation."""

    id: int
    image_path: str
    image_hash: str
    title: str
    ciphertext: str
    nonce: str
    created_at: datetime
    updated_at: datetime


class SilverJubileeDatabase:
    """Separate SQLite store for Silver Jubilee photograph records."""

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS silver_jubilee_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL UNIQUE
                CHECK(length(image_path) BETWEEN 1 AND 4096),
            image_hash TEXT NOT NULL UNIQUE
                CHECK(length(image_hash) = 64 AND image_hash NOT GLOB '*[^0-9a-f]*'),
            title TEXT NOT NULL
                CHECK(length(trim(title)) BETWEEN 1 AND 200),
            ciphertext TEXT NOT NULL CHECK(length(ciphertext) > 0),
            nonce TEXT NOT NULL CHECK(length(nonce) > 0),
            created_at TEXT NOT NULL
                DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at TEXT NOT NULL
                DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        CREATE INDEX IF NOT EXISTS idx_silver_jubilee_records_created_at
            ON silver_jubilee_records(created_at DESC, id DESC);
    """

    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.database_path, timeout=5.0)
        except (OSError, sqlite3.Error) as error:
            raise SecureInfoDatabaseError("The secure information database is unavailable.") from error

        connection.row_factory = sqlite3.Row
        try:
            with connection:
                connection.execute("PRAGMA busy_timeout = 5000")
                yield connection
        except sqlite3.Error as error:
            raise SecureInfoDatabaseError("The secure information database operation failed.") from error
        finally:
            connection.close()

    @staticmethod
    def _validate_image_path(image_path: str) -> str:
        if not isinstance(image_path, str):
            raise ValueError("Image path or identifier must be a string.")
        value = image_path.strip()
        if (
            not value
            or len(value) > _MAX_IMAGE_PATH_LENGTH
            or "\x00" in value
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("Image path or identifier is invalid.")

        path_parts = value.replace("\\", "/").split("/")
        if ".." in path_parts:
            raise ValueError("Image path must not contain parent-directory traversal.")
        return value

    @staticmethod
    def _validate_image_hash(image_hash: str) -> str:
        if not isinstance(image_hash, str) or not _IMAGE_HASH_PATTERN.fullmatch(image_hash):
            raise ValueError("Image hash must be a lowercase SHA-256 hex digest.")
        return image_hash

    @staticmethod
    def _validate_title(title: str) -> str:
        if not isinstance(title, str):
            raise ValueError("Title must be a string.")
        value = title.strip()
        if not value or len(value) > _MAX_TITLE_LENGTH:
            raise ValueError(f"Title must contain between 1 and {_MAX_TITLE_LENGTH} characters.")
        return value

    @staticmethod
    def _record_from_row(row: sqlite3.Row | None) -> SilverJubileeRecord | None:
        if row is None:
            return None
        return SilverJubileeRecord(
            id=row["id"],
            image_path=row["image_path"],
            image_hash=row["image_hash"],
            title=row["title"],
            ciphertext=row["ciphertext"],
            nonce=row["nonce"],
            created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00")),
        )

    @staticmethod
    def _validate_record_id(record_id: int) -> int:
        if isinstance(record_id, bool) or not isinstance(record_id, int) or record_id < 1:
            raise ValueError("Record ID must be a positive integer.")
        return record_id

    def initialize(self) -> None:
        """Create the dedicated table and its listing index if they do not exist."""
        with self._transaction() as connection:
            connection.executescript(self._SCHEMA)

    def register_record(
        self,
        image_path: str,
        image_hash: str,
        title: str,
        information: str,
    ) -> SilverJubileeRecord:
        """Encrypt information and register a unique photograph record."""
        image_path = self._validate_image_path(image_path)
        image_hash = self._validate_image_hash(image_hash)
        title = self._validate_title(title)
        encrypted = encrypt_information(information)
        return self.register_encrypted_record(
            image_path=image_path,
            image_hash=image_hash,
            title=title,
            ciphertext=encrypted["ciphertext"],
            nonce=encrypted["nonce"],
        )

    @staticmethod
    def _validate_encrypted_payload(ciphertext: str, nonce: str) -> tuple[str, str]:
        if not isinstance(ciphertext, str) or not isinstance(nonce, str):
            raise ValueError("Encrypted information payload is invalid.")
        try:
            ciphertext_bytes = base64.b64decode(ciphertext, validate=True)
            nonce_bytes = base64.b64decode(nonce, validate=True)
        except (binascii.Error, ValueError, UnicodeEncodeError) as error:
            raise ValueError("Encrypted information payload is invalid.") from error
        if (
            not ciphertext_bytes
            or base64.b64encode(ciphertext_bytes).decode("ascii") != ciphertext
            or len(nonce_bytes) != 12
            or base64.b64encode(nonce_bytes).decode("ascii") != nonce
            or len(ciphertext_bytes) < 16
        ):
            raise ValueError("Encrypted information payload is invalid.")
        return ciphertext, nonce

    def register_encrypted_record(
        self,
        image_path: str,
        image_hash: str,
        title: str,
        ciphertext: str,
        nonce: str,
    ) -> SilverJubileeRecord:
        """Store an already-encrypted payload without receiving plaintext."""
        image_path = self._validate_image_path(image_path)
        image_hash = self._validate_image_hash(image_hash)
        title = self._validate_title(title)
        ciphertext, nonce = self._validate_encrypted_payload(ciphertext, nonce)

        with self._transaction() as connection:
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO silver_jubilee_records
                        (image_path, image_hash, title, ciphertext, nonce)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        image_path,
                        image_hash,
                        title,
                        ciphertext,
                        nonce,
                    ),
                )
            except sqlite3.IntegrityError as error:
                if "UNIQUE constraint failed" in str(error):
                    raise DuplicatePhotographError(
                        "A photograph with this path or image hash is already registered."
                    ) from error
                raise SecureInfoDatabaseError(
                    "The secure information record could not be registered."
                ) from error

            row = connection.execute(
                "SELECT * FROM silver_jubilee_records WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
            record = self._record_from_row(row)
            if record is None:
                raise SecureInfoDatabaseError("The registered record could not be read.")
            return record

    def get_record_by_id(self, record_id: int) -> SilverJubileeRecord | None:
        """Retrieve an encrypted record by its primary key."""
        record_id = self._validate_record_id(record_id)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM silver_jubilee_records WHERE id = ?",
                (record_id,),
            ).fetchone()
        return self._record_from_row(row)

    def find_record_by_image_hash(self, image_hash: str) -> SilverJubileeRecord | None:
        """Find an encrypted record using its SHA-256 image digest."""
        image_hash = self._validate_image_hash(image_hash)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM silver_jubilee_records WHERE image_hash = ?",
                (image_hash,),
            ).fetchone()
        return self._record_from_row(row)

    def update_record(
        self,
        record_id: int,
        title: str,
        information: str,
    ) -> SilverJubileeRecord:
        """Replace a record's title and encrypted information atomically."""
        record_id = self._validate_record_id(record_id)
        title = self._validate_title(title)
        encrypted = encrypt_information(information)
        return self.update_encrypted_record(
            record_id=record_id,
            title=title,
            ciphertext=encrypted["ciphertext"],
            nonce=encrypted["nonce"],
        )

    def update_encrypted_record(
        self,
        record_id: int,
        title: str,
        ciphertext: str,
        nonce: str,
    ) -> SilverJubileeRecord:
        """Replace title and encrypted payload atomically without plaintext."""
        record_id = self._validate_record_id(record_id)
        title = self._validate_title(title)
        ciphertext, nonce = self._validate_encrypted_payload(ciphertext, nonce)

        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE silver_jubilee_records
                SET title = ?,
                    ciphertext = ?,
                    nonce = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = ?
                """,
                (title, ciphertext, nonce, record_id),
            )
            if cursor.rowcount == 0:
                raise RecordNotFoundError(f"Silver Jubilee record {record_id} was not found.")
            row = connection.execute(
                "SELECT * FROM silver_jubilee_records WHERE id = ?",
                (record_id,),
            ).fetchone()
            record = self._record_from_row(row)
            if record is None:
                raise SecureInfoDatabaseError("The updated record could not be read.")
            return record

    def delete_record(self, record_id: int) -> bool:
        """Delete a record and return whether it existed."""
        record_id = self._validate_record_id(record_id)
        with self._transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM silver_jubilee_records WHERE id = ?",
                (record_id,),
            )
            return cursor.rowcount > 0

    def list_records(self, limit: int = 100, offset: int = 0) -> list[SilverJubileeRecord]:
        """List encrypted record metadata, newest first, with bounded pagination."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_LIST_LIMIT:
            raise ValueError(f"Limit must be an integer between 1 and {_MAX_LIST_LIMIT}.")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("Offset must be a non-negative integer.")

        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT * FROM silver_jubilee_records
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [self._record_from_row(row) for row in rows if row is not None]
