"""Password hashing and server-side session storage for Silver Jubilee access."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DEFAULT_AUTH_DATABASE_PATH = Path(__file__).resolve().with_name("auth.db")
_USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]{3,64}$")
_SCRYPT_N = 1 << 15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SCRYPT_MAXMEM = 64 * 1024 * 1024
_MIN_PASSWORD_LENGTH = 12


class AuthenticationStoreError(RuntimeError):
    """Raised when an authentication-store operation fails."""


class DuplicateUserError(ValueError):
    """Raised when a username is already provisioned."""


class AdminAlreadyProvisionedError(ValueError):
    """Raised when an initial administrator already exists."""


class UserNotFoundError(LookupError):
    """Raised when a user-management operation targets no regular user."""


class SelfUserModificationError(ValueError):
    """Raised when an administrator targets their own account."""


@dataclass(frozen=True)
class AuthenticatedUser:
    id: int
    username: str
    role: str


@dataclass(frozen=True)
class ManagedUser:
    id: int
    username: str
    role: str
    is_active: bool
    created_at: str


@dataclass(frozen=True)
class StoredSession:
    user: AuthenticatedUser
    csrf_token_hash: str
    expires_at: int


def _validate_username(username: str) -> str:
    if not isinstance(username, str) or not _USERNAME_PATTERN.fullmatch(username):
        raise ValueError("Username must be 3–64 letters, numbers, '.', '_' or '-'.")
    return username


def _validate_password(password: str) -> str:
    if not isinstance(password, str) or not _MIN_PASSWORD_LENGTH <= len(password) <= 1024:
        raise ValueError("Password must contain between 12 and 1024 characters.")
    if "\x00" in password:
        raise ValueError("Password contains an invalid character.")
    return password


def hash_password(password: str) -> str:
    """Hash a password with salted, memory-hard scrypt; never store the input."""
    password = _validate_password(password)
    salt = secrets.token_bytes(16)
    derived_key = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
        maxmem=_SCRYPT_MAXMEM,
    )
    return "$".join(
        (
            "scrypt",
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(derived_key).decode("ascii"),
        )
    )


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password hash safely, returning False for malformed values."""
    if not isinstance(password, str) or not isinstance(password_hash, str):
        return False
    try:
        algorithm, n_text, r_text, p_text, salt_text, expected_text = password_hash.split("$")
        n, r, p = int(n_text), int(r_text), int(p_text)
        if (
            algorithm != "scrypt"
            or n != _SCRYPT_N
            or r != _SCRYPT_R
            or p != _SCRYPT_P
        ):
            return False
        salt = base64.b64decode(salt_text, validate=True)
        expected = base64.b64decode(expected_text, validate=True)
        if (
            len(salt) != 16
            or len(expected) != _SCRYPT_DKLEN
            or base64.b64encode(salt).decode("ascii") != salt_text
            or base64.b64encode(expected).decode("ascii") != expected_text
        ):
            return False
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=len(expected),
            maxmem=_SCRYPT_MAXMEM,
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError, UnicodeEncodeError, binascii.Error):
        return False


_DUMMY_PASSWORD_HASH = hash_password(secrets.token_urlsafe(32))


class AuthenticationStore:
    """Store users and revocable opaque sessions in a separate SQLite database."""

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS secure_info_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
            is_active INTEGER NOT NULL DEFAULT 1 CHECK(is_active IN (0, 1)),
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS secure_info_sessions (
            token_hash TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES secure_info_users(id) ON DELETE CASCADE,
            csrf_token_hash TEXT NOT NULL,
            expires_at INTEGER NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        );

        CREATE INDEX IF NOT EXISTS idx_secure_info_sessions_expiry
            ON secure_info_sessions(expires_at);
    """

    def __init__(self, database_path: str | Path = DEFAULT_AUTH_DATABASE_PATH) -> None:
        self.database_path = Path(database_path)

    @contextmanager
    def _transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.database_path, timeout=5.0)
        except (OSError, sqlite3.Error) as error:
            raise AuthenticationStoreError("The authentication store is unavailable.") from error
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("PRAGMA busy_timeout = 5000")
                if immediate:
                    connection.execute("BEGIN IMMEDIATE")
                yield connection
        except sqlite3.Error as error:
            raise AuthenticationStoreError("The authentication store operation failed.") from error
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._transaction() as connection:
            connection.executescript(self._SCHEMA)
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(secure_info_users)")
            }
            if "is_active" not in columns:
                connection.execute(
                    """
                    ALTER TABLE secure_info_users
                    ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1
                        CHECK(is_active IN (0, 1))
                    """
                )

    @staticmethod
    def _user_from_row(row: sqlite3.Row | None) -> AuthenticatedUser | None:
        if row is None:
            return None
        return AuthenticatedUser(id=row["id"], username=row["username"], role=row["role"])

    @staticmethod
    def _managed_user_from_row(row: sqlite3.Row | None) -> ManagedUser | None:
        if row is None:
            return None
        return ManagedUser(
            id=row["id"],
            username=row["username"],
            role=row["role"],
            is_active=bool(row["is_active"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _validate_user_id(user_id: int) -> int:
        if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id < 1:
            raise ValueError("User ID must be a positive integer.")
        return user_id

    def create_admin(self, username: str, password: str) -> AuthenticatedUser:
        """Create the initial admin exactly once, atomically, without a public API."""
        username = _validate_username(username)
        password_hash = hash_password(password)
        with self._transaction(immediate=True) as connection:
            if connection.execute(
                "SELECT 1 FROM secure_info_users WHERE role = 'admin' LIMIT 1"
            ).fetchone():
                raise AdminAlreadyProvisionedError(
                    "An administrator has already been provisioned."
                )
            try:
                cursor = connection.execute(
                    "INSERT INTO secure_info_users (username, password_hash, role) VALUES (?, ?, 'admin')",
                    (username, password_hash),
                )
            except sqlite3.IntegrityError as error:
                raise DuplicateUserError("That username is already provisioned.") from error
            row = connection.execute(
                "SELECT id, username, role FROM secure_info_users WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
            user = self._user_from_row(row)
            if user is None:
                raise AuthenticationStoreError("The administrator could not be provisioned.")
            return user

    def create_user(self, username: str, password: str) -> AuthenticatedUser:
        """Create a normal user for controlled, local operator provisioning."""
        username = _validate_username(username)
        password_hash = hash_password(password)
        with self._transaction(immediate=True) as connection:
            try:
                cursor = connection.execute(
                    "INSERT INTO secure_info_users (username, password_hash, role) VALUES (?, ?, 'user')",
                    (username, password_hash),
                )
            except sqlite3.IntegrityError as error:
                raise DuplicateUserError("That username is already provisioned.") from error
            row = connection.execute(
                "SELECT id, username, role FROM secure_info_users WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
            user = self._user_from_row(row)
            if user is None:
                raise AuthenticationStoreError("The user could not be provisioned.")
            return user

    def list_users(self) -> list[ManagedUser]:
        """Return regular-user account metadata without authentication secrets."""
        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT id, username, role, is_active, created_at
                FROM secure_info_users
                WHERE role = 'user'
                ORDER BY created_at DESC, id DESC
                """
            ).fetchall()
        return [
            user
            for row in rows
            if (user := self._managed_user_from_row(row)) is not None
        ]

    def get_managed_user(self, user_id: int) -> ManagedUser | None:
        """Return one regular user's allowlisted account metadata."""
        user_id = self._validate_user_id(user_id)
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT id, username, role, is_active, created_at
                FROM secure_info_users
                WHERE id = ? AND role = 'user'
                """,
                (user_id,),
            ).fetchone()
        return self._managed_user_from_row(row)

    def set_user_active(
        self,
        user_id: int,
        is_active: bool,
        administrator_id: int,
    ) -> ManagedUser:
        """Change a regular user's status and revoke sessions when disabling."""
        user_id = self._validate_user_id(user_id)
        administrator_id = self._validate_user_id(administrator_id)
        if user_id == administrator_id:
            raise SelfUserModificationError("Administrators cannot modify their own account.")
        if not isinstance(is_active, bool):
            raise ValueError("Account status must be active or inactive.")

        with self._transaction(immediate=True) as connection:
            cursor = connection.execute(
                """
                UPDATE secure_info_users
                SET is_active = ?
                WHERE id = ? AND role = 'user'
                """,
                (int(is_active), user_id),
            )
            if cursor.rowcount != 1:
                raise UserNotFoundError("Regular user not found.")
            if not is_active:
                connection.execute(
                    "DELETE FROM secure_info_sessions WHERE user_id = ?",
                    (user_id,),
                )
            row = connection.execute(
                """
                SELECT id, username, role, is_active, created_at
                FROM secure_info_users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()
            user = self._managed_user_from_row(row)
            if user is None:
                raise AuthenticationStoreError("The user status could not be updated.")
            return user

    def reset_user_password(
        self,
        user_id: int,
        password: str,
        administrator_id: int,
    ) -> None:
        """Reset a regular user's password and revoke all of their sessions."""
        user_id = self._validate_user_id(user_id)
        administrator_id = self._validate_user_id(administrator_id)
        if user_id == administrator_id:
            raise SelfUserModificationError("Administrators cannot modify their own account.")
        password_hash = hash_password(password)
        with self._transaction(immediate=True) as connection:
            cursor = connection.execute(
                """
                UPDATE secure_info_users
                SET password_hash = ?
                WHERE id = ? AND role = 'user'
                """,
                (password_hash, user_id),
            )
            if cursor.rowcount != 1:
                raise UserNotFoundError("Regular user not found.")
            connection.execute(
                "DELETE FROM secure_info_sessions WHERE user_id = ?",
                (user_id,),
            )

    def verify_credentials(self, username: str, password: str) -> AuthenticatedUser | None:
        """Authenticate credentials with a generic failure result."""
        if not isinstance(username, str) or not isinstance(password, str):
            verify_password("", _DUMMY_PASSWORD_HASH)
            return None
        try:
            with self._transaction() as connection:
                row = connection.execute(
                    """
                    SELECT id, username, role, password_hash, is_active
                    FROM secure_info_users
                    WHERE username = ?
                    """,
                    (username,),
                ).fetchone()
        except AuthenticationStoreError:
            raise

        if row is None:
            verify_password(password, _DUMMY_PASSWORD_HASH)
            return None
        password_matches = verify_password(password, row["password_hash"])
        if not password_matches or not row["is_active"]:
            return None
        return self._user_from_row(row)

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("ascii")).hexdigest()

    def create_session(
        self,
        user_id: int,
        ttl_seconds: int,
    ) -> tuple[str, str, int]:
        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int) or ttl_seconds < 1:
            raise ValueError("Session lifetime must be a positive number of seconds.")
        token = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(32)
        expires_at = int(time.time()) + ttl_seconds
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO secure_info_sessions
                    (token_hash, user_id, csrf_token_hash, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    self._token_hash(token),
                    user_id,
                    self._token_hash(csrf_token),
                    expires_at,
                ),
            )
        return token, csrf_token, expires_at

    def get_session(self, token: str) -> StoredSession | None:
        if not isinstance(token, str) or not token or len(token) > 256:
            return None
        try:
            token_hash = self._token_hash(token)
        except UnicodeEncodeError:
            return None
        now = int(time.time())
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT users.id, users.username, users.role,
                       sessions.csrf_token_hash, sessions.expires_at
                FROM secure_info_sessions AS sessions
                JOIN secure_info_users AS users ON users.id = sessions.user_id
                WHERE sessions.token_hash = ? AND users.is_active = 1
                """,
                (token_hash,),
            ).fetchone()
            if row is None:
                return None
            if row["expires_at"] <= now:
                connection.execute(
                    "DELETE FROM secure_info_sessions WHERE token_hash = ?",
                    (token_hash,),
                )
                return None
        user = self._user_from_row(row)
        if user is None:
            return None
        return StoredSession(
            user=user,
            csrf_token_hash=row["csrf_token_hash"],
            expires_at=row["expires_at"],
        )

    def revoke_session(self, token: str) -> None:
        if not isinstance(token, str) or not token or len(token) > 256:
            return
        try:
            token_hash = self._token_hash(token)
        except UnicodeEncodeError:
            return
        with self._transaction() as connection:
            connection.execute(
                "DELETE FROM secure_info_sessions WHERE token_hash = ?",
                (token_hash,),
            )
