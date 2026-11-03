"""Application service for encrypted Silver Jubilee information records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import TypeAlias

from visionsearch_web_ui.secure_info.database import (
    SilverJubileeDatabase,
    SilverJubileeRecord,
)
from visionsearch_web_ui.secure_info.auth import AuthenticatedUser
from visionsearch_web_ui.secure_info.encryption import (
    EncryptionConfigurationError,
    EncryptionError,
    decrypt_information,
    encrypt_information,
)

_IMAGE_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")
_MAX_REFERENCE_LENGTH = 4096
_MAX_TITLE_LENGTH = 200
_DEFAULT_LIST_LIMIT = 100
_MAX_LIST_LIMIT = 100


@dataclass(frozen=True)
class RegistrationMetadata:
    """Administrative listing data with no encrypted or plaintext payload."""

    id: int
    image_path: str
    image_hash: str
    title: str
    created_at: datetime
    updated_at: datetime


RetrievedInformationRecord: TypeAlias = SilverJubileeRecord


@dataclass(frozen=True)
class AuthorizedSilverJubileeInformation:
    record_id: int
    title: str
    information: str


class SecureInformationRetrievalError(RuntimeError):
    """Raised when a registered information payload cannot be safely decrypted."""


class SilverJubileeService:
    """Validate, encrypt, and persist Silver Jubilee information."""

    def __init__(self, database: SilverJubileeDatabase) -> None:
        self._database = database

    @staticmethod
    def _validate_image_reference(image_reference: str) -> str:
        if not isinstance(image_reference, str):
            raise ValueError("Photograph reference must be a string.")
        value = image_reference.strip()
        normalized = value.replace("\\", "/")
        components = normalized.split("/")
        if (
            not value
            or len(value) > _MAX_REFERENCE_LENGTH
            or "\x00" in value
            or any(ord(character) < 32 for character in value)
            or normalized.startswith("/")
            or _WINDOWS_DRIVE_PATTERN.match(normalized)
            or any(component in {"", ".", ".."} for component in components)
        ):
            raise ValueError("Photograph reference must be a safe relative path or identifier.")
        return normalized

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
            raise ValueError("Title must contain between 1 and 200 characters.")
        return value

    @staticmethod
    def _validate_information(information: str) -> str:
        if not isinstance(information, str) or not information.strip():
            raise ValueError("Information must be a non-empty string.")
        return information

    @staticmethod
    def _metadata(record: SilverJubileeRecord) -> RegistrationMetadata:
        return RegistrationMetadata(
            id=record.id,
            image_path=record.image_path,
            image_hash=record.image_hash,
            title=record.title,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    def register_information(
        self,
        image_reference: str,
        image_hash: str,
        title: str,
        information: str,
    ) -> RegistrationMetadata:
        """Validate and encrypt information before storing a photograph registration."""
        image_reference = self._validate_image_reference(image_reference)
        image_hash = self._validate_image_hash(image_hash)
        title = self._validate_title(title)
        information = self._validate_information(information)
        encrypted = encrypt_information(information)
        record = self._database.register_encrypted_record(
            image_path=image_reference,
            image_hash=image_hash,
            title=title,
            ciphertext=encrypted["ciphertext"],
            nonce=encrypted["nonce"],
        )
        return self._metadata(record)

    def retrieve_information(
        self,
        image_hash: str,
    ) -> RetrievedInformationRecord | None:
        """Find a registered photograph and return metadata plus ciphertext only."""
        image_hash = self._validate_image_hash(image_hash)
        return self._database.find_record_by_image_hash(image_hash)

    def retrieve_authorized_information(
        self,
        image_hash: str,
        user: AuthenticatedUser,
        expected_image_reference: str | None = None,
    ) -> AuthorizedSilverJubileeInformation | None:
        """Decrypt only for a valid authenticated role after a registration match."""
        if (
            not isinstance(user, AuthenticatedUser)
            or isinstance(user.id, bool)
            or not isinstance(user.id, int)
            or user.id < 1
            or user.role not in {"user", "admin"}
        ):
            raise PermissionError("An authenticated account is required.")

        record = self.retrieve_information(image_hash)
        if record is None:
            return None
        if expected_image_reference is not None:
            expected_image_reference = self._validate_image_reference(
                expected_image_reference
            )
            if record.image_path != expected_image_reference:
                return None

        try:
            information = decrypt_information(
                {"nonce": record.nonce, "ciphertext": record.ciphertext}
            )
        except (EncryptionConfigurationError, EncryptionError) as error:
            raise SecureInformationRetrievalError(
                "Registered information could not be securely retrieved."
            ) from error
        return AuthorizedSilverJubileeInformation(
            record_id=record.id,
            title=record.title,
            information=information,
        )

    def update_information(
        self,
        record_id: int,
        title: str,
        information: str,
    ) -> RegistrationMetadata:
        """Encrypt replacement information with a fresh nonce and update atomically."""
        if isinstance(record_id, bool) or not isinstance(record_id, int) or record_id < 1:
            raise ValueError("Record ID must be a positive integer.")
        title = self._validate_title(title)
        information = self._validate_information(information)
        encrypted = encrypt_information(information)
        record = self._database.update_encrypted_record(
            record_id=record_id,
            title=title,
            ciphertext=encrypted["ciphertext"],
            nonce=encrypted["nonce"],
        )
        return self._metadata(record)

    def delete_registration(self, record_id: int) -> bool:
        """Delete a registration by its database record ID."""
        if isinstance(record_id, bool) or not isinstance(record_id, int) or record_id < 1:
            raise ValueError("Record ID must be a positive integer.")
        return self._database.delete_record(record_id)

    def list_registrations(
        self,
        limit: int = _DEFAULT_LIST_LIMIT,
        offset: int = 0,
    ) -> list[RegistrationMetadata]:
        """Return paginated administrative metadata, excluding all payload fields."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_LIST_LIMIT:
            raise ValueError(f"Limit must be an integer between 1 and {_MAX_LIST_LIMIT}.")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("Offset must be a non-negative integer.")
        return [
            self._metadata(record)
            for record in self._database.list_records(limit=limit, offset=offset)
        ]
