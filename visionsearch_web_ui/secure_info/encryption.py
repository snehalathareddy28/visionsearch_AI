"""AES-256-GCM helpers for Silver Jubilee information."""

from __future__ import annotations

import base64
import binascii
import os
from pathlib import Path
from typing import Mapping, TypedDict

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_KEY_ENVIRONMENT_VARIABLE = "VISIONSEARCH_AES_KEY"
_NONCE_LENGTH = 12
_KEY_LENGTH = 32

load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)


class EncryptedInformation(TypedDict):
    """Base64-encoded AES-GCM nonce and authenticated ciphertext."""

    nonce: str
    ciphertext: str


class EncryptionError(ValueError):
    """Raised when information cannot be encrypted or safely decrypted."""


class EncryptionConfigurationError(RuntimeError):
    """Raised when the encryption key is missing or invalid."""


def generate_encryption_key() -> str:
    """Explicitly generate a Base64-encoded 256-bit key for environment setup."""
    return base64.b64encode(AESGCM.generate_key(bit_length=256)).decode("ascii")


def get_encryption_key() -> bytes:
    """Load and validate the Base64-encoded 32-byte key from the environment."""
    encoded_key = os.environ.get(_KEY_ENVIRONMENT_VARIABLE)
    if not encoded_key:
        raise EncryptionConfigurationError(
            f"{_KEY_ENVIRONMENT_VARIABLE} must be configured."
        )

    try:
        key = base64.b64decode(encoded_key, validate=True)
    except (binascii.Error, ValueError, UnicodeEncodeError) as error:
        raise EncryptionConfigurationError(
            f"{_KEY_ENVIRONMENT_VARIABLE} must be valid Base64 encoding of exactly 32 bytes."
        ) from error

    if (
        len(key) != _KEY_LENGTH
        or base64.b64encode(key).decode("ascii") != encoded_key
    ):
        raise EncryptionConfigurationError(
            f"{_KEY_ENVIRONMENT_VARIABLE} must decode to exactly 32 bytes."
        )
    return key


def _decode_base64(value: object) -> bytes:
    if not isinstance(value, str) or not value:
        raise EncryptionError("Encrypted information is invalid or could not be authenticated.")

    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError, UnicodeEncodeError) as error:
        raise EncryptionError(
            "Encrypted information is invalid or could not be authenticated."
        ) from error

    if base64.b64encode(decoded).decode("ascii") != value:
        raise EncryptionError(
            "Encrypted information is invalid or could not be authenticated."
        )
    return decoded


def encrypt_information(plaintext: str) -> EncryptedInformation:
    """Encrypt non-empty UTF-8 text with AES-256-GCM and a fresh 12-byte nonce."""
    if not isinstance(plaintext, str) or not plaintext.strip():
        raise EncryptionError("Plaintext must be a non-empty string.")

    try:
        plaintext_bytes = plaintext.encode("utf-8")
    except UnicodeEncodeError as error:
        raise EncryptionError("Plaintext must be valid UTF-8 text.") from error

    key = get_encryption_key()
    nonce = os.urandom(_NONCE_LENGTH)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext_bytes, None)
    return {
        "nonce": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
    }


def decrypt_information(encrypted_data: Mapping[str, str]) -> str:
    """Authenticate and decrypt a Base64 nonce/ciphertext pair."""
    if not isinstance(encrypted_data, Mapping):
        raise EncryptionError("Encrypted information is invalid or could not be authenticated.")

    nonce = _decode_base64(encrypted_data.get("nonce"))
    ciphertext = _decode_base64(encrypted_data.get("ciphertext"))
    if len(nonce) != _NONCE_LENGTH or len(ciphertext) < 16:
        raise EncryptionError("Encrypted information is invalid or could not be authenticated.")

    key = get_encryption_key()
    try:
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
        return plaintext.decode("utf-8")
    except (InvalidTag, UnicodeDecodeError) as error:
        raise EncryptionError(
            "Encrypted information is invalid or could not be authenticated."
        ) from error
