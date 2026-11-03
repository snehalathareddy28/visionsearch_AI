import base64
import os
import unittest
from unittest.mock import patch

from visionsearch_web_ui.secure_info.encryption import (
    EncryptionConfigurationError,
    EncryptionError,
    decrypt_information,
    encrypt_information,
    generate_encryption_key,
    get_encryption_key,
)


TEST_KEY = base64.b64encode(b"visionsearch-test-key-32-bytes!!").decode("ascii")


class SecureInfoEncryptionTests(unittest.TestCase):
    def test_round_trip(self):
        plaintext = "Silver Jubilee history and milestones"
        with patch.dict(os.environ, {"VISIONSEARCH_AES_KEY": TEST_KEY}):
            encrypted = encrypt_information(plaintext)
            self.assertEqual(decrypt_information(encrypted), plaintext)

    def test_each_encryption_uses_a_different_nonce(self):
        with patch.dict(os.environ, {"VISIONSEARCH_AES_KEY": TEST_KEY}):
            first = encrypt_information("same information")
            second = encrypt_information("same information")
        self.assertNotEqual(first["nonce"], second["nonce"])

    def test_explicit_key_generation_returns_a_32_byte_key(self):
        generated_key = generate_encryption_key()
        self.assertEqual(len(base64.b64decode(generated_key, validate=True)), 32)

    def test_missing_key_is_rejected_without_generating_a_replacement(self):
        with patch.dict(os.environ):
            os.environ.pop("VISIONSEARCH_AES_KEY", None)
            with self.assertRaises(EncryptionConfigurationError):
                get_encryption_key()

    def test_invalid_key_base64_is_rejected(self):
        with patch.dict(os.environ, {"VISIONSEARCH_AES_KEY": "not-base64!"}):
            with self.assertRaises(EncryptionConfigurationError):
                get_encryption_key()

    def test_key_with_wrong_decoded_length_is_rejected(self):
        short_key = base64.b64encode(b"too short").decode("ascii")
        with patch.dict(os.environ, {"VISIONSEARCH_AES_KEY": short_key}):
            with self.assertRaises(EncryptionConfigurationError):
                get_encryption_key()

    def test_incorrect_key_is_rejected(self):
        wrong_key = base64.b64encode(b"different-key-which-is-32-bytes!").decode("ascii")
        with patch.dict(os.environ, {"VISIONSEARCH_AES_KEY": TEST_KEY}):
            encrypted = encrypt_information("private information")
        with patch.dict(os.environ, {"VISIONSEARCH_AES_KEY": wrong_key}):
            with self.assertRaises(EncryptionError):
                decrypt_information(encrypted)

    def test_modified_ciphertext_is_rejected(self):
        with patch.dict(os.environ, {"VISIONSEARCH_AES_KEY": TEST_KEY}):
            encrypted = encrypt_information("private information")
            ciphertext = bytearray(
                base64.b64decode(encrypted["ciphertext"], validate=True)
            )
            ciphertext[0] ^= 1
            encrypted["ciphertext"] = base64.b64encode(ciphertext).decode("ascii")
            with self.assertRaises(EncryptionError):
                decrypt_information(encrypted)

    def test_invalid_nonce_length_is_rejected(self):
        with patch.dict(os.environ, {"VISIONSEARCH_AES_KEY": TEST_KEY}):
            encrypted = encrypt_information("private information")
        encrypted["nonce"] = base64.b64encode(b"short nonce").decode("ascii")
        with patch.dict(os.environ, {"VISIONSEARCH_AES_KEY": TEST_KEY}):
            with self.assertRaises(EncryptionError):
                decrypt_information(encrypted)

    def test_invalid_ciphertext_base64_is_rejected(self):
        encrypted = {
            "nonce": base64.b64encode(b"0" * 12).decode("ascii"),
            "ciphertext": "not-base64!",
        }
        with patch.dict(os.environ, {"VISIONSEARCH_AES_KEY": TEST_KEY}):
            with self.assertRaises(EncryptionError):
                decrypt_information(encrypted)

    def test_empty_or_invalid_plaintext_is_rejected(self):
        with patch.dict(os.environ, {"VISIONSEARCH_AES_KEY": TEST_KEY}):
            for plaintext in ("", " \t\n", None, b"not text", "\ud800"):
                with self.subTest(plaintext=plaintext):
                    with self.assertRaises(EncryptionError):
                        encrypt_information(plaintext)


if __name__ == "__main__":
    unittest.main()
