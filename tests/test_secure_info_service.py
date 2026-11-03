import base64
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from visionsearch_web_ui.secure_info.database import (
    SecureInfoDatabaseError,
    SilverJubileeDatabase,
)
from visionsearch_web_ui.secure_info.encryption import decrypt_information
from visionsearch_web_ui.secure_info.service import SilverJubileeService


TEST_KEY = base64.b64encode(b"visionsearch-test-key-32-bytes!!").decode("ascii")
PRIVATE_INFORMATION = "Sensitive anniversary details only for authorized users"
IMAGE_HASH = "d" * 64


class SilverJubileeServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "service.db"
        self.database = SilverJubileeDatabase(self.database_path)
        self.database.initialize()
        self.service = SilverJubileeService(self.database)
        self.environment = patch.dict(os.environ, {"VISIONSEARCH_AES_KEY": TEST_KEY})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary_directory.cleanup()

    def register(self, **overrides):
        values = {
            "image_reference": "uploads/silver-jubilee.jpg",
            "image_hash": IMAGE_HASH,
            "title": "Silver Jubilee",
            "information": PRIVATE_INFORMATION,
        }
        values.update(overrides)
        return self.service.register_information(**values)

    def test_successful_registration_returns_metadata_only(self):
        registration = self.register()
        self.assertEqual(registration.image_path, "uploads/silver-jubilee.jpg")
        self.assertEqual(registration.title, "Silver Jubilee")
        self.assertFalse(hasattr(registration, "ciphertext"))
        self.assertFalse(hasattr(registration, "information"))

    def test_information_is_encrypted_before_persistence(self):
        registration = self.register()
        record = self.database.get_record_by_id(registration.id)
        self.assertIsNotNone(record)
        self.assertNotEqual(record.ciphertext, PRIVATE_INFORMATION)
        self.assertEqual(
            decrypt_information({"nonce": record.nonce, "ciphertext": record.ciphertext}),
            PRIVATE_INFORMATION,
        )
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            rows = connection.execute(
                "SELECT ciphertext, nonce FROM silver_jubilee_records WHERE id = ?",
                (registration.id,),
            ).fetchone()
        self.assertNotIn(PRIVATE_INFORMATION.encode("utf-8"), b"".join(value.encode("ascii") for value in rows))

    def test_retrieval_by_hash_returns_metadata_and_ciphertext_not_plaintext(self):
        registration = self.register()
        record = self.service.retrieve_information(IMAGE_HASH)
        self.assertIsNotNone(record)
        self.assertEqual(record.id, registration.id)
        self.assertTrue(record.ciphertext)
        self.assertTrue(record.nonce)
        self.assertFalse(hasattr(record, "information"))
        self.assertNotIn(PRIVATE_INFORMATION, repr(record))

    def test_missing_image_registration_returns_none(self):
        self.assertIsNone(self.service.retrieve_information("e" * 64))

    def test_successful_update_uses_fresh_nonce(self):
        registration = self.register()
        original = self.service.retrieve_information(IMAGE_HASH)
        updated = self.service.update_information(
            registration.id,
            "Updated title",
            "Updated sensitive details",
        )
        current = self.service.retrieve_information(IMAGE_HASH)
        self.assertEqual(updated.title, "Updated title")
        self.assertNotEqual(current.nonce, original.nonce)
        self.assertEqual(
            decrypt_information({"nonce": current.nonce, "ciphertext": current.ciphertext}),
            "Updated sensitive details",
        )

    def test_deletion_removes_registration(self):
        registration = self.register()
        self.assertTrue(self.service.delete_registration(registration.id))
        self.assertFalse(self.service.delete_registration(registration.id))
        self.assertIsNone(self.service.retrieve_information(IMAGE_HASH))

    def test_listing_is_bounded_and_contains_metadata_only(self):
        first = self.register()
        self.register(image_reference="uploads/second.jpg", image_hash="e" * 64)
        entries = self.service.list_registrations(limit=1)
        self.assertEqual(len(entries), 1)
        self.assertFalse(hasattr(entries[0], "ciphertext"))
        self.assertFalse(hasattr(entries[0], "nonce"))
        self.assertFalse(hasattr(entries[0], "information"))
        self.assertEqual(len(self.service.list_registrations(limit=10, offset=1)), 1)
        self.assertIn(first.id, {item.id for item in self.service.list_registrations(limit=10)})

    def test_invalid_inputs_are_rejected(self):
        for override in (
            {"image_reference": "../secret.jpg"},
            {"image_reference": "C:\\private\\secret.jpg"},
            {"image_reference": "/private/secret.jpg"},
            {"image_hash": "invalid"},
            {"title": "  "},
            {"information": " \t "},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.register(**override)
        with self.assertRaises(ValueError):
            self.service.update_information(0, "Title", "Information")
        with self.assertRaises(ValueError):
            self.service.list_registrations(limit=101)

    def test_plaintext_is_never_returned_by_retrieval_or_listing(self):
        registration = self.register()
        retrieved = self.service.retrieve_information(IMAGE_HASH)
        listed = self.service.list_registrations(limit=10)
        self.assertNotIn(PRIVATE_INFORMATION, repr(retrieved))
        self.assertNotIn(PRIVATE_INFORMATION, repr(listed))
        self.assertNotIn(PRIVATE_INFORMATION, repr(registration))

    def test_database_errors_propagate_without_sensitive_data(self):
        with patch.object(
            self.database,
            "register_encrypted_record",
            side_effect=SecureInfoDatabaseError("The secure information database operation failed."),
        ):
            with self.assertRaises(SecureInfoDatabaseError) as caught:
                self.register()
        self.assertNotIn(PRIVATE_INFORMATION, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
