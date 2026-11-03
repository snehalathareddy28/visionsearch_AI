import base64
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from visionsearch_web_ui.secure_info.database import (
    DuplicatePhotographError,
    RecordNotFoundError,
    SecureInfoDatabaseError,
    SilverJubileeDatabase,
)
from visionsearch_web_ui.secure_info.encryption import decrypt_information


TEST_KEY = base64.b64encode(b"visionsearch-test-key-32-bytes!!").decode("ascii")
PLAINTEXT = "Silver Jubilee history that must remain encrypted"
IMAGE_HASH = "a" * 64


class SilverJubileeDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "nested" / "test.db"
        self.database = SilverJubileeDatabase(self.database_path)
        self.environment = patch.dict(os.environ, {"VISIONSEARCH_AES_KEY": TEST_KEY})
        self.environment.start()
        self.database.initialize()

    def tearDown(self):
        self.environment.stop()
        self.temporary_directory.cleanup()

    def register(self, **overrides):
        values = {
            "image_path": "uploads/silver-jubilee.jpg",
            "image_hash": IMAGE_HASH,
            "title": "Silver Jubilee",
            "information": PLAINTEXT,
        }
        values.update(overrides)
        return self.database.register_record(**values)

    def test_initialization_creates_separate_schema_and_index(self):
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            indexes = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index'"
                )
            }
        self.assertIn("silver_jubilee_records", tables)
        self.assertNotIn("region_vectors", tables)
        self.assertIn("idx_silver_jubilee_records_created_at", indexes)

    def test_successful_registration_stores_decryptable_ciphertext(self):
        record = self.register()
        self.assertEqual(record.image_path, "uploads/silver-jubilee.jpg")
        self.assertEqual(record.image_hash, IMAGE_HASH)
        self.assertEqual(decrypt_information({
            "nonce": record.nonce,
            "ciphertext": record.ciphertext,
        }), PLAINTEXT)

    def test_duplicate_path_or_image_hash_is_rejected(self):
        self.register()
        with self.assertRaises(DuplicatePhotographError):
            self.register(image_path="uploads/different.jpg", image_hash=IMAGE_HASH)
        with self.assertRaises(DuplicatePhotographError):
            self.register(image_path="uploads/silver-jubilee.jpg", image_hash="b" * 64)

    def test_retrieval_by_id_and_image_hash(self):
        inserted = self.register()
        self.assertEqual(self.database.get_record_by_id(inserted.id), inserted)
        self.assertEqual(self.database.find_record_by_image_hash(IMAGE_HASH), inserted)
        self.assertIsNone(self.database.get_record_by_id(999))
        self.assertIsNone(self.database.find_record_by_image_hash("c" * 64))

    def test_update_changes_title_and_encrypted_information(self):
        inserted = self.register()
        updated = self.database.update_record(inserted.id, "Updated title", "Updated private details")
        self.assertEqual(updated.title, "Updated title")
        self.assertNotEqual(updated.ciphertext, inserted.ciphertext)
        self.assertEqual(updated.image_path, inserted.image_path)
        self.assertEqual(
            decrypt_information({"nonce": updated.nonce, "ciphertext": updated.ciphertext}),
            "Updated private details",
        )
        self.assertGreaterEqual(updated.updated_at, inserted.updated_at)

    def test_update_of_missing_record_raises(self):
        with self.assertRaises(RecordNotFoundError):
            self.database.update_record(123, "Title", "Private information")

    def test_delete_record_and_report_absence(self):
        record = self.register()
        self.assertTrue(self.database.delete_record(record.id))
        self.assertFalse(self.database.delete_record(record.id))
        self.assertIsNone(self.database.get_record_by_id(record.id))

    def test_list_records_is_paged_and_contains_no_plaintext_field(self):
        first = self.register()
        second = self.register(
            image_path="uploads/another.jpg",
            image_hash="b" * 64,
            title="Another title",
        )
        page = self.database.list_records(limit=1)
        self.assertEqual(len(page), 1)
        self.assertIn(page[0].id, {first.id, second.id})
        self.assertEqual(len(self.database.list_records(limit=10, offset=1)), 1)
        self.assertFalse(hasattr(page[0], "information"))

    def test_plaintext_is_not_stored_in_database_file(self):
        self.register()
        database_bytes = self.database_path.read_bytes()
        self.assertNotIn(PLAINTEXT.encode("utf-8"), database_bytes)
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            columns = {
                row[1] for row in connection.execute(
                    "PRAGMA table_info(silver_jubilee_records)"
                )
            }
        self.assertNotIn("information", columns)
        self.assertIn("ciphertext", columns)

    def test_invalid_inputs_are_rejected(self):
        invalid_records = (
            {"image_path": "../outside.jpg"},
            {"image_path": "uploads/../outside.jpg"},
            {"image_path": ""},
            {"image_hash": "not-a-sha256-hash"},
            {"title": "   "},
            {"information": " \n "},
        )
        for override in invalid_records:
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.register(**override)
        with self.assertRaises(ValueError):
            self.database.get_record_by_id(0)
        with self.assertRaises(ValueError):
            self.database.list_records(limit=0)

    def test_failed_update_rolls_back_and_reports_database_error(self):
        record = self.register()
        with closing(sqlite3.connect(self.database_path)) as connection, connection:
            connection.execute(
                """
                CREATE TRIGGER reject_test_title
                BEFORE UPDATE ON silver_jubilee_records
                WHEN NEW.title = 'trigger failure'
                BEGIN
                    SELECT RAISE(ABORT, 'forced test failure');
                END
                """
            )

        with self.assertRaises(SecureInfoDatabaseError):
            self.database.update_record(record.id, "trigger failure", "Changed details")
        current = self.database.get_record_by_id(record.id)
        self.assertEqual(current, record)


if __name__ == "__main__":
    unittest.main()
