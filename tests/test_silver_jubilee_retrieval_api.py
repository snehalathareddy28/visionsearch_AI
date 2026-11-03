import base64
import hashlib
import io
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from visionsearch_web_ui import api
from visionsearch_web_ui.region import api_regions
from visionsearch_web_ui.secure_info import auth_api
from visionsearch_web_ui.secure_info.auth import AuthenticationStore
from visionsearch_web_ui.secure_info.database import SilverJubileeDatabase
from visionsearch_web_ui.secure_info.service import SilverJubileeService


TEST_KEY = base64.b64encode(b"visionsearch-test-key-32-bytes!!").decode("ascii")
WRONG_TEST_KEY = base64.b64encode(b"incorrect-test-key-32-bytes-long").decode("ascii")
PRIVATE_INFORMATION = "Authenticated Silver Jubilee information"
USER_PASSWORD = "retrieval-user-test-password"
ADMIN_PASSWORD = "retrieval-admin-test-password"


def image_bytes(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(output, format="PNG")
    return output.getvalue()


class FakeClipModel:
    def encode_image(self, image: Image.Image) -> np.ndarray:
        vector = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
        vector /= np.linalg.norm(vector)
        return vector.reshape(1, -1)

    def encode_text(self, text: str) -> np.ndarray:
        return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)


class SilverJubileeRetrievalApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.old_root = api.ROOT
        self.old_model = api._model
        self.old_index_data = api._index_data
        self.old_get_index = api.get_index
        self.old_threshold = api.SILVER_JUBILEE_SIMILARITY_THRESHOLD
        self.old_ambiguity_margin = api.SILVER_JUBILEE_AMBIGUITY_MARGIN

        api.ROOT = self.root
        api._model = FakeClipModel()
        api.SILVER_JUBILEE_SIMILARITY_THRESHOLD = 0.75
        api.SILVER_JUBILEE_AMBIGUITY_MARGIN = 0.02

        self.registered_path = self.root / "dataset" / "registered.png"
        self.unregistered_path = self.root / "dataset" / "unregistered.png"
        self.registered_path.parent.mkdir(parents=True)
        self.registered_path.write_bytes(image_bytes((255, 0, 0)))
        self.unregistered_path.write_bytes(image_bytes((0, 255, 0)))
        self.paths = [str(self.registered_path), str(self.unregistered_path)]
        self.index = faiss.IndexFlatIP(3)
        self.index.add(np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
        api._index_data = (self.index, self.paths)
        api.get_index = lambda: api._index_data

        self.auth_store = AuthenticationStore(self.root / "auth.db")
        self.auth_store.initialize()
        self.user = self.auth_store.create_user("retrieval-user", USER_PASSWORD)
        self.admin = self.auth_store.create_admin("retrieval-admin", ADMIN_PASSWORD)
        self.database = SilverJubileeDatabase(self.root / "silver-jubilee.db")
        self.database.initialize()
        self.service = SilverJubileeService(self.database)

        self.old_overrides = dict(api.app.dependency_overrides)
        api.app.dependency_overrides[auth_api.get_authentication_store] = lambda: self.auth_store
        api.app.dependency_overrides[auth_api.get_registration_service] = lambda: self.service
        self.environment = patch.dict(
            os.environ,
            {
                "VISIONSEARCH_AES_KEY": TEST_KEY,
                "VISIONSEARCH_AUTH_COOKIE_SECURE": "false",
            },
        )
        self.environment.start()
        self.client = TestClient(api.app)
        self.registered_hash = api._sha256_file(self.registered_path)
        self.service.register_information(
            "dataset/registered.png",
            self.registered_hash,
            "Silver Jubilee",
            PRIVATE_INFORMATION,
        )

    def tearDown(self):
        self.client.close()
        self.environment.stop()
        api.app.dependency_overrides.clear()
        api.app.dependency_overrides.update(self.old_overrides)
        api.ROOT = self.old_root
        api._model = self.old_model
        api._index_data = self.old_index_data
        api.get_index = self.old_get_index
        api.SILVER_JUBILEE_SIMILARITY_THRESHOLD = self.old_threshold
        api.SILVER_JUBILEE_AMBIGUITY_MARGIN = self.old_ambiguity_margin
        self.temporary_directory.cleanup()

    def authenticate(self, user_id: int | None = None) -> str:
        token, csrf_token, _ = self.auth_store.create_session(
            user_id or self.user.id,
            300,
        )
        self.client.cookies.set(
            auth_api._AUTH_COOKIE_NAME,
            token,
            path="/api",
        )
        self.client.cookies.set(
            auth_api._CSRF_COOKIE_NAME,
            csrf_token,
            path="/api",
        )
        self.csrf_token = csrf_token
        return csrf_token

    def post_retrieval(self, image: bytes | None = None, csrf_token: str | None = None):
        token = csrf_token or getattr(self, "csrf_token", None)
        return self.client.post(
            "/api/silver-jubilee/search",
            files={
                "file": (
                    "query.png",
                    image if image is not None else image_bytes((255, 0, 0)),
                    "image/png",
                )
            },
            headers={"X-CSRF-Token": token} if token else {},
        )

    def set_index(self, paths: list[Path], vectors: list[list[float]]) -> None:
        index = faiss.IndexFlatIP(3)
        index.add(np.asarray(vectors, dtype=np.float32))
        self.index = index
        self.paths = [str(path) for path in paths]
        api._index_data = (self.index, self.paths)

    def test_authenticated_user_matches_registered_image_and_gets_information(self):
        csrf = self.authenticate()
        response = self.post_retrieval(csrf_token=csrf)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["matched"])
        self.assertEqual(response.json()["title"], "Silver Jubilee")
        self.assertEqual(response.json()["information"], PRIVATE_INFORMATION)

    def test_authenticated_user_uploading_unregistered_image_gets_no_information(self):
        self.authenticate()
        response = self.post_retrieval(image_bytes((0, 255, 0)))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["matched"])
        self.assertEqual(response.json()["reason"], "no_registered_information")
        self.assertNotIn("information", response.json())

    def test_unauthenticated_user_cannot_retrieve_information(self):
        response = self.post_retrieval()
        self.assertEqual(response.status_code, 401)
        self.assertNotIn(PRIVATE_INFORMATION, response.text)

    def test_authenticated_retrieval_requires_csrf_token(self):
        self.authenticate()
        response = self.client.post(
            "/api/silver-jubilee/search",
            files={"file": ("query.png", image_bytes((255, 0, 0)), "image/png")},
        )
        self.assertEqual(response.status_code, 403)
        self.assertNotIn(PRIVATE_INFORMATION, response.text)

    def test_invalid_or_expired_session_is_rejected(self):
        self.client.cookies.set(
            auth_api._AUTH_COOKIE_NAME,
            "invalid-session-token",
            path="/api",
        )
        self.client.cookies.set(
            auth_api._CSRF_COOKIE_NAME,
            "csrf",
            path="/api",
        )
        self.assertEqual(
            self.post_retrieval(csrf_token="csrf").status_code,
            401,
        )
        token, csrf, _ = self.auth_store.create_session(self.user.id, 300)
        with closing(sqlite3.connect(self.auth_store.database_path)) as connection, connection:
            connection.execute("UPDATE secure_info_sessions SET expires_at = 1")
        self.client.cookies.set(auth_api._AUTH_COOKIE_NAME, token, path="/api")
        self.client.cookies.set(auth_api._CSRF_COOKIE_NAME, csrf, path="/api")
        self.assertEqual(self.post_retrieval(csrf_token=csrf).status_code, 401)

    def test_normal_user_cannot_administer_registrations(self):
        csrf = self.authenticate()
        response = self.client.post(
            "/api/secure-info/registrations",
            json={
                "image_reference": "dataset/registered.png",
                "image_hash": self.registered_hash,
                "title": "Unauthorized",
                "information": PRIVATE_INFORMATION,
            },
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(response.status_code, 403)

    def test_no_match_above_configured_similarity_threshold(self):
        api.SILVER_JUBILEE_SIMILARITY_THRESHOLD = 0.8
        self.set_index(
            [self.unregistered_path],
            [[0.6, 0.8, 0.0]],
        )
        self.authenticate()
        response = self.post_retrieval()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["matched"])
        self.assertEqual(response.json()["reason"], "no_confident_match")

    def test_ambiguous_near_tie_returns_no_information(self):
        nearby_path = self.root / "dataset" / "nearby.png"
        nearby_path.write_bytes(image_bytes((255, 10, 0)))
        self.set_index(
            [self.registered_path, nearby_path],
            [[1, 0, 0], [0.999, 0.04, 0]],
        )
        self.authenticate()
        response = self.post_retrieval()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["matched"])
        self.assertEqual(response.json()["reason"], "ambiguous_match")
        self.assertNotIn("information", response.json())

    def test_registered_image_absent_from_active_index_cannot_return_information(self):
        self.set_index(
            [self.unregistered_path],
            [[0.8, 0.6, 0]],
        )
        self.authenticate()
        response = self.post_retrieval()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["matched"])
        self.assertEqual(response.json()["reason"], "no_registered_information")

    def test_tampered_ciphertext_returns_safe_failure(self):
        with closing(sqlite3.connect(self.database.database_path)) as connection, connection:
            connection.execute(
                "UPDATE silver_jubilee_records SET ciphertext = ?",
                ("dGFtcGVyZWQ=",),
            )
        self.authenticate()
        response = self.post_retrieval()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(PRIVATE_INFORMATION, response.text)
        self.assertNotIn("ciphertext", response.text.lower())

    def test_incorrect_encryption_key_returns_safe_failure(self):
        with patch.dict(os.environ, {"VISIONSEARCH_AES_KEY": WRONG_TEST_KEY}):
            self.authenticate()
            response = self.post_retrieval()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(PRIVATE_INFORMATION, response.text)

    def test_ciphertext_and_nonce_are_not_exposed_on_success(self):
        self.authenticate()
        response = self.post_retrieval()
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("ciphertext", response.json())
        self.assertNotIn("nonce", response.json())

    def test_invalid_registration_reference_is_rejected(self):
        csrf = self.authenticate(self.admin.id)
        response = self.client.post(
            "/api/secure-info/registrations",
            json={
                "image_reference": "dataset/not-indexed.png",
                "image_hash": "1" * 64,
                "title": "Rejected",
                "information": PRIVATE_INFORMATION,
            },
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(response.status_code, 400)

    def test_existing_text_image_and_region_search_routes_remain_operational(self):
        text_response = self.client.post(
            "/api/search/text",
            json={"query": "red college building", "top_k": 5},
        )
        self.assertEqual(text_response.status_code, 200, text_response.text)

        image_response = self.client.post(
            "/api/search/image",
            files={"file": ("query.png", image_bytes((255, 0, 0)), "image/png")},
        )
        self.assertEqual(image_response.status_code, 200, image_response.text)
        self.assertTrue(image_response.json()["results"])

        with patch.object(api_regions, "search_regions", return_value=[]):
            region_response = self.client.post(
                "/api/region/search/text",
                json={
                    "query": "building",
                    "top_k": 5,
                    "search_scope": "both",
                    "request_id": "phase6-region-regression",
                },
            )
        self.assertEqual(region_response.status_code, 200, region_response.text)


if __name__ == "__main__":
    unittest.main()
