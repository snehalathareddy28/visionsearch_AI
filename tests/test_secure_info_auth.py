import base64
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from visionsearch_web_ui import api
from visionsearch_web_ui.secure_info import auth_api
from visionsearch_web_ui.secure_info.auth import (
    AdminAlreadyProvisionedError,
    AuthenticationStore,
    hash_password,
    verify_password,
)
from visionsearch_web_ui.secure_info.database import (
    SecureInfoDatabaseError,
    SilverJubileeDatabase,
)
from visionsearch_web_ui.secure_info.service import SilverJubileeService


TEST_KEY = base64.b64encode(b"visionsearch-test-key-32-bytes!!").decode("ascii")
ADMIN_PASSWORD = "Admin-test-password-2026!"
USER_PASSWORD = "User-test-password-2026!"
MANAGED_USER_PASSWORD = "Managed-user-temporary-password!"
RESET_USER_PASSWORD = "Managed-user-reset-password!"
PRIVATE_INFORMATION = "Non-public anniversary information"


class SecureInfoAuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.auth_store = AuthenticationStore(root / "auth.db")
        self.auth_store.initialize()
        self.admin = self.auth_store.create_admin("admin-user", ADMIN_PASSWORD)
        self.user = self.auth_store.create_user("normal-user", USER_PASSWORD)
        self.registration_database = SilverJubileeDatabase(root / "silver-jubilee.db")
        self.registration_database.initialize()
        self.service = SilverJubileeService(self.registration_database)
        self.overrides_before = dict(api.app.dependency_overrides)
        api.app.dependency_overrides[auth_api.get_authentication_store] = lambda: self.auth_store
        api.app.dependency_overrides[auth_api.get_registration_service] = lambda: self.service
        self.registration_validation = patch.object(
            auth_api,
            "validate_indexed_image_reference",
            side_effect=lambda image_reference, image_hash: image_reference,
        )
        self.registration_validation.start()
        self.client = TestClient(api.app)
        self.cookie_settings = patch.dict(
            os.environ,
            {
                "VISIONSEARCH_AUTH_COOKIE_SECURE": "false",
                "VISIONSEARCH_AES_KEY": TEST_KEY,
            },
        )
        self.cookie_settings.start()

    def tearDown(self):
        self.client.close()
        self.registration_validation.stop()
        api.app.dependency_overrides.clear()
        api.app.dependency_overrides.update(self.overrides_before)
        self.cookie_settings.stop()
        self.temporary_directory.cleanup()

    def csrf_token(self):
        response = self.client.get("/api/secure-info/auth/csrf")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["csrf_token"]

    def login(self, username="admin-user", password=ADMIN_PASSWORD):
        csrf = self.csrf_token()
        response = self.client.post(
            "/api/secure-info/auth/login",
            json={"username": username, "password": password},
            headers={"X-CSRF-Token": csrf},
        )
        return response

    def test_password_hash_is_salted_and_never_plaintext(self):
        first_hash = hash_password(ADMIN_PASSWORD)
        second_hash = hash_password(ADMIN_PASSWORD)
        self.assertNotEqual(first_hash, ADMIN_PASSWORD)
        self.assertNotEqual(first_hash, second_hash)
        self.assertTrue(verify_password(ADMIN_PASSWORD, first_hash))
        self.assertFalse(verify_password("wrong password", first_hash))
        with closing(sqlite3.connect(self.auth_store.database_path)) as connection, connection:
            row = connection.execute(
                "SELECT password_hash FROM secure_info_users WHERE username = ?",
                ("admin-user",),
            ).fetchone()
        self.assertTrue(row[0].startswith("scrypt$"))
        self.assertTrue(verify_password(ADMIN_PASSWORD, row[0]))
        self.assertNotIn(ADMIN_PASSWORD, row[0])

    def test_successful_login_sets_http_only_cookie_and_returns_no_secret(self):
        response = self.login()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["user"]["role"], "admin")
        self.assertNotIn("password", response.json())
        self.assertNotIn("token", response.json())
        cookie = response.cookies.get(auth_api._AUTH_COOKIE_NAME)
        self.assertTrue(cookie)
        set_cookie = response.headers["set-cookie"].lower()
        self.assertIn("httponly", set_cookie)
        self.assertIn("samesite=lax", set_cookie)
        self.assertEqual(
            self.client.get("/api/secure-info/auth/me").json(),
            {"username": "admin-user", "role": "admin"},
        )

    def test_invalid_password_fails_with_generic_response(self):
        response = self.login(password="incorrect-test-password")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Invalid username or password.")
        self.assertNotIn("incorrect-test-password", response.text)

    def test_missing_and_invalid_sessions_are_rejected(self):
        response = self.client.get("/api/secure-info/auth/me")
        self.assertEqual(response.status_code, 401)

        self.client.cookies.set(
            auth_api._AUTH_COOKIE_NAME,
            "not-a-valid-session",
            path="/api/secure-info",
        )
        invalid_response = self.client.get("/api/secure-info/auth/me")
        self.assertEqual(invalid_response.status_code, 401)

    def test_expired_session_is_rejected(self):
        token, _, _ = self.auth_store.create_session(self.admin.id, 60)
        with closing(sqlite3.connect(self.auth_store.database_path)) as connection, connection:
            connection.execute(
                "UPDATE secure_info_sessions SET expires_at = 1"
            )
        self.client.cookies.set(
            auth_api._AUTH_COOKIE_NAME,
            token,
            path="/api/secure-info",
        )
        response = self.client.get("/api/secure-info/auth/me")
        self.assertEqual(response.status_code, 401)

    def test_duplicate_admin_provisioning_is_rejected(self):
        with self.assertRaises(AdminAlreadyProvisionedError):
            self.auth_store.create_admin("second-admin", "Another-admin-test-password!")

    def test_admin_can_create_regular_users_and_duplicate_names_are_rejected(self):
        login = self.login()
        csrf = login.json()["csrf_token"]
        create = self.client.post(
            "/api/secure-info/users",
            json={"username": "managed-user", "password": MANAGED_USER_PASSWORD},
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(create.status_code, 201, create.text)
        account = create.json()
        self.assertEqual(account["username"], "managed-user")
        self.assertEqual(account["role"], "user")
        self.assertTrue(account["is_active"])
        self.assertIn("created_at", account)
        for secret_field in ("password", "password_hash", "csrf_token", "session_token"):
            self.assertNotIn(secret_field, account)
            self.assertNotIn(secret_field, create.text)
        self.assertNotIn(MANAGED_USER_PASSWORD, create.text)
        self.assertIsNotNone(
            self.auth_store.verify_credentials("managed-user", MANAGED_USER_PASSWORD)
        )

        duplicate = self.client.post(
            "/api/secure-info/users",
            json={"username": "managed-user", "password": MANAGED_USER_PASSWORD},
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(duplicate.status_code, 409)
        self.assertNotIn(MANAGED_USER_PASSWORD, duplicate.text)

        administrator_attempt = self.client.post(
            "/api/secure-info/users",
            json={
                "username": "another-admin",
                "password": MANAGED_USER_PASSWORD,
                "role": "admin",
            },
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(administrator_attempt.status_code, 400)
        self.assertNotIn(
            "another-admin",
            {account.username for account in self.auth_store.list_users()},
        )
        invalid_username = self.client.post(
            "/api/secure-info/users",
            json={"username": "not a valid username", "password": MANAGED_USER_PASSWORD},
            headers={"X-CSRF-Token": csrf},
        )
        invalid_password = self.client.post(
            "/api/secure-info/users",
            json={"username": "valid-username", "password": "short"},
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(invalid_username.status_code, 400)
        self.assertEqual(invalid_password.status_code, 400)
        self.assertNotIn("short", invalid_password.text)

    def test_user_management_list_is_admin_only_and_exposes_safe_metadata(self):
        self.auth_store.create_user("listed-user", MANAGED_USER_PASSWORD)
        login = self.login()
        listing = self.client.get("/api/secure-info/users")
        self.assertEqual(listing.status_code, 200, listing.text)
        self.assertNotIn(
            "admin-user",
            {item["username"] for item in listing.json()},
        )
        account = next(item for item in listing.json() if item["username"] == "listed-user")
        self.assertEqual(account["role"], "user")
        self.assertIn("is_active", account)
        self.assertIn("created_at", account)
        self.assertNotIn("password_hash", listing.text)
        self.assertNotIn(MANAGED_USER_PASSWORD, listing.text)
        self.assertEqual(login.status_code, 200)

        regular_user_login = self.login("normal-user", USER_PASSWORD)
        self.assertEqual(regular_user_login.status_code, 200)
        self.assertEqual(self.client.get("/api/secure-info/users").status_code, 403)
        headers = {"X-CSRF-Token": regular_user_login.json()["csrf_token"]}
        self.assertEqual(
            self.client.post(
                "/api/secure-info/users",
                json={"username": "blocked-user", "password": MANAGED_USER_PASSWORD},
                headers=headers,
            ).status_code,
            403,
        )
        self.assertEqual(
            self.client.patch(
                f"/api/secure-info/users/{self.user.id}/status",
                json={"is_active": False},
                headers=headers,
            ).status_code,
            403,
        )
        self.assertEqual(
            self.client.post(
                f"/api/secure-info/users/{self.user.id}/password",
                json={"password": RESET_USER_PASSWORD},
                headers=headers,
            ).status_code,
            403,
        )

    def test_user_management_mutations_require_valid_csrf(self):
        self.login()
        payload = {"username": "csrf-user", "password": MANAGED_USER_PASSWORD}
        missing = self.client.post("/api/secure-info/users", json=payload)
        invalid = self.client.post(
            "/api/secure-info/users",
            json=payload,
            headers={"X-CSRF-Token": "invalid-csrf-token"},
        )
        self.assertEqual(missing.status_code, 403)
        self.assertEqual(invalid.status_code, 403)
        self.assertNotIn(
            "csrf-user",
            {account.username for account in self.auth_store.list_users()},
        )

    def test_regular_user_activation_and_deactivation_control_login_and_sessions(self):
        login = self.login()
        csrf = login.json()["csrf_token"]
        created = self.client.post(
            "/api/secure-info/users",
            json={"username": "status-user", "password": MANAGED_USER_PASSWORD},
            headers={"X-CSRF-Token": csrf},
        )
        user_id = created.json()["id"]
        user_client = TestClient(api.app)
        try:
            user_csrf = user_client.get("/api/secure-info/auth/csrf").json()["csrf_token"]
            user_login = user_client.post(
                "/api/secure-info/auth/login",
                json={"username": "status-user", "password": MANAGED_USER_PASSWORD},
                headers={"X-CSRF-Token": user_csrf},
            )
            self.assertEqual(user_login.status_code, 200, user_login.text)
            user_csrf = user_login.json()["csrf_token"]
            session_token = user_client.cookies.get(auth_api._AUTH_COOKIE_NAME)

            deactivated = self.client.patch(
                f"/api/secure-info/users/{user_id}/status",
                json={"is_active": False},
                headers={"X-CSRF-Token": csrf},
            )
            self.assertEqual(deactivated.status_code, 200, deactivated.text)
            self.assertFalse(deactivated.json()["is_active"])
            self.assertIsNone(self.auth_store.get_session(session_token))
            self.assertEqual(
                user_client.get("/api/secure-info/auth/me").status_code,
                401,
            )
            blocked_login = user_client.post(
                "/api/secure-info/auth/login",
                json={"username": "status-user", "password": MANAGED_USER_PASSWORD},
                headers={"X-CSRF-Token": user_csrf},
            )
            self.assertEqual(blocked_login.status_code, 401)

            activated = self.client.patch(
                f"/api/secure-info/users/{user_id}/status",
                json={"is_active": True},
                headers={"X-CSRF-Token": csrf},
            )
            self.assertEqual(activated.status_code, 200, activated.text)
            self.assertTrue(activated.json()["is_active"])
            successful_login = user_client.post(
                "/api/secure-info/auth/login",
                json={"username": "status-user", "password": MANAGED_USER_PASSWORD},
                headers={"X-CSRF-Token": user_csrf},
            )
            self.assertEqual(successful_login.status_code, 200, successful_login.text)
        finally:
            user_client.close()

    def test_password_reset_revokes_sessions_and_new_password_can_login(self):
        login = self.login()
        csrf = login.json()["csrf_token"]
        created = self.client.post(
            "/api/secure-info/users",
            json={"username": "reset-user", "password": MANAGED_USER_PASSWORD},
            headers={"X-CSRF-Token": csrf},
        )
        user_id = created.json()["id"]
        user_client = TestClient(api.app)
        try:
            user_csrf = user_client.get("/api/secure-info/auth/csrf").json()["csrf_token"]
            user_login = user_client.post(
                "/api/secure-info/auth/login",
                json={"username": "reset-user", "password": MANAGED_USER_PASSWORD},
                headers={"X-CSRF-Token": user_csrf},
            )
            self.assertEqual(user_login.status_code, 200, user_login.text)
            user_csrf = user_login.json()["csrf_token"]
            session_token = user_client.cookies.get(auth_api._AUTH_COOKIE_NAME)

            reset = self.client.post(
                f"/api/secure-info/users/{user_id}/password",
                json={"password": RESET_USER_PASSWORD},
                headers={"X-CSRF-Token": csrf},
            )
            self.assertEqual(reset.status_code, 200, reset.text)
            self.assertNotIn(RESET_USER_PASSWORD, reset.text)
            self.assertNotIn("password_hash", reset.text)
            self.assertIsNone(self.auth_store.get_session(session_token))
            self.assertIsNone(
                self.auth_store.verify_credentials("reset-user", MANAGED_USER_PASSWORD)
            )

            old_password_login = user_client.post(
                "/api/secure-info/auth/login",
                json={"username": "reset-user", "password": MANAGED_USER_PASSWORD},
                headers={"X-CSRF-Token": user_csrf},
            )
            self.assertEqual(old_password_login.status_code, 401)
            new_password_login = user_client.post(
                "/api/secure-info/auth/login",
                json={"username": "reset-user", "password": RESET_USER_PASSWORD},
                headers={"X-CSRF-Token": user_csrf},
            )
            self.assertEqual(new_password_login.status_code, 200, new_password_login.text)
        finally:
            user_client.close()

    def test_administrator_cannot_deactivate_or_reset_own_account(self):
        login = self.login()
        csrf = login.json()["csrf_token"]
        status = self.client.patch(
            f"/api/secure-info/users/{self.admin.id}/status",
            json={"is_active": False},
            headers={"X-CSRF-Token": csrf},
        )
        reset = self.client.post(
            f"/api/secure-info/users/{self.admin.id}/password",
            json={"password": RESET_USER_PASSWORD},
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(status.status_code, 409)
        self.assertEqual(reset.status_code, 409)
        self.assertIsNotNone(self.auth_store.verify_credentials("admin-user", ADMIN_PASSWORD))

    def test_existing_authentication_database_is_migrated_without_changing_accounts(self):
        legacy_path = Path(self.temporary_directory.name) / "legacy-auth.db"
        with closing(sqlite3.connect(legacy_path)) as connection, connection:
            connection.execute(
                """
                CREATE TABLE secure_info_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO secure_info_users (username, password_hash, role, created_at)
                VALUES (?, ?, 'user', '2025-01-02T03:04:05.000Z')
                """,
                ("legacy-user", hash_password(USER_PASSWORD)),
            )
        legacy_store = AuthenticationStore(legacy_path)
        legacy_store.initialize()
        self.assertIsNotNone(legacy_store.verify_credentials("legacy-user", USER_PASSWORD))
        accounts = legacy_store.list_users()
        self.assertEqual(len(accounts), 1)
        self.assertEqual(accounts[0].username, "legacy-user")
        self.assertTrue(accounts[0].is_active)
        self.assertEqual(accounts[0].created_at, "2025-01-02T03:04:05.000Z")

    def test_user_cannot_access_admin_registration_operations(self):
        response = self.login("normal-user", USER_PASSWORD)
        self.assertEqual(response.status_code, 200, response.text)
        listing = self.client.get("/api/secure-info/registrations")
        self.assertEqual(listing.status_code, 403)
        mutation = self.client.post(
            "/api/secure-info/registrations",
            json={
                "image_reference": "uploads/college.jpg",
                "image_hash": "f" * 64,
                "title": "Milestone",
                "information": PRIVATE_INFORMATION,
            },
            headers={"X-CSRF-Token": response.json()["csrf_token"]},
        )
        self.assertEqual(mutation.status_code, 403)

    def test_admin_registration_management_exposes_metadata_only(self):
        login = self.login()
        csrf = login.json()["csrf_token"]
        create = self.client.post(
            "/api/secure-info/registrations",
            json={
                "image_reference": "uploads/college.jpg",
                "image_hash": "f" * 64,
                "title": "Silver Jubilee",
                "information": PRIVATE_INFORMATION,
            },
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(create.status_code, 201, create.text)
        self.assertNotIn("ciphertext", create.json())
        self.assertNotIn("nonce", create.json())
        self.assertNotIn(PRIVATE_INFORMATION, create.text)

        listing = self.client.get("/api/secure-info/registrations")
        self.assertEqual(listing.status_code, 200, listing.text)
        self.assertEqual(len(listing.json()), 1)
        self.assertNotIn("ciphertext", listing.text)
        self.assertNotIn("nonce", listing.text)
        self.assertNotIn(PRIVATE_INFORMATION, listing.text)

        updated = self.client.put(
            f"/api/secure-info/registrations/{create.json()['id']}",
            json={"title": "Updated", "information": "Updated private information"},
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertNotIn("ciphertext", updated.json())
        deleted = self.client.delete(
            f"/api/secure-info/registrations/{create.json()['id']}",
            headers={"X-CSRF-Token": csrf},
        )
        self.assertEqual(deleted.status_code, 204)

    def test_admin_mutation_requires_csrf_token(self):
        self.login()
        response = self.client.post(
            "/api/secure-info/registrations",
            json={
                "image_reference": "uploads/college.jpg",
                "image_hash": "a" * 64,
                "title": "Title",
                "information": PRIVATE_INFORMATION,
            },
        )
        self.assertEqual(response.status_code, 403)

    def test_logout_revokes_session(self):
        login = self.login()
        response = self.client.post(
            "/api/secure-info/auth/logout",
            headers={"X-CSRF-Token": login.json()["csrf_token"]},
        )
        self.assertEqual(response.status_code, 204)
        self.assertEqual(
            self.client.get("/api/secure-info/auth/me").status_code,
            401,
        )

    def test_database_failure_returns_safe_api_error(self):
        login = self.login()

        class FailingService:
            def register_information(self, **_):
                raise SecureInfoDatabaseError("The secure information database operation failed.")

        api.app.dependency_overrides[auth_api.get_registration_service] = lambda: FailingService()
        response = self.client.post(
            "/api/secure-info/registrations",
            json={
                "image_reference": "uploads/college.jpg",
                "image_hash": "b" * 64,
                "title": "Title",
                "information": PRIVATE_INFORMATION,
            },
            headers={"X-CSRF-Token": login.json()["csrf_token"]},
        )
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(PRIVATE_INFORMATION, response.text)

    def test_cors_does_not_allow_wildcard_credentials(self):
        cors = next(
            middleware
            for middleware in api.app.user_middleware
            if middleware.cls.__name__ == "CORSMiddleware"
        )
        self.assertNotIn("*", cors.kwargs["allow_origins"])


if __name__ == "__main__":
    unittest.main()
