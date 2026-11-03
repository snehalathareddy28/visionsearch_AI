"""Authentication and admin-only Silver Jubilee registration routes."""

from __future__ import annotations

import hmac
import hashlib
import os
import secrets
import time
from dataclasses import asdict
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field

from visionsearch_web_ui.secure_info.auth import (
    AuthenticationStore,
    AuthenticationStoreError,
    AuthenticatedUser,
    DuplicateUserError,
    ManagedUser,
    SelfUserModificationError,
    UserNotFoundError,
)
from visionsearch_web_ui.secure_info.database import (
    DEFAULT_DATABASE_PATH,
    DuplicatePhotographError,
    RecordNotFoundError,
    SecureInfoDatabaseError,
    SilverJubileeDatabase,
)
from visionsearch_web_ui.secure_info.encryption import EncryptionConfigurationError
from visionsearch_web_ui.secure_info.service import (
    RegistrationMetadata,
    SilverJubileeService,
)

router = APIRouter(prefix="/api/secure-info", tags=["Silver Jubilee Information"])
_AUTH_COOKIE_NAME = "visionsearch_secure_session"
_CSRF_COOKIE_NAME = "visionsearch_secure_csrf"
_COOKIE_PATH = "/api"


def _session_ttl_seconds() -> int:
    try:
        value = int(os.getenv("VISIONSEARCH_AUTH_SESSION_TTL_MINUTES", "60")) * 60
    except ValueError as error:
        raise RuntimeError("The secure-information session lifetime is misconfigured.") from error
    if value < 60 or value > 24 * 60 * 60:
        raise RuntimeError("The secure-information session lifetime is misconfigured.")
    return value


def _cookie_secure() -> bool:
    value = os.getenv("VISIONSEARCH_AUTH_COOKIE_SECURE", "true").strip().lower()
    if value not in {"true", "false", "1", "0"}:
        raise RuntimeError("The secure-information cookie setting is misconfigured.")
    return value in {"true", "1"}


def get_authentication_store() -> AuthenticationStore:
    store = AuthenticationStore()
    store.initialize()
    return store


def get_registration_service() -> SilverJubileeService:
    database = SilverJubileeDatabase(DEFAULT_DATABASE_PATH)
    database.initialize()
    return SilverJubileeService(database)


def _csrf_matches(
    request: Request,
    submitted_token: str | None,
    stored_token_hash: str | None = None,
) -> bool:
    cookie_token = request.cookies.get(_CSRF_COOKIE_NAME)
    if (
        not cookie_token
        or not submitted_token
        or not hmac.compare_digest(cookie_token, submitted_token)
    ):
        return False
    if stored_token_hash is not None:
        submitted_hash = hashlib.sha256(submitted_token.encode("ascii", errors="ignore")).hexdigest()
        return hmac.compare_digest(submitted_hash, stored_token_hash)
    return True


def get_current_user(
    request: Request,
    store: Annotated[AuthenticationStore, Depends(get_authentication_store)],
) -> AuthenticatedUser:
    token = request.cookies.get(_AUTH_COOKIE_NAME)
    if token is None:
        raise HTTPException(status_code=401, detail="Authentication is required.")
    try:
        session = store.get_session(token)
    except AuthenticationStoreError as error:
        raise HTTPException(status_code=503, detail="Authentication is temporarily unavailable.") from error
    if session is None:
        raise HTTPException(status_code=401, detail="Authentication is required.")
    return session.user


def require_authenticated_user(
    request: Request,
    current_user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    store: Annotated[AuthenticationStore, Depends(get_authentication_store)],
    x_csrf_token: Annotated[str | None, Header(alias="X-CSRF-Token")] = None,
) -> AuthenticatedUser:
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        token = request.cookies.get(_AUTH_COOKIE_NAME)
        try:
            session = store.get_session(token) if token else None
        except AuthenticationStoreError as error:
            raise HTTPException(status_code=503, detail="Authentication is temporarily unavailable.") from error
        csrf_hash = session.csrf_token_hash if session is not None else None
        if not _csrf_matches(request, x_csrf_token, csrf_hash):
            raise HTTPException(status_code=403, detail="A valid CSRF token is required.")
    return current_user


def require_admin(
    request: Request,
    current_user: AuthenticatedUser = Depends(get_current_user),
    store: AuthenticationStore = Depends(get_authentication_store),
    x_csrf_token: str | None = Header(default=None, alias="X-CSRF-Token"),
) -> AuthenticatedUser:
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Administrator access is required.")
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        token = request.cookies.get(_AUTH_COOKIE_NAME)
        try:
            session = store.get_session(token) if token else None
        except AuthenticationStoreError as error:
            raise HTTPException(status_code=503, detail="Authentication is temporarily unavailable.") from error
        csrf_hash = session.csrf_token_hash if session is not None else None
        if not _csrf_matches(request, x_csrf_token, csrf_hash):
            raise HTTPException(status_code=403, detail="A valid CSRF token is required.")
    return current_user


def _metadata_response(metadata: RegistrationMetadata) -> dict:
    return asdict(metadata)


def _managed_user_response(user: ManagedUser) -> dict:
    return asdict(user)


def _user_management_payload(body: Any, fields: set[str]) -> dict[str, Any]:
    if not isinstance(body, dict) or set(body) != fields:
        raise HTTPException(status_code=400, detail="The user-management request is invalid.")
    return body


def _raise_user_management_error(error: Exception) -> None:
    if isinstance(error, DuplicateUserError):
        raise HTTPException(status_code=409, detail="That username is already provisioned.") from error
    if isinstance(error, UserNotFoundError):
        raise HTTPException(status_code=404, detail="Regular user not found.") from error
    if isinstance(error, SelfUserModificationError):
        raise HTTPException(
            status_code=409,
            detail="Administrators cannot modify their own account.",
        ) from error
    if isinstance(error, ValueError):
        raise HTTPException(status_code=400, detail=str(error)) from error
    if isinstance(error, AuthenticationStoreError):
        raise HTTPException(
            status_code=503,
            detail="Authentication is temporarily unavailable.",
        ) from error
    raise HTTPException(
        status_code=503,
        detail="The user-management request could not be completed.",
    ) from error


def validate_indexed_image_reference(image_reference: str, image_hash: str) -> str:
    """Return a project-relative reference only if it matches an active indexed file."""
    from visionsearch_web_ui import api as main_api

    validated_reference = SilverJubileeService._validate_image_reference(image_reference)
    reference_path = (main_api.ROOT / validated_reference).resolve()
    try:
        reference_path.relative_to(main_api.ROOT.resolve())
    except ValueError as error:
        raise ValueError("Photograph reference is not in the active image collection.") from error

    indexed_paths = main_api.get_index()[1]
    indexed_match = any(
        main_api.safe_result_path(indexed_path) == reference_path
        for indexed_path in indexed_paths
    )
    if not indexed_match or main_api._supported_image_media_type(reference_path) is None:
        raise ValueError("Photograph must be a supported image in the active image collection.")
    if main_api._sha256_file(reference_path) != image_hash:
        raise ValueError("Photograph hash does not match the indexed image.")
    return reference_path.relative_to(main_api.ROOT.resolve()).as_posix()


def _raise_registration_error(error: Exception) -> None:
    if isinstance(error, DuplicatePhotographError):
        raise HTTPException(status_code=409, detail=str(error)) from error
    if isinstance(error, RecordNotFoundError):
        raise HTTPException(status_code=404, detail="Registration not found.") from error
    if isinstance(error, ValueError):
        raise HTTPException(status_code=400, detail=str(error)) from error
    if isinstance(
        error,
        (SecureInfoDatabaseError, AuthenticationStoreError, EncryptionConfigurationError),
    ):
        raise HTTPException(status_code=503, detail="Secure information is temporarily unavailable.") from error
    raise HTTPException(status_code=503, detail="Secure information could not be processed.") from error


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=12, max_length=1024)


class RegistrationRequest(BaseModel):
    image_reference: str = Field(min_length=1, max_length=4096)
    image_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    title: str = Field(min_length=1, max_length=200)
    information: str = Field(min_length=1, max_length=100_000)


class RegistrationUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    information: str = Field(min_length=1, max_length=100_000)


class RegistrationMetadataResponse(BaseModel):
    id: int
    image_path: str
    image_hash: str
    title: str
    created_at: datetime
    updated_at: datetime


class UserManagementResponse(BaseModel):
    id: int
    username: str
    role: str
    is_active: bool
    created_at: datetime


@router.get("/auth/csrf")
def issue_csrf_token(response: Response) -> dict[str, str]:
    token = secrets.token_urlsafe(32)
    response.set_cookie(
        _CSRF_COOKIE_NAME,
        token,
        httponly=False,
        secure=_cookie_secure(),
        samesite="lax",
        path=_COOKIE_PATH,
        max_age=3600,
    )
    return {"csrf_token": token}


@router.post("/auth/login")
def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    store: Annotated[AuthenticationStore, Depends(get_authentication_store)],
    x_csrf_token: Annotated[str | None, Header(alias="X-CSRF-Token")] = None,
) -> dict:
    if not _csrf_matches(request, x_csrf_token):
        raise HTTPException(status_code=403, detail="A valid CSRF token is required.")
    try:
        user = store.verify_credentials(body.username, body.password)
    except AuthenticationStoreError as error:
        raise HTTPException(status_code=503, detail="Authentication is temporarily unavailable.") from error
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    try:
        ttl_seconds = _session_ttl_seconds()
        session_token, csrf_token, expires_at = store.create_session(
            user.id,
            ttl_seconds,
        )
    except (AuthenticationStoreError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail="Authentication is temporarily unavailable.") from error

    secure = _cookie_secure()
    response.set_cookie(
        _AUTH_COOKIE_NAME,
        session_token,
        httponly=True,
        secure=secure,
        samesite="lax",
        path=_COOKIE_PATH,
        max_age=ttl_seconds,
    )
    response.set_cookie(
        _CSRF_COOKIE_NAME,
        csrf_token,
        httponly=False,
        secure=secure,
        samesite="lax",
        path=_COOKIE_PATH,
        max_age=max(1, expires_at - int(time.time())),
    )
    return {
        "user": {"username": user.username, "role": user.role},
        "csrf_token": csrf_token,
        "expires_at": expires_at,
    }


@router.post("/auth/logout", status_code=204)
def logout(
    request: Request,
    response: Response,
    store: Annotated[AuthenticationStore, Depends(get_authentication_store)],
    x_csrf_token: Annotated[str | None, Header(alias="X-CSRF-Token")] = None,
) -> Response:
    token = request.cookies.get(_AUTH_COOKIE_NAME)
    session = None
    if token:
        try:
            session = store.get_session(token)
        except AuthenticationStoreError as error:
            raise HTTPException(status_code=503, detail="Authentication is temporarily unavailable.") from error
    csrf_hash = session.csrf_token_hash if session is not None else None
    if not _csrf_matches(request, x_csrf_token, csrf_hash):
        raise HTTPException(status_code=403, detail="A valid CSRF token is required.")
    if token:
        try:
            store.revoke_session(token)
        except AuthenticationStoreError as error:
            raise HTTPException(status_code=503, detail="Authentication is temporarily unavailable.") from error
    secure = _cookie_secure()
    response.delete_cookie(
        _AUTH_COOKIE_NAME,
        path=_COOKIE_PATH,
        secure=secure,
        httponly=True,
        samesite="lax",
    )
    response.delete_cookie(
        _CSRF_COOKIE_NAME,
        path=_COOKIE_PATH,
        secure=secure,
        httponly=False,
        samesite="lax",
    )
    response.status_code = 204
    return response


@router.get("/auth/me")
def current_user(
    user: Annotated[AuthenticatedUser, Depends(require_authenticated_user)],
) -> dict[str, str]:
    return {"username": user.username, "role": user.role}


@router.post(
    "/users",
    response_model=UserManagementResponse,
    status_code=201,
)
def create_managed_user(
    body: Annotated[Any, Body()],
    _: Annotated[AuthenticatedUser, Depends(require_admin)],
    store: Annotated[AuthenticationStore, Depends(get_authentication_store)],
) -> dict:
    payload = _user_management_payload(body, {"username", "password"})
    username = payload["username"]
    password = payload["password"]
    if not isinstance(username, str) or not isinstance(password, str):
        raise HTTPException(
            status_code=400,
            detail="Username and password must be strings.",
        )
    try:
        user = store.create_user(username, password)
        managed_user = store.get_managed_user(user.id)
        if managed_user is None:
            raise UserNotFoundError("Regular user not found.")
        return _managed_user_response(managed_user)
    except (
        DuplicateUserError,
        UserNotFoundError,
        SelfUserModificationError,
        AuthenticationStoreError,
        ValueError,
    ) as error:
        _raise_user_management_error(error)


@router.get("/users", response_model=list[UserManagementResponse])
def list_managed_users(
    _: Annotated[AuthenticatedUser, Depends(require_admin)],
    store: Annotated[AuthenticationStore, Depends(get_authentication_store)],
) -> list[dict]:
    try:
        return [_managed_user_response(user) for user in store.list_users()]
    except AuthenticationStoreError as error:
        _raise_user_management_error(error)


@router.patch(
    "/users/{user_id}/status",
    response_model=UserManagementResponse,
)
def update_managed_user_status(
    user_id: int,
    body: Annotated[Any, Body()],
    administrator: Annotated[AuthenticatedUser, Depends(require_admin)],
    store: Annotated[AuthenticationStore, Depends(get_authentication_store)],
) -> dict:
    payload = _user_management_payload(body, {"is_active"})
    is_active = payload["is_active"]
    if not isinstance(is_active, bool):
        raise HTTPException(status_code=400, detail="Account status must be active or inactive.")
    try:
        user = store.set_user_active(user_id, is_active, administrator.id)
        return _managed_user_response(user)
    except (UserNotFoundError, SelfUserModificationError, AuthenticationStoreError, ValueError) as error:
        _raise_user_management_error(error)


@router.post(
    "/users/{user_id}/password",
    response_model=UserManagementResponse,
)
def reset_managed_user_password(
    user_id: int,
    body: Annotated[Any, Body()],
    administrator: Annotated[AuthenticatedUser, Depends(require_admin)],
    store: Annotated[AuthenticationStore, Depends(get_authentication_store)],
) -> dict:
    payload = _user_management_payload(body, {"password"})
    password = payload["password"]
    if not isinstance(password, str):
        raise HTTPException(status_code=400, detail="Password must be a string.")
    try:
        store.reset_user_password(user_id, password, administrator.id)
        user = store.get_managed_user(user_id)
        if user is None:
            raise UserNotFoundError("Regular user not found.")
        return _managed_user_response(user)
    except (UserNotFoundError, SelfUserModificationError, AuthenticationStoreError, ValueError) as error:
        _raise_user_management_error(error)


@router.post(
    "/registrations",
    response_model=RegistrationMetadataResponse,
    status_code=201,
)
def create_registration(
    body: RegistrationRequest,
    _: Annotated[AuthenticatedUser, Depends(require_admin)],
    service: Annotated[SilverJubileeService, Depends(get_registration_service)],
) -> dict:
    try:
        validated_reference = validate_indexed_image_reference(
            body.image_reference,
            body.image_hash,
        )

        metadata = service.register_information(
            image_reference=validated_reference,
            image_hash=body.image_hash,
            title=body.title,
            information=body.information,
        )
        return _metadata_response(metadata)
    except (
        ValueError,
        DuplicatePhotographError,
        SecureInfoDatabaseError,
        EncryptionConfigurationError,
    ) as error:
        _raise_registration_error(error)


@router.get("/registrations", response_model=list[RegistrationMetadataResponse])
def list_registrations(
    _: Annotated[AuthenticatedUser, Depends(require_admin)],
    service: Annotated[SilverJubileeService, Depends(get_registration_service)],
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    try:
        return [_metadata_response(item) for item in service.list_registrations(limit, offset)]
    except (ValueError, SecureInfoDatabaseError) as error:
        _raise_registration_error(error)


@router.put(
    "/registrations/{record_id}",
    response_model=RegistrationMetadataResponse,
)
def update_registration(
    record_id: int,
    body: RegistrationUpdateRequest,
    _: Annotated[AuthenticatedUser, Depends(require_admin)],
    service: Annotated[SilverJubileeService, Depends(get_registration_service)],
) -> dict:
    try:
        return _metadata_response(
            service.update_information(record_id, body.title, body.information)
        )
    except (
        ValueError,
        RecordNotFoundError,
        SecureInfoDatabaseError,
        EncryptionConfigurationError,
    ) as error:
        _raise_registration_error(error)


@router.delete("/registrations/{record_id}", status_code=204)
def delete_registration(
    record_id: int,
    _: Annotated[AuthenticatedUser, Depends(require_admin)],
    service: Annotated[SilverJubileeService, Depends(get_registration_service)],
) -> Response:
    try:
        if not service.delete_registration(record_id):
            raise HTTPException(status_code=404, detail="Registration not found.")
    except ValueError as error:
        _raise_registration_error(error)
    return Response(status_code=204)
