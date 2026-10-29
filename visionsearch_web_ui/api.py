import hashlib
import io
import json
import logging
import os
import re
import tempfile
import threading
import uuid
import warnings
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from models.clip_model import load_clip_model
from services.search_service import load_index, search_loaded, search_text_loaded

ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "index" / "faiss.index"
PATHS_PATH = ROOT / "index" / "image_paths.txt"
EMBEDDINGS_PATH = ROOT / "index" / "image_embeddings.npy"
INDEX_TRANSACTION_PATH = ROOT / "index" / ".upload-index-transaction.json"
UPLOAD_DIR = ROOT / "uploads"
FRONTEND = Path(__file__).resolve().parent / "frontend"
MAX_UPLOAD_FILE_BYTES = int(os.getenv("VISIONSEARCH_MAX_UPLOAD_FILE_MB", "10")) * 1024 * 1024
MAX_UPLOAD_REQUEST_BYTES = int(os.getenv("VISIONSEARCH_MAX_UPLOAD_REQUEST_MB", "25")) * 1024 * 1024
MAX_UPLOAD_FILES_PER_REQUEST = int(os.getenv("VISIONSEARCH_MAX_UPLOAD_FILES_PER_REQUEST", "20"))
SUPPORTED_IMAGE_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}

app = FastAPI(title="VisionSearch AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_index_data = None
_indexed_digest_paths: tuple[str, ...] | None = None
_indexed_content_digests: set[str] = set()
_index_lock = threading.RLock()
logger = logging.getLogger(__name__)

class TextQuery(BaseModel):
    query: str
    top_k: int = 10

def get_model():
    global _model
    if _model is None:
        _model = load_clip_model()
    return _model

def get_index():
    global _index_data
    with _index_lock:
        _recover_index_transaction()
        if _index_data is None:
            if not INDEX_PATH.exists() or not PATHS_PATH.exists():
                raise HTTPException(
                    status_code=400,
                    detail="Image index is missing. Build it using your existing app first, then restart this API."
                )
            _index_data = load_index(INDEX_PATH, PATHS_PATH)
        return _index_data

def _format_size_limit(byte_count: int) -> str:
    megabyte = 1024 * 1024
    if byte_count % megabyte == 0:
        return f"{byte_count // megabyte} MB"
    return f"{byte_count} bytes"

def _safe_relative_path(value: str) -> tuple[Path, str]:
    normalized = value.replace("\\", "/")
    if (
        not normalized
        or normalized.startswith("/")
        or re.match(r"^[A-Za-z]:", normalized)
        or "\x00" in normalized
    ):
        raise ValueError("Invalid relative image path.")

    parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Invalid relative image path.")

    clean_parts = []
    for part in parts[:-1]:
        clean = re.sub(r"[^A-Za-z0-9._-]+", "_", part).strip(" ._")
        if not clean:
            raise ValueError("Invalid folder name.")
        clean_parts.append(clean)

    clean_name = re.sub(r"[^A-Za-z0-9._-]+", "_", parts[-1]).strip(" ._")
    if clean_name in {"", ".", ".."}:
        raise ValueError("Invalid filename.")
    return Path(*clean_parts), clean_name

def _validate_image(path: Path) -> tuple[str, str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image_format = image.format
                image.verify()
            with Image.open(path) as image:
                image.load()
    except Exception as error:
        raise ValueError("The file is corrupted or is not a supported image.") from error

    if image_format not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError("Supported image formats are JPEG, PNG, and WebP.")
    return image_format, SUPPORTED_IMAGE_FORMATS[image_format]

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _unique_upload_path(relative_directory: Path, clean_name: str, extension: str, digest: str) -> Path:
    upload_root = UPLOAD_DIR.resolve()
    destination_directory = (UPLOAD_DIR / relative_directory).resolve()
    try:
        destination_directory.relative_to(upload_root)
    except ValueError as error:
        raise ValueError("Invalid upload folder path.") from error

    destination_directory.mkdir(parents=True, exist_ok=True)
    stem = Path(clean_name).stem.strip(" ._") or "image"
    base_name = f"{stem}__sha256_{digest}"
    candidate = destination_directory / f"{base_name}{extension}"
    collision = 1
    while candidate.exists():
        candidate = destination_directory / f"{base_name}_{collision}{extension}"
        collision += 1

    resolved = candidate.resolve()
    try:
        resolved.relative_to(upload_root)
    except ValueError as error:
        raise ValueError("Invalid upload destination.") from error
    return resolved

def _recover_index_transaction() -> None:
    if not INDEX_TRANSACTION_PATH.exists():
        return

    try:
        transaction = json.loads(INDEX_TRANSACTION_PATH.read_text(encoding="utf-8"))
        artifact_entries = transaction["artifacts"]
        expected_destinations = {
            path.name: path for path in (INDEX_PATH, PATHS_PATH, EMBEDDINGS_PATH)
        }
        if {entry["name"] for entry in artifact_entries} != set(expected_destinations):
            raise ValueError("The upload transaction has an invalid artifact list.")

        for entry in reversed(artifact_entries):
            destination = expected_destinations[entry["name"]]
            backup_name = entry["backup"]
            if backup_name is not None:
                if Path(backup_name).name != backup_name:
                    raise ValueError("The upload transaction has an invalid backup path.")
                backup = destination.with_name(backup_name)
                if backup.exists():
                    if destination.exists():
                        destination.unlink()
                    os.replace(backup, destination)
                elif not destination.exists():
                    raise OSError(f"Neither the original nor its backup exists for {destination.name}.")
            elif destination.exists():
                destination.unlink()

        upload_root = UPLOAD_DIR.resolve()
        for relative_path in transaction["uploads"]:
            uploaded_path = (ROOT / relative_path).resolve()
            uploaded_path.relative_to(upload_root)
            uploaded_path.unlink(missing_ok=True)

        INDEX_TRANSACTION_PATH.unlink()
    except Exception as error:
        logger.exception("Could not recover an interrupted image-index update.")
        raise RuntimeError("The interrupted image-index update could not be recovered.") from error

def _write_index_artifacts(
    index,
    embeddings: np.ndarray,
    paths: list[str],
    upload_files: list[tuple[Path, Path]] | None = None,
) -> None:
    destinations = [INDEX_PATH, PATHS_PATH, EMBEDDINGS_PATH]
    temporary_paths: list[Path] = []
    transaction_id = uuid.uuid4().hex
    upload_files = upload_files or []
    marker_temporary_path = None

    try:
        for destination in destinations:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            temporary_paths.append(temporary_path)
            if destination == INDEX_PATH:
                faiss.write_index(index, str(temporary_path))
            elif destination == PATHS_PATH:
                temporary_path.write_text("\n".join(paths) + "\n", encoding="utf-8")
            else:
                with temporary_path.open("wb") as output:
                    np.save(output, embeddings, allow_pickle=False)

        transaction = {
            "artifacts": [
                {
                    "name": destination.name,
                    "backup": (
                        f".{destination.name}.{transaction_id}.bak"
                        if destination.exists()
                        else None
                    ),
                }
                for destination in destinations
            ],
            "uploads": [
                destination.resolve().relative_to(ROOT.resolve()).as_posix()
                for _, destination in upload_files
            ],
        }
        INDEX_TRANSACTION_PATH.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=".upload-index-transaction.",
            suffix=".tmp",
            dir=INDEX_TRANSACTION_PATH.parent,
            delete=False,
        ) as marker_temporary:
            marker_temporary_path = Path(marker_temporary.name)
            marker_temporary.write(json.dumps(transaction).encode("utf-8"))
        os.replace(marker_temporary_path, INDEX_TRANSACTION_PATH)

        for temporary_upload, destination in upload_files:
            if destination.exists():
                raise FileExistsError("An upload destination unexpectedly already exists.")
            os.rename(temporary_upload, destination)

        for destination, temporary_path in zip(destinations, temporary_paths):
            entry = next(item for item in transaction["artifacts"] if item["name"] == destination.name)
            backup_name = entry["backup"]
            if backup_name is not None:
                backup = destination.with_name(backup_name)
                os.replace(destination, backup)
            os.replace(temporary_path, destination)

        INDEX_TRANSACTION_PATH.unlink()
        for entry in transaction["artifacts"]:
            if entry["backup"] is not None:
                backup = INDEX_PATH.parent / entry["backup"]
                try:
                    backup.unlink()
                except OSError:
                    logger.warning("Could not remove a temporary index backup.")
    except Exception:
        if INDEX_TRANSACTION_PATH.exists():
            _recover_index_transaction()
        raise
    finally:
        for path in temporary_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove a temporary index file.")
        if marker_temporary_path is not None:
            try:
                marker_temporary_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove a temporary transaction marker.")

def safe_result_path(value: Any) -> Path | None:
    # Existing search service commonly returns (path, score) pairs; support dicts too.
    if isinstance(value, dict):
        value = value.get("path") or value.get("image_path") or value.get("filepath") or value.get("file")
    elif isinstance(value, (tuple, list)) and value:
        value = value[0]
    if not isinstance(value, (str, Path)):
        return None
    p = Path(value)
    if not p.is_absolute():
        p = ROOT / p
    try:
        p = p.resolve()
        p.relative_to(ROOT.resolve())
    except (ValueError, OSError):
        return None
    return p if p.is_file() else None

def normalize_results(results):
    output = []
    for item in results:
        p = safe_result_path(item)
        if p:
            try:
                rel = p.relative_to(ROOT.resolve()).as_posix()
                score = None
                if isinstance(item, dict):
                    score = item.get("score") or item.get("similarity")
                elif isinstance(item, (tuple, list)) and len(item) > 1:
                    score = item[1]
                output.append({
                    "name": p.name,
                    "category": p.parent.name,
                    "image_url": "/image/" + rel,
                    "score": float(score) if score is not None else None,
                    "category_match": bool(
                        isinstance(item, dict) and item.get("category_match")
                    ),
                })
            except (TypeError, ValueError):
                output.append({"name": p.name, "image_url": "/image/" + rel, "score": None})
    return output

def _get_indexed_content_digests(paths: list[str]) -> set[str]:
    global _indexed_digest_paths, _indexed_content_digests
    path_snapshot = tuple(paths)
    if path_snapshot != _indexed_digest_paths:
        digests = set()
        for image_path in paths:
            resolved_path = safe_result_path(image_path)
            if resolved_path is None:
                continue
            try:
                digest = _sha256_file(resolved_path)
            except OSError as error:
                logger.exception("Could not verify an indexed image for duplicate detection.")
                raise HTTPException(
                    status_code=503,
                    detail="Existing images could not be checked for duplicates; verify that indexed image files are available.",
                ) from error
            digests.add(digest)
        _indexed_digest_paths = path_snapshot
        _indexed_content_digests = digests
    return set(_indexed_content_digests)

@app.get("/api/health")
def health():
    return {"status": "ok", "index_ready": INDEX_PATH.exists() and PATHS_PATH.exists()}

@app.get("/api/upload/limits")
def upload_limits():
    if (
        MAX_UPLOAD_FILE_BYTES < 1
        or MAX_UPLOAD_REQUEST_BYTES < 1
        or MAX_UPLOAD_FILES_PER_REQUEST < 1
    ):
        raise HTTPException(status_code=500, detail="Upload size limits are configured incorrectly.")
    return {
        "max_file_bytes": MAX_UPLOAD_FILE_BYTES,
        "max_request_bytes": MAX_UPLOAD_REQUEST_BYTES,
        "max_files": MAX_UPLOAD_FILES_PER_REQUEST,
    }

@app.post("/api/search/text")
def text_search(body: TextQuery):
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Enter a search description.")
    index, paths = get_index()
    model = get_model()
    embedding = model.encode_text(query)
    results = search_text_loaded(
        query,
        index,
        paths,
        embedding,
        top_k=max(1, min(body.top_k, 50)),
    )
    return {"query": query, "results": normalize_results(results)}

@app.post("/api/search/image")
async def image_search(file: UploadFile = File(...), top_k: int = 10):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid image file.")
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="The uploaded file could not be opened as an image.")
    index, paths = get_index()
    model = get_model()
    embedding = model.encode_image(image)
    results = search_loaded(index, paths, embedding, top_k=max(1, min(top_k, 50)))
    return {"results": normalize_results(results)}

@app.post("/api/upload")
async def upload_images(
    files: list[UploadFile] = File(...),
    relative_paths: list[str] = Form(default=[]),
):
    global _index_data, _indexed_digest_paths, _indexed_content_digests
    if not files:
        raise HTTPException(status_code=400, detail="Choose at least one image to upload.")
    if relative_paths and len(relative_paths) != len(files):
        raise HTTPException(status_code=400, detail="The upload paths do not match the selected files.")
    if (
        MAX_UPLOAD_FILE_BYTES < 1
        or MAX_UPLOAD_REQUEST_BYTES < 1
        or MAX_UPLOAD_FILES_PER_REQUEST < 1
    ):
        raise HTTPException(status_code=500, detail="Upload size limits are configured incorrectly.")
    if len(files) > MAX_UPLOAD_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"Upload at most {MAX_UPLOAD_FILES_PER_REQUEST} images per batch.",
        )
    file_sizes = [upload.size for upload in files]
    if all(file_size is not None for file_size in file_sizes):
        total_file_bytes = sum(file_size for file_size in file_sizes if file_size is not None)
        if total_file_bytes > MAX_UPLOAD_REQUEST_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Batch exceeds the {_format_size_limit(MAX_UPLOAD_REQUEST_BYTES)} total upload limit.",
            )

    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        staging_directory = UPLOAD_DIR / ".tmp"
        staging_directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        logger.exception("Upload storage is unavailable.")
        raise HTTPException(status_code=500, detail="Upload storage is unavailable.") from error
    staged: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    skipped_files: list[dict[str, str]] = []
    file_results: list[dict[str, str]] = []
    request_bytes = 0

    try:
        for position, upload in enumerate(files):
            original_name = Path((upload.filename or "image").replace("\\", "/")).name or "image"
            client_path = relative_paths[position] if relative_paths else (upload.filename or "image")
            file_result = {
                "filename": original_name,
                "relative_path": client_path,
                "status": "failed",
            }
            file_results.append(file_result)
            file_size = upload.size
            if file_size is not None:
                if file_size > MAX_UPLOAD_FILE_BYTES:
                    error_message = (
                        f"File exceeds the {_format_size_limit(MAX_UPLOAD_FILE_BYTES)} per-file limit."
                    )
                    file_result["error"] = error_message
                    errors.append({
                        "filename": original_name,
                        "relative_path": client_path,
                        "error": error_message,
                    })
                    continue
                if request_bytes + file_size > MAX_UPLOAD_REQUEST_BYTES:
                    error_message = (
                        f"Request exceeds the {_format_size_limit(MAX_UPLOAD_REQUEST_BYTES)} total upload limit."
                    )
                    file_result["error"] = error_message
                    errors.append({
                        "filename": original_name,
                        "relative_path": client_path,
                        "error": error_message,
                    })
                    continue
                request_bytes += file_size

            try:
                relative_directory, clean_name = _safe_relative_path(client_path)
            except ValueError as error:
                file_result["error"] = str(error)
                errors.append({
                    "filename": original_name,
                    "relative_path": client_path,
                    "error": str(error),
                })
                continue

            temporary_path = None
            file_bytes = 0
            try:
                with tempfile.NamedTemporaryFile(
                    prefix="upload-",
                    suffix=".tmp",
                    dir=staging_directory,
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    while True:
                        chunk = await upload.read(1024 * 1024)
                        if not chunk:
                            break
                        file_bytes += len(chunk)
                        if file_bytes > MAX_UPLOAD_FILE_BYTES:
                            raise ValueError(
                                f"File exceeds the {_format_size_limit(MAX_UPLOAD_FILE_BYTES)} per-file limit."
                            )
                        if file_size is None and request_bytes + file_bytes > MAX_UPLOAD_REQUEST_BYTES:
                            raise ValueError(
                                f"Request exceeds the {_format_size_limit(MAX_UPLOAD_REQUEST_BYTES)} total upload limit."
                            )
                        temporary.write(chunk)

                if file_bytes == 0:
                    raise ValueError("The selected file is empty.")
                if file_size is None:
                    request_bytes += file_bytes
                _, extension = _validate_image(temporary_path)
                digest = _sha256_file(temporary_path)
                staged.append({
                    "temporary_path": temporary_path,
                    "directory": relative_directory,
                    "name": clean_name,
                    "extension": extension,
                    "digest": digest,
                    "filename": original_name,
                    "relative_path": client_path,
                    "position": position,
                })
            except ValueError as error:
                file_result["error"] = str(error)
                errors.append({
                    "filename": original_name,
                    "relative_path": client_path,
                    "error": str(error),
                })
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            except OSError:
                logger.exception("Could not stage an uploaded image.")
                error_message = "The image could not be stored temporarily."
                file_result["error"] = error_message
                errors.append({
                    "filename": original_name,
                    "relative_path": client_path,
                    "error": error_message,
                })
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

        if not staged:
            failed_count = sum(result["status"] == "failed" for result in file_results)
            return {
                "uploaded_count": 0,
                "indexed_count": 0,
                "successful_count": 0,
                "duplicate_count": 0,
                "skipped_count": 0,
                "failed_count": failed_count,
                "skipped_files": skipped_files,
                "errors": errors,
                "file_results": file_results,
            }

        with _index_lock:
            _recover_index_transaction()
            if _index_data is None:
                _index_data = get_index()
            current_index, current_paths = _index_data

            try:
                existing_embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=False)
            except (OSError, ValueError) as error:
                logger.exception("Could not load persisted image embeddings for upload.")
                raise HTTPException(
                    status_code=503,
                    detail="The saved embedding collection is missing or damaged; rebuild the existing index before uploading.",
                ) from error

            if (
                existing_embeddings.ndim != 2
                or existing_embeddings.shape[0] != len(current_paths)
                or existing_embeddings.shape[1] != current_index.d
                or current_index.ntotal != len(current_paths)
            ):
                raise HTTPException(
                    status_code=503,
                    detail="The saved index, embeddings, and image paths are inconsistent; rebuild the existing index before uploading.",
                )

            known_digests = _get_indexed_content_digests(current_paths)
            model = get_model()
            new_vectors = []
            new_paths = []
            new_files = []
            seen_digests = set(known_digests)

            for item in staged:
                digest = item["digest"]
                if digest in seen_digests:
                    file_result = file_results[item["position"]]
                    file_result["status"] = "duplicate"
                    file_result["reason"] = "Identical image content is already indexed."
                    skipped_files.append({
                        "filename": item["filename"],
                        "relative_path": item["relative_path"],
                        "reason": file_result["reason"],
                    })
                    continue
                try:
                    with Image.open(item["temporary_path"]) as image:
                        vector = np.asarray(model.encode_image(image.convert("RGB")), dtype=np.float32)
                    if vector.ndim == 1:
                        vector = vector.reshape(1, -1)
                    if vector.shape != (1, current_index.d) or not np.isfinite(vector).all():
                        raise ValueError("CLIP returned an invalid embedding.")
                    norm = np.linalg.norm(vector, axis=1, keepdims=True)
                    if np.any(norm <= 1e-12):
                        raise ValueError("CLIP returned an empty embedding.")
                    vector = vector / norm
                except Exception:
                    logger.exception("CLIP could not index uploaded image %s.", item["filename"])
                    error_message = "CLIP could not generate an embedding for this image."
                    file_result = file_results[item["position"]]
                    file_result["error"] = error_message
                    errors.append({
                        "filename": item["filename"],
                        "relative_path": item["relative_path"],
                        "error": error_message,
                    })
                    continue

                try:
                    destination = _unique_upload_path(
                        item["directory"], item["name"], item["extension"], digest
                    )
                except (OSError, ValueError):
                    logger.exception("Could not prepare a safe destination for uploaded image %s.", item["filename"])
                    error_message = "A safe storage location could not be prepared for this image."
                    file_result = file_results[item["position"]]
                    file_result["error"] = error_message
                    errors.append({
                        "filename": item["filename"],
                        "relative_path": item["relative_path"],
                        "error": error_message,
                    })
                    continue

                file_results[item["position"]]["status"] = "indexed"
                new_vectors.append(vector[0])
                new_paths.append(str(destination))
                new_files.append((item["temporary_path"], destination, item["filename"]))
                seen_digests.add(digest)

            if new_vectors:
                appended_embeddings = np.asarray(new_vectors, dtype=np.float32)
                combined_embeddings = np.concatenate((existing_embeddings, appended_embeddings), axis=0)
                updated_index = faiss.clone_index(current_index)
                updated_index.add(appended_embeddings)
                combined_paths = [*current_paths, *new_paths]
                try:
                    _write_index_artifacts(
                        updated_index,
                        combined_embeddings,
                        combined_paths,
                        [(temporary_path, destination) for temporary_path, destination, _ in new_files],
                    )
                except Exception as error:
                    logger.exception("Could not persist uploaded images and index artifacts.")
                    raise HTTPException(
                        status_code=500,
                        detail="The upload could not be indexed; the existing collection was preserved.",
                    ) from error

                _index_data = (updated_index, combined_paths)
                _indexed_content_digests.update(seen_digests)
                _indexed_digest_paths = tuple(combined_paths)

            successful_count = sum(result["status"] == "indexed" for result in file_results)
            duplicate_count = sum(result["status"] == "duplicate" for result in file_results)
            failed_count = sum(result["status"] == "failed" for result in file_results)
            return {
                "uploaded_count": successful_count,
                "indexed_count": successful_count,
                "successful_count": successful_count,
                "duplicate_count": duplicate_count,
                "skipped_count": duplicate_count,
                "failed_count": failed_count,
                "skipped_files": skipped_files,
                "errors": errors,
                "file_results": file_results,
            }
    finally:
        for item in staged:
            try:
                item["temporary_path"].unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove a staged image upload.")
        for upload in files:
            await upload.close()

@app.get("/image/{image_path:path}")
def serve_image(image_path: str):
    p = (ROOT / image_path).resolve()
    try:
        p.relative_to(ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid image path.")
    if not p.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(p)

app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
