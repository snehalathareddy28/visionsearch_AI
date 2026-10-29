import io
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from visionsearch_web_ui import api


class FakeClipModel:
    def encode_image(self, image):
        vector = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
        return (vector / np.linalg.norm(vector)).reshape(1, -1)

    def encode_text(self, text):
        vector = np.asarray([1.0, 0.0, 0.0] if "legacy" not in text.lower() else [0.0, 0.0, 1.0])
        return vector.astype(np.float32).reshape(1, -1)


def image_bytes(color, image_format="PNG", size=(16, 16)):
    output = io.BytesIO()
    Image.new("RGB", size, color=color).save(output, format=image_format)
    return output.getvalue()


class UploadApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.old_values = {
            "ROOT": api.ROOT,
            "INDEX_PATH": api.INDEX_PATH,
            "PATHS_PATH": api.PATHS_PATH,
            "EMBEDDINGS_PATH": api.EMBEDDINGS_PATH,
            "INDEX_TRANSACTION_PATH": api.INDEX_TRANSACTION_PATH,
            "UPLOAD_DIR": api.UPLOAD_DIR,
            "MAX_UPLOAD_FILE_BYTES": api.MAX_UPLOAD_FILE_BYTES,
            "MAX_UPLOAD_REQUEST_BYTES": api.MAX_UPLOAD_REQUEST_BYTES,
            "MAX_UPLOAD_FILES_PER_REQUEST": api.MAX_UPLOAD_FILES_PER_REQUEST,
            "_index_data": api._index_data,
            "_indexed_digest_paths": api._indexed_digest_paths,
            "_indexed_content_digests": set(api._indexed_content_digests),
            "_model": api._model,
        }
        api.ROOT = self.root
        api.INDEX_PATH = self.root / "index" / "faiss.index"
        api.PATHS_PATH = self.root / "index" / "image_paths.txt"
        api.EMBEDDINGS_PATH = self.root / "index" / "image_embeddings.npy"
        api.INDEX_TRANSACTION_PATH = self.root / "index" / ".upload-index-transaction.json"
        api.UPLOAD_DIR = self.root / "uploads"
        api.MAX_UPLOAD_FILE_BYTES = 10 * 1024 * 1024
        api.MAX_UPLOAD_REQUEST_BYTES = 100 * 1024 * 1024
        api.MAX_UPLOAD_FILES_PER_REQUEST = 20
        api._index_data = None
        api._indexed_digest_paths = None
        api._indexed_content_digests = set()
        api._model = FakeClipModel()

        old_image = self.root / "dataset" / "legacy" / "old.png"
        old_image.parent.mkdir(parents=True)
        old_image.write_bytes(image_bytes((10, 10, 220)))
        old_paths = [str(old_image)]
        old_embeddings = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
        old_index = faiss.IndexFlatIP(3)
        old_index.add(old_embeddings)
        api.INDEX_PATH.parent.mkdir(parents=True)
        faiss.write_index(old_index, str(api.INDEX_PATH))
        api.PATHS_PATH.write_text("\n".join(old_paths) + "\n", encoding="utf-8")
        np.save(api.EMBEDDINGS_PATH, old_embeddings, allow_pickle=False)
        self.client = TestClient(api.app)

    def tearDown(self):
        self.client.close()
        for name, value in self.old_values.items():
            setattr(api, name, value)
        self.temporary_directory.cleanup()

    def post_upload(self, entries):
        files = [
            ("files", (filename, content, "application/octet-stream"))
            for filename, content, relative_path in entries
        ]
        data = {
            "relative_paths": [relative_path for filename, content, relative_path in entries]
        }
        return self.client.post("/api/upload", data=data, files=files)

    def test_single_upload_searches_by_text_and_image_and_survives_reload(self):
        response = self.post_upload([
            ("red.png", image_bytes((240, 20, 20)), "red.png"),
        ])

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["uploaded_count"], 1)
        self.assertEqual(response.json()["indexed_count"], 1)
        self.assertEqual(response.json()["successful_count"], 1)
        self.assertEqual(response.json()["duplicate_count"], 0)
        self.assertEqual(response.json()["failed_count"], 0)

        text_response = self.client.post(
            "/api/search/text",
            json={"query": "uploaded red item", "top_k": 5},
        )
        self.assertEqual(text_response.status_code, 200, text_response.text)
        uploaded_result = next(
            result for result in text_response.json()["results"]
            if result["image_url"].startswith("/image/uploads/")
        )
        self.assertEqual(self.client.get(uploaded_result["image_url"]).status_code, 200)

        image_response = self.client.post(
            "/api/search/image?top_k=5",
            files={"file": ("query.png", image_bytes((240, 20, 20)), "image/png")},
        )
        self.assertEqual(image_response.status_code, 200, image_response.text)
        self.assertTrue(image_response.json()["results"][0]["image_url"].startswith("/image/uploads/"))

        api._index_data = None
        api._indexed_digest_paths = None
        api._indexed_content_digests = set()
        reloaded_index, reloaded_paths = api.get_index()
        self.assertEqual(reloaded_index.ntotal, 2)
        self.assertIn(str(self.root / "dataset" / "legacy" / "old.png"), reloaded_paths)
        self.assertEqual(len(np.load(api.EMBEDDINGS_PATH, allow_pickle=False)), 2)
        duplicate_after_reload = self.post_upload([
            ("red-again.png", image_bytes((240, 20, 20)), "red-again.png"),
        ])
        self.assertEqual(duplicate_after_reload.status_code, 200, duplicate_after_reload.text)
        self.assertEqual(duplicate_after_reload.json()["duplicate_count"], 1)
        self.assertEqual(api.get_index()[0].ntotal, 2)

        legacy_response = self.client.post(
            "/api/search/text",
            json={"query": "legacy image", "top_k": 5},
        )
        self.assertEqual(legacy_response.status_code, 200, legacy_response.text)
        self.assertTrue(any("old.png" in result["name"] for result in legacy_response.json()["results"]))

    def test_multiple_folder_upload_keeps_nested_relative_paths(self):
        response = self.post_upload([
            ("one.jpg", image_bytes((220, 30, 30), "JPEG"), "trip/day-one/one.jpg"),
            ("two.webp", image_bytes((30, 220, 30), "WEBP"), "trip/day-two/two.webp"),
        ])

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["indexed_count"], 2)
        paths = api.get_index()[1]
        self.assertTrue(any(str(Path("uploads") / "trip" / "day-one") in path for path in paths))
        self.assertTrue(any(str(Path("uploads") / "trip" / "day-two") in path for path in paths))

    def test_invalid_corrupt_and_unsupported_files_are_reported(self):
        response = self.post_upload([
            ("broken.png", b"not an image", "broken.png"),
            ("bitmap.bmp", image_bytes((20, 40, 80), "BMP"), "bitmap.bmp"),
            ("good.png", image_bytes((40, 80, 220)), "good.png"),
        ])

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["indexed_count"], 1)
        self.assertEqual(response.json()["duplicate_count"], 0)
        self.assertEqual(response.json()["failed_count"], 2)
        self.assertEqual(response.json()["skipped_count"], 0)
        self.assertEqual(len(response.json()["errors"]), 2)
        self.assertEqual(api.get_index()[0].ntotal, 2)
        self.assertFalse(any(api.UPLOAD_DIR.rglob("broken*.png")))
        self.assertTrue(any(api.UPLOAD_DIR.rglob("good__sha256_*.png")))

    def test_name_collisions_do_not_overwrite_and_repeated_content_is_skipped(self):
        response = self.post_upload([
            ("same.png", image_bytes((230, 20, 20)), "same.png"),
            ("same.png", image_bytes((20, 230, 20)), "same.png"),
            ("same.png", image_bytes((230, 20, 20)), "same.png"),
        ])

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["uploaded_count"], 2)
        self.assertEqual(response.json()["skipped_count"], 1)
        self.assertEqual(response.json()["duplicate_count"], 1)
        self.assertEqual(response.json()["failed_count"], 0)
        uploaded_paths = [
            Path(path) for path in api.get_index()[1]
            if Path(path).is_relative_to(api.UPLOAD_DIR)
        ]
        self.assertEqual(len(uploaded_paths), 2)
        self.assertEqual(len({path.name for path in uploaded_paths}), 2)
        self.assertTrue(all(path.is_file() for path in uploaded_paths))

        repeated = self.post_upload([
            ("again.png", image_bytes((230, 20, 20)), "again.png"),
        ])
        self.assertEqual(repeated.status_code, 200, repeated.text)
        self.assertEqual(repeated.json()["indexed_count"], 0)
        self.assertEqual(repeated.json()["duplicate_count"], 1)
        self.assertEqual(api.get_index()[0].ntotal, 3)

    def test_duplicates_match_legacy_content_but_same_name_different_content_is_indexed(self):
        legacy_path = self.root / "dataset" / "legacy" / "old.png"
        response = self.post_upload([
            ("old.png", legacy_path.read_bytes(), "old.png"),
            ("old.png", image_bytes((20, 220, 20)), "old.png"),
        ])

        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["successful_count"], 1)
        self.assertEqual(result["duplicate_count"], 1)
        self.assertEqual(result["failed_count"], 0)
        self.assertEqual(
            [item["status"] for item in result["file_results"]],
            ["duplicate", "indexed"],
        )
        self.assertEqual(api.get_index()[0].ntotal, 2)
        self.assertTrue(any("old__sha256_" in Path(path).name for path in api.get_index()[1]))

    def test_large_folder_batches_append_and_are_searchable_immediately(self):
        api.MAX_UPLOAD_FILES_PER_REQUEST = 20
        unique_entries = []
        for position in range(750):
            color = (
                position % 255 + 1,
                (position // 255) % 255 + 1,
                position // (255 * 255) + 1,
            )
            unique_entries.append((
                f"image-{position}.png",
                image_bytes(color),
                f"folder/day-{position // 100}/image-{position}.png",
            ))
        entries = [
            *unique_entries,
            *[
                (f"duplicate-{position}.png", content, f"folder/repeats/{position}.png")
                for position, (_, content, _) in enumerate(unique_entries[:50])
            ],
        ]

        uploaded_totals = {"successful_count": 0, "duplicate_count": 0, "failed_count": 0}
        for start in range(0, len(entries), api.MAX_UPLOAD_FILES_PER_REQUEST):
            response = self.post_upload(entries[start:start + api.MAX_UPLOAD_FILES_PER_REQUEST])
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["failed_count"], 0)
            for count in uploaded_totals:
                uploaded_totals[count] += response.json()[count]

        self.assertEqual(uploaded_totals, {
            "successful_count": 750,
            "duplicate_count": 50,
            "failed_count": 0,
        })
        self.assertEqual(api.get_index()[0].ntotal, 751)
        self.assertEqual(len(api.get_index()[1]), 751)
        search_response = self.client.post(
            "/api/search/text",
            json={"query": "uploaded red image", "top_k": 10},
        )
        self.assertEqual(search_response.status_code, 200, search_response.text)
        self.assertTrue(any(
            result["image_url"].startswith("/image/uploads/")
            for result in search_response.json()["results"]
        ))

    def test_upload_limits_endpoint_and_batch_file_limit(self):
        api.MAX_UPLOAD_FILES_PER_REQUEST = 1
        limits = self.client.get("/api/upload/limits")
        self.assertEqual(limits.status_code, 200)
        self.assertEqual(limits.json(), {
            "max_file_bytes": api.MAX_UPLOAD_FILE_BYTES,
            "max_request_bytes": api.MAX_UPLOAD_REQUEST_BYTES,
            "max_files": 1,
        })

        response = self.post_upload([
            ("one.png", image_bytes((220, 20, 20)), "one.png"),
            ("two.png", image_bytes((20, 220, 20)), "two.png"),
        ])
        self.assertEqual(response.status_code, 413)
        self.assertEqual(api.get_index()[0].ntotal, 1)

    def test_path_traversal_is_rejected(self):
        response = self.post_upload([
            ("escape.png", image_bytes((230, 20, 20)), "../escape.png"),
        ])

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["indexed_count"], 0)
        self.assertIn("Invalid relative image path", response.json()["errors"][0]["error"])
        self.assertFalse((self.root.parent / "escape.png").exists())

    def test_per_file_and_total_upload_limits_are_enforced(self):
        content = image_bytes((200, 30, 30))
        api.MAX_UPLOAD_FILE_BYTES = 50
        per_file_response = self.post_upload([("large.png", content, "large.png")])
        self.assertEqual(per_file_response.status_code, 200, per_file_response.text)
        self.assertIn("per-file limit", per_file_response.json()["errors"][0]["error"])
        self.assertEqual(per_file_response.json()["failed_count"], 1)
        self.assertEqual(per_file_response.json()["skipped_count"], 0)

        api.MAX_UPLOAD_FILE_BYTES = len(content)
        api.MAX_UPLOAD_REQUEST_BYTES = 1024 * 1024
        mixed_response = self.post_upload([
            ("too-large.png", image_bytes((10, 200, 30), size=(512, 512)), "too-large.png"),
            ("valid.png", content, "valid.png"),
        ])
        self.assertEqual(mixed_response.status_code, 200, mixed_response.text)
        self.assertEqual(mixed_response.json()["successful_count"], 1)
        self.assertEqual(mixed_response.json()["failed_count"], 1)

        api.MAX_UPLOAD_FILE_BYTES = 1024 * 1024
        api.MAX_UPLOAD_REQUEST_BYTES = len(content) + 1
        request_response = self.post_upload([
            ("first.png", content, "first.png"),
            ("second.png", content + b"x", "second.png"),
        ])
        self.assertEqual(request_response.status_code, 413, request_response.text)
        self.assertIn("total upload limit", request_response.json()["detail"])
        self.assertEqual(api.get_index()[0].ntotal, 2)

    def test_embedding_failure_and_failed_persistence_preserve_existing_collection(self):
        api._model.encode_image = lambda image: (_ for _ in ()).throw(RuntimeError("model failed"))
        failed_embedding = self.post_upload([
            ("error.png", image_bytes((220, 20, 20)), "error.png"),
        ])
        self.assertEqual(failed_embedding.status_code, 200, failed_embedding.text)
        self.assertEqual(failed_embedding.json()["indexed_count"], 0)
        self.assertEqual(api.get_index()[0].ntotal, 1)

        api._model = FakeClipModel()
        retry = self.post_upload([
            ("error.png", image_bytes((220, 20, 20)), "error.png"),
        ])
        self.assertEqual(retry.status_code, 200, retry.text)
        self.assertEqual(retry.json()["successful_count"], 1)
        self.assertEqual(api.get_index()[0].ntotal, 2)

        api._model = FakeClipModel()
        with patch.object(api, "_write_index_artifacts", side_effect=OSError("disk full")):
            failed_update = self.post_upload([
                ("disk.png", image_bytes((20, 220, 20)), "disk.png"),
            ])
        self.assertEqual(failed_update.status_code, 500)
        self.assertEqual(api.get_index()[0].ntotal, 2)
        self.assertEqual(len(api.PATHS_PATH.read_text(encoding="utf-8").splitlines()), 2)
        self.assertFalse(any(api.UPLOAD_DIR.rglob("disk__sha256_*.png")))

    def test_artifact_replacement_rolls_back_after_partial_commit_failure(self):
        original_artifacts = [
            api.INDEX_PATH.read_bytes(),
            api.PATHS_PATH.read_bytes(),
            api.EMBEDDINGS_PATH.read_bytes(),
        ]
        next_index = faiss.clone_index(api.get_index()[0])
        next_index.add(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
        next_embeddings = np.vstack((
            np.load(api.EMBEDDINGS_PATH, allow_pickle=False),
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        ))
        next_paths = [*api.get_index()[1], str(self.root / "uploads" / "new.png")]
        replace = api.os.replace
        replacements = 0

        def fail_during_replacement(source, destination):
            nonlocal replacements
            replacements += 1
            if replacements == 4:
                raise OSError("simulated replacement failure")
            return replace(source, destination)

        with patch.object(api.os, "replace", side_effect=fail_during_replacement):
            with self.assertRaises(OSError):
                api._write_index_artifacts(next_index, next_embeddings, next_paths)

        self.assertEqual(api.INDEX_PATH.read_bytes(), original_artifacts[0])
        self.assertEqual(api.PATHS_PATH.read_bytes(), original_artifacts[1])
        self.assertEqual(api.EMBEDDINGS_PATH.read_bytes(), original_artifacts[2])
        self.assertEqual(faiss.read_index(str(api.INDEX_PATH)).ntotal, 1)

    def test_index_load_recovers_an_interrupted_transaction(self):
        destinations = [api.INDEX_PATH, api.PATHS_PATH, api.EMBEDDINGS_PATH]
        backups = []
        for destination in destinations:
            backup = destination.with_name(f".{destination.name}.interrupted.bak")
            destination.replace(backup)
            destination.write_bytes(b"partial update")
            backups.append(backup)
        uploaded_image = api.UPLOAD_DIR / "orphan.png"
        uploaded_image.parent.mkdir(parents=True, exist_ok=True)
        uploaded_image.write_bytes(b"uncommitted upload")
        api.INDEX_TRANSACTION_PATH.write_text(
            json.dumps({
                "artifacts": [
                    {"name": destination.name, "backup": backup.name}
                    for destination, backup in zip(destinations, backups)
                ],
                "uploads": [uploaded_image.relative_to(api.ROOT).as_posix()],
            }),
            encoding="utf-8",
        )

        api._index_data = None
        recovered_index, recovered_paths = api.get_index()
        self.assertEqual(recovered_index.ntotal, 1)
        self.assertEqual(len(recovered_paths), 1)
        self.assertFalse(uploaded_image.exists())
        self.assertFalse(api.INDEX_TRANSACTION_PATH.exists())

    def test_concurrent_uploads_leave_index_and_metadata_aligned(self):
        entries = [
            ("first.png", image_bytes((230, 30, 30)), "first.png"),
            ("second.png", image_bytes((30, 230, 30)), "second.png"),
        ]

        def send(entry):
            with TestClient(api.app) as client:
                filename, content, relative_path = entry
                return client.post(
                    "/api/upload",
                    data={"relative_paths": relative_path},
                    files=[("files", (filename, content, "image/png"))],
                )

        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(executor.map(send, entries))

        self.assertTrue(all(response.status_code == 200 for response in responses))
        api._index_data = None
        index, paths = api.get_index()
        embeddings = np.load(api.EMBEDDINGS_PATH, allow_pickle=False)
        self.assertEqual(index.ntotal, 3)
        self.assertEqual(len(paths), 3)
        self.assertEqual(embeddings.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
