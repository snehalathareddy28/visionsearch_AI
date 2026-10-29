"""Generate CLIP image embeddings and build a FAISS index."""

import shutil
import tarfile
from pathlib import Path
from pathlib import PurePosixPath

import faiss
import numpy as np
from PIL import Image, UnidentifiedImageError

from models.clip_model import ClipModel, load_clip_model
from utils.image_utils import SUPPORTED_EXTENSIONS, list_images

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIRECTORY = PROJECT_ROOT / "dataset" / "caltech-101-selected"
DEFAULT_INDEX_DIRECTORY = PROJECT_ROOT / "index"
DEFAULT_EMBEDDINGS_PATH = DEFAULT_INDEX_DIRECTORY / "image_embeddings.npy"
DEFAULT_PATHS_PATH = DEFAULT_INDEX_DIRECTORY / "image_paths.txt"
DEFAULT_FAISS_PATH = DEFAULT_INDEX_DIRECTORY / "faiss.index"


def count_image_archive_members(archive_path) -> int:
    """Count supported regular image files inside a tar archive."""
    with tarfile.open(archive_path, mode="r:*") as archive:
        return sum(
            1
            for member in archive
            if member.isfile()
            and PurePosixPath(member.name.replace("\\", "/")).suffix.lower()
            in SUPPORTED_EXTENSIONS
        )


def extract_image_archive(archive_path, destination) -> int:
    """Extract supported image files from a tar archive into a dataset folder."""
    archive_path = Path(archive_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    extracted_count = 0

    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive:
            member_path = PurePosixPath(member.name.replace("\\", "/"))
            if (
                not member.isfile()
                or member_path.is_absolute()
                or ".." in member_path.parts
                or (member_path.parts and ":" in member_path.parts[0])
                or member_path.suffix.lower() not in SUPPORTED_EXTENSIONS
            ):
                continue

            image_path = (destination / Path(*member_path.parts)).resolve()
            try:
                image_path.relative_to(destination_root)
            except ValueError as error:
                raise ValueError(
                    f"Archive image path escapes the dataset folder: {member.name}"
                ) from error

            source = archive.extractfile(member)
            if source is None:
                continue

            image_path.parent.mkdir(parents=True, exist_ok=True)
            with source, image_path.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted_count += 1

    if extracted_count == 0:
        raise ValueError(f"No supported images found in archive {archive_path}")

    return extracted_count


def build_index(
    image_directory=DEFAULT_IMAGE_DIRECTORY,
    embeddings_path=DEFAULT_EMBEDDINGS_PATH,
    paths_path=DEFAULT_PATHS_PATH,
    faiss_path=DEFAULT_FAISS_PATH,
    clip_model: ClipModel | None = None,
) -> int:
    """Encode valid images, persist aligned artifacts, and create an IP index."""
    image_paths = list_images(image_directory)
    if not image_paths:
        raise ValueError(f"No supported images found in {image_directory}")

    model = clip_model or load_clip_model()
    embeddings = []
    valid_paths = []
    for position, image_path in enumerate(image_paths, start=1):
        print(f"Processing {position}/{len(image_paths)}: {image_path.name}")
        try:
            with Image.open(image_path) as image:
                embeddings.append(model.encode_image(image.convert("RGB"))[0])
            valid_paths.append(str(image_path))
        except (OSError, UnidentifiedImageError, ValueError, TypeError) as error:
            print(f"Skipping invalid image {image_path}: {error}")

    if not embeddings:
        raise ValueError("No valid images could be processed")

    embedding_array = np.asarray(embeddings, dtype=np.float32)
    if embedding_array.ndim != 2 or embedding_array.shape[1] != 512:
        raise ValueError(f"Expected CLIP embeddings with shape (n, 512), got {embedding_array.shape}")

    output_paths = [Path(embeddings_path), Path(paths_path), Path(faiss_path)]
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embedding_array)
    Path(paths_path).write_text("\n".join(valid_paths) + "\n", encoding="utf-8")

    index = faiss.IndexFlatIP(512)
    index.add(embedding_array)
    faiss.write_index(index, str(faiss_path))

    print(f"Images processed: {len(valid_paths)}")
    print(f"Embedding dimension: {embedding_array.shape[1]}")
    print("FAISS index created successfully.")
    print(f"Number of vectors: {index.ntotal}")
    print(f"Dimension: {index.d}")
    return len(valid_paths)
