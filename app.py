import streamlit as st
import tarfile
from pathlib import Path
from pathlib import PurePosixPath
from PIL import Image

from models.clip_model import load_clip_model
from services.search_service import load_index, search_loaded, search_text_loaded
from services.indexing_service import (
    build_index,
    extract_image_archive,
)
from utils.image_utils import SUPPORTED_EXTENSIONS, list_images
from utils.ui_theme import apply_theme, render_brand, render_result_cards


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VisionSearch AI",
    page_icon="🔍",
    layout="wide"
)
apply_theme()


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_PATH = PROJECT_ROOT / "index" / "faiss.index"
PATHS_PATH = PROJECT_ROOT / "index" / "image_paths.txt"
EMBEDDINGS_PATH = PROJECT_ROOT / "index" / "image_embeddings.npy"

IMAGE_DIRECTORY = PROJECT_ROOT / "dataset" / "images"
CALTECH_SELECTED_IMAGE_DIRECTORY = PROJECT_ROOT / "dataset" / "caltech-101-selected"
CALTECH_ARCHIVE_PATH = PROJECT_ROOT / "caltech-101" / "101_ObjectCategories.tar.gz"
CALTECH_IMAGE_DIRECTORY = PROJECT_ROOT / "dataset" / "caltech-101"


# ============================================================
# LOAD CLIP MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading CLIP model...")
def get_cached_clip_model():
    """
    Load and cache the CLIP model.

    The model is loaded only once during the Streamlit session
    instead of loading it for every search.
    """
    return load_clip_model()


# ============================================================
# LOAD FAISS INDEX
# ============================================================

@st.cache_resource(show_spinner="Loading image index...")
def get_cached_index():
    """
    Load and cache the FAISS index and image paths.
    """
    return load_index(
        INDEX_PATH,
        PATHS_PATH
    )


@st.cache_data(show_spinner=False)
def get_caltech_image_count(archive_path: str, modified_time: float) -> int:
    with tarfile.open(archive_path, mode="r:*") as archive:
        return sum(
            1
            for member in archive
            if member.isfile()
            and PurePosixPath(member.name.replace("\\", "/")).suffix.lower()
            in SUPPORTED_EXTENSIONS
        )


# ============================================================
# FILE UPLOAD HELPERS
# ============================================================


def reset_uploaded_file(uploaded_file):
    """Reset a Streamlit uploaded file back to the beginning of its bytes."""
    if uploaded_file is not None and hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    return uploaded_file


# ============================================================
# DISPLAY SEARCH RESULTS
# ============================================================

def display_results(results):
    """
    Display ranked image retrieval results.
    """
    render_result_cards(results, PROJECT_ROOT)


# ============================================================
# TITLE
# ============================================================

render_brand()
st.markdown(
    "<h1 style='font-size:2.6rem;margin:0'>Find any image, by meaning.</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "Search your collection with natural language or a reference image. "
    "Powered by CLIP embeddings and FAISS retrieval."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Search settings")

    search_mode = st.radio(
        "Search Mode",
        [
            "Text Search",
            "Image Search"
        ]
    )

    top_k = st.selectbox(
        "Number of Results",
        [5, 10, 20],
        index=0
    )

    st.divider()

    # --------------------------------------------------------
    # INDEX STATUS
    # --------------------------------------------------------

    st.header("Index status")

    if INDEX_PATH.exists() and PATHS_PATH.exists():

        st.success("Index ready")

        try:

            _, indexed_paths = get_cached_index()

            st.info(
                f"Indexed Images: {len(indexed_paths)}"
            )

        except FileNotFoundError:

            st.warning(
                "Index files are missing."
            )

        except Exception as error:

            st.warning(
                f"Index status error: {error}"
            )

    else:

        st.warning(
            "Index Not Built"
        )

        st.info(
            "Add images to dataset/images and "
            "build the image index."
        )


# ============================================================
# TEXT SEARCH
# ============================================================

if search_mode == "Text Search":

    st.header("Text-to-image search")

    query = st.text_input(
        "Describe the image you are looking for",
        placeholder=(
            "Example: A dog running on green grass"
        )
    )

    search_button = st.button(
        "🔍 Search Images",
        type="primary"
    )

    if search_button:

        if not query.strip():

            st.warning(
                "Please enter a search query."
            )

        else:

            try:

                # Load FAISS index
                index, paths = get_cached_index()

                # Load CLIP
                model = get_cached_clip_model()

                # Convert text into CLIP embedding
                query_embedding = model.encode_text(
                    query.strip()
                )

                # Search FAISS
                results = search_text_loaded(
                    query.strip(),
                    index,
                    paths,
                    query_embedding,
                    top_k=top_k
                )

                st.subheader(
                    f"Search Results for: {query}"
                )

                display_results(results)

            except FileNotFoundError as error:

                st.error(
                    f"Index Error: {error}"
                )

            except Exception as error:

                st.error(
                    f"Search Error: {error}"
                )


# ============================================================
# IMAGE SEARCH
# ============================================================

else:

    st.header("Image-to-image search")

    uploaded_image = st.file_uploader(
        "Upload an image to find similar images",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
            "bmp"
        ]
    )

    if uploaded_image is not None:

        try:

            reset_uploaded_file(uploaded_image)
            preview_image = Image.open(
                uploaded_image
            ).convert("RGB")

            st.image(
                preview_image,
                caption="Uploaded Image",
                width=250
            )

        except Exception as error:

            st.error(
                f"Unable to open uploaded image: {error}"
            )

    search_button = st.button(
        "🔍 Find Similar Images",
        type="primary"
    )

    if search_button:

        if uploaded_image is None:

            st.warning(
                "Please upload an image."
            )

        else:

            try:

                reset_uploaded_file(uploaded_image)

                # Open uploaded image
                image = Image.open(
                    uploaded_image
                ).convert("RGB")

                # Load FAISS index
                index, paths = get_cached_index()

                # Load CLIP
                model = get_cached_clip_model()

                # Convert image into CLIP embedding
                query_embedding = model.encode_image(
                    image
                )

                # Search FAISS
                results = search_loaded(
                    index,
                    paths,
                    query_embedding,
                    top_k=top_k
                )

                st.subheader(
                    "Similar Images"
                )

                display_results(results)

            except FileNotFoundError as error:

                st.error(
                    f"Index Error: {error}"
                )

            except Exception as error:

                st.error(
                    f"Search Error: {error}"
                )


# ============================================================
# DATASET MANAGEMENT
# ============================================================

st.divider()

st.header("Dataset management")

my_images_option = "My Images (dataset/images)"
selected_caltech_option = "Caltech-101 Selected (dataset/caltech-101-selected)"
archive_caltech_option = "Caltech-101 (from archive)"
dataset_options = [my_images_option]
selected_caltech_available = CALTECH_SELECTED_IMAGE_DIRECTORY.is_dir()
if selected_caltech_available:
    dataset_options.append(selected_caltech_option)
caltech_archive_available = CALTECH_ARCHIVE_PATH.is_file()
if caltech_archive_available:
    dataset_options.append(archive_caltech_option)

default_dataset_index = 0
if selected_caltech_available:
    default_dataset_index = dataset_options.index(selected_caltech_option)
elif caltech_archive_available:
    default_dataset_index = dataset_options.index(archive_caltech_option)

selected_dataset = st.selectbox(
    "Dataset to index",
    dataset_options,
    index=default_dataset_index,
)

if selected_dataset == selected_caltech_option:
    selected_images = list_images(CALTECH_SELECTED_IMAGE_DIRECTORY)
    st.success(
        f"Caltech-101 Selected — {len(selected_images):,} supported image(s) "
        f"detected in `{CALTECH_SELECTED_IMAGE_DIRECTORY}`."
    )
    st.warning(
        "Building this index replaces the existing FAISS index, image embeddings, "
        "and image paths. Confirm this folder contains every image you want searchable."
    )
elif selected_dataset == archive_caltech_option:
    if caltech_archive_available:
        extracted_images = list_images(CALTECH_IMAGE_DIRECTORY)
        archive_image_count = get_caltech_image_count(
            str(CALTECH_ARCHIVE_PATH),
            CALTECH_ARCHIVE_PATH.stat().st_mtime
        )
        st.success(
            f"Caltech-101 selected — {archive_image_count:,} supported images "
            f"detected in `{CALTECH_ARCHIVE_PATH.name}`."
        )
        st.info(
            f"Extracted images currently detected: {len(extracted_images)}"
        )
        st.write(
            "Click **Build / Rebuild Image Index** below to extract and index "
            f"the Caltech-101 images into `{CALTECH_IMAGE_DIRECTORY}`."
        )
    else:
        st.error(
            f"Caltech-101 archive not found: `{CALTECH_ARCHIVE_PATH}`"
        )
else:
    local_images = list_images(IMAGE_DIRECTORY)
    st.write(
        f"Supported images detected in `dataset/images`: {len(local_images)}"
    )

if not caltech_archive_available:
    st.caption(
        f"To enable Caltech-101, place `101_ObjectCategories.tar.gz` "
        f"under `{CALTECH_ARCHIVE_PATH.parent}` and refresh the app."
    )


# ============================================================
# BUILD / REBUILD INDEX
# ============================================================

if st.button(
    "🔄 Build / Rebuild Image Index"
):

    try:

        with st.spinner(
            "Building image index..."
        ):

            selected_image_directory = IMAGE_DIRECTORY
            if selected_dataset == selected_caltech_option:
                selected_image_directory = CALTECH_SELECTED_IMAGE_DIRECTORY
            elif selected_dataset == archive_caltech_option:
                if not CALTECH_ARCHIVE_PATH.is_file():
                    raise FileNotFoundError(
                        f"Caltech-101 archive not found: {CALTECH_ARCHIVE_PATH}"
                    )
                extracted_count = extract_image_archive(
                    CALTECH_ARCHIVE_PATH,
                    CALTECH_IMAGE_DIRECTORY
                )
                st.success(
                    f"Extracted {extracted_count} image(s) from Caltech-101."
                )
                selected_image_directory = CALTECH_IMAGE_DIRECTORY

            detected_images = list_images(selected_image_directory)
            st.info(
                f"Detected {len(detected_images)} supported image(s) to index "
                f"from `{selected_image_directory}`."
            )
            count = build_index(
                image_directory=selected_image_directory,
                embeddings_path=EMBEDDINGS_PATH,
                paths_path=PATHS_PATH,
                faiss_path=INDEX_PATH,
                clip_model=get_cached_clip_model()
            )

        # Clear cached index because a new index was created
        get_cached_index.clear()

        st.success(
            f"Successfully indexed {count} image(s)!"
        )

        st.info(
            f"FAISS index saved to: {INDEX_PATH}"
        )

    except ValueError as error:

        st.error(
            f"Index Build Error: {error}"
        )

    except Exception as error:

        st.error(
            f"Unexpected Index Build Error: {error}"
        )


# ============================================================
# PROJECT INFORMATION
# ============================================================

with st.expander("ℹ️ About VisionSearch AI"):

    st.write(
        """
        **VisionSearch AI** is an intelligent multimodal
        semantic image retrieval system.

        **Technologies used:**

        - CLIP for multimodal image and text embeddings
        - FAISS for vector similarity search
        - PyTorch for model execution
        - Hugging Face Transformers for CLIP
        - Pillow for image processing
        - Streamlit for the user interface

        **Supported search modes:**

        1. Text-to-Image Retrieval
        2. Image-to-Image Retrieval

        The system converts images and queries into
        normalized CLIP embeddings and uses FAISS
        inner-product similarity search to retrieve
        the most relevant images.
        """
    ) 