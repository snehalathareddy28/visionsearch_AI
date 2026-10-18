import streamlit as st
from pathlib import Path
from PIL import Image

from services.history_service import save_search
from services.search_service import search_by_image
from utils.ui_theme import apply_theme, render_brand, render_result_cards

st.set_page_config(page_title="Image Search · VisionSearch AI", page_icon="🖼️", layout="wide")
apply_theme()
render_brand()

st.title("Search with an image")
st.markdown("Upload a reference image to discover visually similar images in your collection.")

image = st.file_uploader(
    "Choose a reference image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)
limit = st.select_slider("Number of results", options=[5, 10, 20], value=10)

if image is not None:
    try:
        preview = Image.open(image).convert("RGB")
        st.image(preview, caption="Reference image", width=280)
    except (OSError, ValueError) as error:
        st.error(f"Unable to open the uploaded image: {error}")
        image = None

if st.button("Find similar images", type="primary", disabled=image is None):
    try:
        image.seek(0)
        with st.spinner("Finding visually similar images..."):
            results = search_by_image(Image.open(image).convert("RGB"), limit=limit)
        save_search("Uploaded image", "image", results)
        st.subheader("Similar images")
        render_result_cards(results, Path(__file__).resolve().parents[1])
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        st.error(f"Search could not be completed: {error}")
