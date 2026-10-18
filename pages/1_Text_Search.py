import streamlit as st
from pathlib import Path

from services.history_service import save_search
from services.search_service import search_by_text
from utils.ui_theme import apply_theme, render_brand, render_result_cards

st.set_page_config(page_title="Text Search · VisionSearch AI", page_icon="🔍", layout="wide")
apply_theme()
render_brand()

st.title("Search with text")
st.markdown("Describe the image you have in mind and find semantically similar results.")

with st.form("text_search_form"):
    query = st.text_input(
        "Describe an image",
        placeholder="e.g. a golden retriever running through a field",
    )
    limit = st.select_slider("Number of results", options=[5, 10, 20], value=10)
    submitted = st.form_submit_button("Search images", type="primary")

if submitted:
    if not query.strip():
        st.warning("Enter a description to search.")
    else:
        try:
            with st.spinner("Searching your image collection..."):
                results = search_by_text(query.strip(), limit=limit)
            save_search(query.strip(), "text", results)
            st.subheader(f"Results for “{query.strip()}”")
            render_result_cards(results, Path(__file__).resolve().parents[1])
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            st.error(f"Search could not be completed: {error}")
