import streamlit as st
from pathlib import Path

from services.history_service import save_search
from services.hybrid_service import hybrid_search
from services.attribute_service import extract_attributes
from utils.ui_theme import apply_theme, render_brand, render_result_cards

st.set_page_config(page_title="Smart Search · VisionSearch AI", page_icon="✨", layout="wide")
apply_theme()
render_brand()

st.title("Smart search")
st.markdown("Combine semantic retrieval with query details such as colors and scenes.")

with st.form("smart_search_form"):
    query = st.text_input(
        "What are you looking for?",
        placeholder="e.g. a red bicycle parked on a city street",
    )
    limit = st.select_slider("Number of results", options=[5, 10, 20], value=10)
    submitted = st.form_submit_button("Search intelligently", type="primary")

if submitted:
    if not query.strip():
        st.warning("Enter a description to search.")
    else:
        try:
            with st.spinner("Analyzing your query and searching..."):
                attributes = extract_attributes(query.strip())
                results = hybrid_search(query.strip(), limit=limit)
            save_search(query.strip(), "smart", results)
            st.subheader("Detected query details")
            if attributes["colors"]:
                st.caption("Colors: " + ", ".join(attributes["colors"]))
            if attributes["scenes"]:
                st.caption("Scenes: " + ", ".join(attributes["scenes"]))
            if attributes["objects"]:
                st.caption("Keywords: " + ", ".join(attributes["objects"]))
            st.subheader("Search results")
            render_result_cards(results, Path(__file__).resolve().parents[1])
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            st.error(f"Search could not be completed: {error}")
