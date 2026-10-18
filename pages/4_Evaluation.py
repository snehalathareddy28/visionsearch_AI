import streamlit as st
from utils.ui_theme import apply_theme, render_brand

st.set_page_config(page_title="Evaluation · VisionSearch AI", page_icon="📊", layout="wide")
apply_theme()
render_brand()

st.title("Search evaluation")
st.markdown("Evaluate retrieval quality against a labeled query set.")

left, right = st.columns(2)
with left:
    st.metric("Evaluation dataset", "Not configured")
with right:
    st.metric("Quality metrics", "Awaiting dataset")

st.info(
    "Add labeled search queries and their relevant image paths to enable "
    "precision, recall, and ranking-quality measurements. No evaluation "
    "scores are shown until a dataset is configured."
)
