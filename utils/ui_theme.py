"""Shared dark dashboard styling and presentation helpers."""

from pathlib import Path

import streamlit as st


def apply_theme() -> None:
    """Apply the shared VisionSearch AI dark navy and violet theme."""
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            --vs-bg: #0b1020;
            --vs-panel: #121a2d;
            --vs-panel-hover: #1a2540;
            --vs-border: #27334d;
            --vs-violet: #8173ff;
            --vs-violet-light: #a99fff;
            --vs-text: #f3f5ff;
            --vs-muted: #9aa8c7;
        }

        .stApp {
            background:
                radial-gradient(ellipse at 85% 0%, rgba(91, 74, 191, .14), transparent 34%),
                var(--vs-bg);
            color: var(--vs-text);
        }

        [data-testid="stHeader"] {
            background: rgba(11, 16, 32, .88);
        }

        [data-testid="stSidebar"] {
            background: #0e1527;
            border-right: 1px solid var(--vs-border);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label {
            color: var(--vs-text);
        }

        .block-container {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: var(--vs-text) !important;
            letter-spacing: -.025em;
        }

        p, label, [data-testid="stMarkdownContainer"] {
            color: var(--vs-text);
        }

        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p {
            color: var(--vs-muted) !important;
        }

        [data-testid="stMetric"] {
            background: var(--vs-panel);
            border: 1px solid var(--vs-border);
            border-radius: 16px;
            padding: 16px 18px;
        }

        [data-testid="stMetricLabel"] {
            color: var(--vs-muted) !important;
        }

        [data-testid="stMetricValue"] {
            color: var(--vs-text) !important;
        }

        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"] > div {
            background: #111a2d !important;
            color: var(--vs-text) !important;
            border-color: var(--vs-border) !important;
            border-radius: 12px !important;
        }

        .stTextInput input:focus,
        .stTextArea textarea:focus {
            border-color: var(--vs-violet) !important;
            box-shadow: 0 0 0 1px var(--vs-violet) !important;
        }

        .stButton > button,
        .stFormSubmitButton > button {
            background: var(--vs-violet);
            color: #fff;
            border: 1px solid var(--vs-violet);
            border-radius: 11px;
            padding: .55rem 1.1rem;
            font-weight: 600;
            transition: all .2s ease;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            background: #6b5ce7;
            color: #fff;
            border-color: var(--vs-violet-light);
            transform: translateY(-1px);
        }

        [data-testid="stFileUploader"] {
            background: var(--vs-panel);
            border: 1px dashed #465578;
            border-radius: 14px;
            padding: 12px;
        }

        [data-testid="stImage"] img {
            border-radius: 12px;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--vs-panel);
            border-color: var(--vs-border);
            border-radius: 16px;
        }

        hr {
            border-color: var(--vs-border);
        }

        a {
            color: var(--vs-violet-light) !important;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        @media (max-width: 768px) {
            .block-container {
                padding: 1rem 1rem 2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand() -> None:
    """Render the shared VisionSearch AI brand header."""
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:14px;padding:4px 0 18px">
          <div style="width:46px;height:46px;border-radius:14px;display:flex;
                      align-items:center;justify-content:center;
                      background:linear-gradient(135deg,#8173ff,#5545d6);
                      color:white;font-size:24px;font-weight:800;
                      box-shadow:0 8px 28px rgba(129,115,255,.22)">V</div>
          <div>
            <div style="color:#f3f5ff;font-size:24px;font-weight:750;
                        letter-spacing:-.7px;line-height:1.2">
              VisionSearch <span style="color:#a99fff">AI</span>
            </div>
            <div style="color:#9aa8c7;font-size:13px;margin-top:4px">
              Intelligent multimodal image retrieval
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_cards(results: list[dict], project_root: Path) -> None:
    """Render ranked image results in responsive cards."""
    if not results:
        st.info("No matching images were found above the similarity threshold.")
        return

    st.caption(f"{len(results)} matching image(s)")
    columns = st.columns(3)
    for position, result in enumerate(results):
        image_path = Path(result["image_path"])
        if not image_path.is_absolute():
            image_path = project_root / image_path

        with columns[position % len(columns)]:
            with st.container(border=True):
                st.markdown(f"#### #{result['rank']}  ·  Match")
                if image_path.is_file():
                    st.image(str(image_path), use_container_width=True)
                else:
                    st.warning(f"Image file missing: {image_path}")
                st.metric("Similarity", f"{result['score']:.4f}")
                st.caption(f"FAISS ID: {result['faiss_id']}")
                category = image_path.parent.name
                if result.get("category_match"):
                    st.caption(f"{category} · matched query category")
                else:
                    st.caption(category)
                st.caption(image_path.name)
