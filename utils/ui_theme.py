"""Shared dark dashboard styling and presentation helpers."""

import html
import json
from pathlib import Path
import re
from urllib.parse import quote

import streamlit as st
import streamlit.components.v1 as components

from visionsearch_web_ui.sharing import (
    LOCAL_API_BASE_URL,
    PUBLIC_BASE_URL,
    PUBLIC_SHARING_ENABLED,
    image_id_for_path,
)


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
                _render_share_options(image_path, project_root)


def _render_share_options(image_path: Path, project_root: Path) -> None:
    if not image_path.is_file():
        return

    image_id = image_id_for_path(image_path)
    if image_id is None:
        st.caption("Sharing is available for JPEG, PNG, and WebP images.")
        return

    public_url = None
    if PUBLIC_SHARING_ENABLED and PUBLIC_BASE_URL:
        public_url = f"{PUBLIC_BASE_URL}/api/share/images/{image_id}"

    try:
        relative_path = image_path.resolve().relative_to(project_root.resolve())
    except (OSError, ValueError):
        relative_path = None
    local_image_url = None
    if relative_path is not None:
        encoded_path = quote(relative_path.as_posix(), safe="/")
        local_image_url = f"{LOCAL_API_BASE_URL}/image/{encoded_path}"
    copy_url = public_url or local_image_url

    message = "Found this image using VisionSearch AI!"
    sharing_text = message
    if copy_url:
        sharing_text = f"{message} {copy_url}"
    whatsapp_url = f"https://wa.me/?text={quote(sharing_text, safe='')}"
    telegram_url = (
        f"https://t.me/share/url?url={quote(copy_url or '', safe='')}"
        f"&text={quote(message, safe='')}"
    )
    email_url = (
        f"mailto:?subject={quote('Image from VisionSearch AI', safe='')}"
        f"&body={quote(sharing_text, safe='')}"
    )

    with st.expander("↗ Share or download", expanded=False):
        if public_url:
            st.markdown(
                f'<a href="{html.escape(whatsapp_url, quote=True)}" target="_blank" rel="noopener noreferrer">WhatsApp</a>'
                f' · <a href="{html.escape(telegram_url, quote=True)}" target="_blank" rel="noopener noreferrer">Telegram</a>'
                f' · <a href="{html.escape(email_url, quote=True)}">Email</a>',
                unsafe_allow_html=True,
            )
        else:
            social_columns = st.columns(3)
            for column, label in zip(social_columns, ("WhatsApp", "Telegram", "Email")):
                column.button(
                    label,
                    disabled=True,
                    key=f"{label.lower()}-share-{image_id}",
                    help="Enable approved public image links to share with external apps.",
                )
            st.caption(
                "Public links are disabled. Configure and explicitly approve a public "
                "indexed collection before enabling external share links."
            )

        if copy_url:
            encoded_url = json.dumps(copy_url).replace("<", "\\u003c")
            components.html(
                f"""
                <button id="copy-image-link" type="button">Copy image link</button>
                <span id="copy-feedback" role="status" aria-live="polite"></span>
                <script>
                  const link = {encoded_url};
                  const button = document.getElementById("copy-image-link");
                  const feedback = document.getElementById("copy-feedback");
                  button.addEventListener("click", async () => {{
                    try {{
                      await navigator.clipboard.writeText(link);
                      feedback.textContent = "Copied.";
                    }} catch {{
                      feedback.textContent = "Clipboard access is unavailable.";
                    }}
                  }});
                </script>
                """,
                height=44,
            )
            if not public_url:
                st.caption("This is a local API link; it will not work on another device.")

        image_url = local_image_url or public_url
        if image_url:
            safe_filename = re.sub(r"[^A-Za-z0-9._-]+", "_", image_path.name)
            encoded_image_url = json.dumps(image_url).replace("<", "\\u003c")
            encoded_filename = json.dumps(safe_filename).replace("<", "\\u003c")
            components.html(
                f"""
                <button id="native-share-image" type="button">Share image…</button>
                <span id="native-share-feedback" role="status" aria-live="polite"></span>
                <script>
                  const imageUrl = {encoded_image_url};
                  const filename = {encoded_filename};
                  const feedback = document.getElementById("native-share-feedback");
                  document.getElementById("native-share-image").addEventListener("click", async () => {{
                    if (!navigator.share) {{
                      feedback.textContent = "Device sharing is unsupported in this browser.";
                      return;
                    }}
                    try {{
                      const response = await fetch(imageUrl);
                      if (!response.ok) throw new Error("Image could not be loaded.");
                      const blob = await response.blob();
                      const file = new File([blob], filename, {{type: blob.type}});
                      if (navigator.canShare && navigator.canShare({{files: [file]}})) {{
                        await navigator.share({{files: [file], title: filename, text: {json.dumps(message)}}});
                      }} else {{
                        await navigator.share({{
                          title: filename,
                          text: {json.dumps(message)},
                          ...( {json.dumps(public_url)} ? {{url: {json.dumps(public_url)}}} : {{}})
                        }});
                      }}
                      feedback.textContent = "Share sheet opened.";
                    }} catch (error) {{
                      if (error.name !== "AbortError") feedback.textContent = error.message;
                    }}
                  }});
                </script>
                """,
                height=44,
            )

        try:
            with image_path.open("rb") as image_file:
                st.download_button(
                    "Download image",
                    data=image_file,
                    file_name=re.sub(r"[^A-Za-z0-9._-]+", "_", image_path.name),
                    mime="application/octet-stream",
                    key=f"download-{image_id}",
                )
        except OSError as error:
            st.error(f"Image download is unavailable: {error}")
