"""Embedding service backed by the cached CLIP model."""

from models.clip_model import load_clip_model


def embed_image(image):
    return load_clip_model().encode_image(image)


def embed_text(text: str):
    return load_clip_model().encode_text(text)
