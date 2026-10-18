"""Manual smoke test for the Hugging Face CLIP model layer."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.clip_model import get_clip_model


def main() -> None:
    clip_model = get_clip_model()

    text_embedding = clip_model.encode_text("a dog playing in a park")
    print(f"Text embedding shape: {text_embedding.shape}")

    test_image = Image.new("RGB", (224, 224), color=(80, 140, 90))
    image_embedding = clip_model.encode_image(test_image)
    print(f"Image embedding shape: {image_embedding.shape}")

    text_norm = np.linalg.norm(text_embedding, axis=1)
    image_norm = np.linalg.norm(image_embedding, axis=1)
    print(f"Text embedding norm: {text_norm[0]:.6f}")
    print(f"Image embedding norm: {image_norm[0]:.6f}")

    assert text_embedding.shape == (1, 512), text_embedding.shape
    assert image_embedding.shape == (1, 512), image_embedding.shape
    assert text_embedding.dtype == np.float32
    assert image_embedding.dtype == np.float32
    assert np.allclose(text_norm, 1.0, atol=1e-5)
    assert np.allclose(image_norm, 1.0, atol=1e-5)
    print("CLIP TEST PASSED")


if __name__ == "__main__":
    main()
