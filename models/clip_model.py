"""Reusable Hugging Face CLIP model wrapper."""

from functools import lru_cache
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"


class ClipModel:
    """Load CLIP once and expose normalized text and image embeddings."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print("Loading CLIP model...")
        try:
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.model.eval()
        except Exception as error:
            raise RuntimeError(f"Unable to load CLIP model '{model_name}': {error}") from error
        print("CLIP model loaded successfully.")

    @staticmethod
    def _embedding_tensor(output: Any) -> torch.Tensor:
        """Handle tensor, tuple, and model-output forms across Transformers versions."""
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            return output[0]
        if hasattr(output, "pooler_output") and isinstance(output.pooler_output, torch.Tensor):
            return output.pooler_output
        if hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, torch.Tensor):
            return output.last_hidden_state[:, 0, :]
        raise RuntimeError("CLIP returned an unsupported embedding output type")

    @staticmethod
    def _normalize(embedding: torch.Tensor) -> np.ndarray:
        normalized = embedding / embedding.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return normalized.detach().cpu().numpy().astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode one non-empty text query as a normalized float32 array."""
        if not isinstance(text, str):
            raise TypeError("Text query must be a string")
        if not text.strip():
            raise ValueError("Text query cannot be empty")
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.inference_mode():
            embedding = self._embedding_tensor(self.model.get_text_features(**inputs))
        return self._normalize(embedding)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode one PIL image as a normalized float32 array."""
        if not isinstance(image, Image.Image):
            raise TypeError("Image must be a PIL.Image.Image instance")
        try:
            image = image.convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        except Exception as error:
            raise ValueError(f"Invalid image: {error}") from error
        with torch.inference_mode():
            embedding = self._embedding_tensor(self.model.get_image_features(**inputs))
        return self._normalize(embedding)


@lru_cache(maxsize=1)
def get_clip_model(model_name: str = MODEL_NAME) -> ClipModel:
    """Return the process-wide cached CLIP model instance."""
    return ClipModel(model_name)


def load_clip_model() -> ClipModel:
    """Backward-compatible access to the cached model instance."""
    return get_clip_model()
