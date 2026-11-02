import numpy as np
from PIL import Image

from visionsearch_web_ui.region.embedder import embed_crops


class BatchClipModel:
    def __init__(self):
        self.batch_sizes = []

    def encode_image(self, image):
        raise AssertionError("Batch encoder should be used when available")

    def encode_images(self, images):
        self.batch_sizes.append(len(images))
        return np.ones((len(images), 512), dtype=np.float32)


def test_embed_crops_uses_batched_clip_encoder():
    model = BatchClipModel()
    crops = [Image.new("RGB", (16, 16)) for _ in range(10)]

    vectors = embed_crops(crops, model, batch_size=4, show_progress=False)

    assert model.batch_sizes == [4, 4, 2]
    assert vectors.shape == (10, 512)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_embed_crops_rejects_non_positive_batch_size():
    model = BatchClipModel()

    try:
        embed_crops([Image.new("RGB", (16, 16))], model, batch_size=0)
    except ValueError as error:
        assert str(error) == "batch_size must be greater than zero"
    else:
        raise AssertionError("Expected invalid batch size to be rejected")
