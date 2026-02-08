import numpy as np
import pytest
from PIL import Image

import com.sofia.uni.fmi.python.ai_service as ai_service


class DummyModel:
    def __init__(self, out: np.ndarray):
        self._out = out

    def predict(self, x, verbose=0):
        return self._out


def test_preprocess_pil_shape_dtype():
    img = Image.new("RGB", (500, 300), color=(255, 0, 0))
    x = ai_service.preprocess_pil(img, image_size=(224, 224))

    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float32
    assert x.shape == (1, 224, 224, 3)


def test_predict_array_rejects_bad_input_shape():
    out = np.zeros((1, ai_service.NUM_CLASSES), dtype=np.float32)
    model = DummyModel(out)

    bad_x = np.zeros((224, 224, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        ai_service.predict_array(model, bad_x, top_k=3)


def test_predict_array_raises_on_unexpected_output_shape():
    out = np.zeros((1, 2, 3), dtype=np.float32)
    model = DummyModel(out)

    x = np.zeros((1, 224, 224, 3), dtype=np.float32)

    with pytest.raises(RuntimeError):
        ai_service.predict_array(model, x, top_k=3)


def test_predict_array_topk_sorted_and_clamped():
    out = np.zeros((1, ai_service.NUM_CLASSES), dtype=np.float32)
    out[0, 1] = 10.0
    out[0, 0] = 5.0
    out[0, 2] = 1.0
    model = DummyModel(out)

    x = np.zeros((1, 224, 224, 3), dtype=np.float32)

    label, conf, top = ai_service.predict_array(model, x, top_k=9999)

    assert label == ai_service.class_names[1]
    assert 0.0 <= conf <= 1.0
    assert len(top) == ai_service.NUM_CLASSES
    assert top[0][1] >= top[1][1] >= top[2][1]
