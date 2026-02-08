import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

import ai_service

client = TestClient(ai_service.app)


class TestModel:
    def predict(self, x, verbose=0):
        out = np.zeros((1, ai_service.NUM_CLASSES), dtype=np.float32)
        out[0, 0] = 10.0
        return out

def make_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (67, 67), color=(0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def test_predict_rejects_non_image_content_type():
    r = client.post("/predict", files={"file": ("kuche.txt", b"hello", "text/plain")})
    assert r.status_code == 415

def test_predict_rejects_empty_file():
    r = client.post("/predict", files={"file": ("kote.jpg", b"", "image/jpeg")})
    assert r.status_code == 400

def test_predict_rejects_invalid_image_bytes():
    r = client.post("/predict", files={"file": ("hamster.jpg", b"not-an-image", "image/jpeg")})
    assert r.status_code == 400

def test_predict_success_with_mock_model(monkeypatch):
    monkeypatch.setattr(ai_service, "get_model_and_size", lambda model_name: (TestModel(), (224, 224)))

    img_bytes = make_jpeg_bytes()
    r = client.post(
        "/predict?top_k=3&model_name=224x224",
        files={"file": ("kote2.jpg", img_bytes, "image/jpeg")},
    )

    assert r.status_code == 200
    body = r.json()

    assert body["model_name"] == "224x224"
    assert body["prediction"]["label"] == ai_service.class_names[0]
    assert 0.0 <= body["prediction"]["confidence"] <= 1.0
    assert len(body["top_k"]) == 3

def test_predict_unknown_model_name():
    img_bytes = make_jpeg_bytes()
    r = client.post(
        "/predict?model_name=does_not_exist",
        files={"file": ("kuche2.jpg", img_bytes, "image/jpeg")},
    )
    assert r.status_code == 400
    assert "Unknown model_name" in r.text

def test_models_endpoint_returns_registry():
    r = client.get("/models")
    assert r.status_code == 200
    data = r.json()
    assert "default" in data
    assert "image_size" in data["default"]

def test_predict_top_k_validation_error(monkeypatch):
    monkeypatch.setattr(ai_service, "get_model_and_size", lambda model_name: (TestModel(), (224, 224)))

    img_bytes = make_jpeg_bytes()

    r = client.post(
        "/predict?top_k=0&model_name=224x224",
        files={"file": ("img.jpg", img_bytes, "image/jpeg")},
    )
    assert r.status_code == 422


def test_predict_array_runtime_error_branch(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("forced runtime error")

    monkeypatch.setattr(ai_service, "get_model_and_size", lambda model_name: (TestModel(), (224, 224)))
    monkeypatch.setattr(ai_service, "predict_array", boom)

    img_bytes = make_jpeg_bytes()
    r = client.post(
        "/predict?top_k=3&model_name=224x224",
        files={"file": ("img.jpg", img_bytes, "image/jpeg")},
    )

    assert r.status_code == 500
    assert "forced runtime error" in r.text
