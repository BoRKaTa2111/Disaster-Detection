import io
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "data" / "train" / "saved_model_disaster_classifier.keras"
CLASSES_PATH = BASE_DIR / "data" / "train" / "class_names.json"

IMG_SIZE = (224, 224)
DEFAULT_TOP_K = 3

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
if not CLASSES_PATH.exists():
    raise FileNotFoundError(f"class_names.json not found at: {CLASSES_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names: List[str] = json.load(f)

NUM_CLASSES = len(class_names)

app = FastAPI(title="Disaster Classifier API", version="1.0")


def preprocess_pil(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    x = np.asarray(img, dtype=np.float32)  # [0..255]
    x = np.expand_dims(x, axis=0)
    return x


def predict_array(x: np.ndarray, top_k: int = DEFAULT_TOP_K) -> Tuple[str, float, List[Tuple[str, float]]]:
    y = model.predict(x, verbose=0)
    if y.ndim != 2 or y.shape[1] != NUM_CLASSES:
        raise RuntimeError(f"Unexpected model output shape: {y.shape}")

    logits = y[0]
    probs = tf.nn.softmax(logits).numpy()

    pred_id = int(np.argmax(probs))
    pred_label = class_names[pred_id]
    pred_conf = float(probs[pred_id])

    top_k = max(1, min(int(top_k), NUM_CLASSES))
    top_idx = np.argsort(probs)[-top_k:][::-1]
    top = [(class_names[i], float(probs[i])) for i in top_idx]

    return pred_label, pred_conf, top

@app.get("/health")
def health():
    return {"status": "ok", "num_classes": NUM_CLASSES, "classes": class_names}


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = DEFAULT_TOP_K):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file (content-type image/*).")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        img = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image. Try JPG/PNG.")

    x = preprocess_pil(img)
    try:
        label, conf, top = predict_array(x, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "filename": file.filename,
        "prediction": {"label": label, "confidence": conf},
        "top_k": [{"label": name, "confidence": p} for name, p in top],
    }
