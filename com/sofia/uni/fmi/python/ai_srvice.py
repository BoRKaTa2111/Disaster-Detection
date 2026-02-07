import io
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "data" / "train" / "saved_model_disaster_classifier.keras"
CLASSES_PATH = BASE_DIR / "data" / "train" / "class_names.json"

IMAGE_SIZE = (224, 224)
DEFAULT_ANSWERS = 3

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
    img = img.resize(IMAGE_SIZE)
    x = np.asarray(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    return x


def predict_array(x: np.ndarray, top_k: int = DEFAULT_ANSWERS) -> Tuple[str, float, List[Tuple[str, float]]]:
    # Validate input early
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"Expected input shape (1,H,W,3). Got: {x.shape}")

    try:
        y = model.predict(x, verbose=0)
    except (tf.errors.InvalidArgumentError, tf.errors.OpError) as e:
        # Typical TF runtime/predict issues (shape mismatch, invalid ops, etc.)
        raise RuntimeError(f"TensorFlow predict failed: {e.__class__.__name__}") from e

    if not isinstance(y, np.ndarray):
        # Keras usually returns np.ndarray, but can vary with settings
        y = np.asarray(y)

    if y.ndim != 2 or y.shape[1] != NUM_CLASSES:
        raise RuntimeError(f"Unexpected model output shape: {y.shape}, expected (*, {NUM_CLASSES})")

    logits = y[0]
    probs = tf.nn.softmax(logits).numpy()

    pred_id = int(np.argmax(probs))
    pred_label = class_names[pred_id]
    pred_conf = float(probs[pred_id])

    top_k = max(1, min(int(top_k), NUM_CLASSES))
    top_idx = np.argsort(probs)[-top_k:][::-1]
    top = [(class_names[i], float(probs[i])) for i in top_idx]

    return pred_label, pred_conf, top


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    top_k: int = Query(DEFAULT_ANSWERS, ge=1, le=100),
):
    # 1. we validate the file type and content
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported media type. Upload an image (image/*).")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    # 2. we decode the image
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()  # force decoding so errors surface
    except UnidentifiedImageError:
        # if image couldn't be decoded image
        raise HTTPException(status_code=400, detail="Could not decode image. Try JPG/PNG/WebP.")
    except OSError:
        # corrupt image
        raise HTTPException(status_code=400, detail="Corrupt or unsupported image data.")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image.") from e

    # 3. preprocess the image for prediction
    try:
        x = preprocess_pil(img)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}") from e

    # 4. prediction
    try:
        label, conf, top = predict_array(x, top_k=top_k)
    except ValueError as e:
        # bad request parameters
        raise HTTPException(status_code=422, detail=str(e)) from e
    except RuntimeError as e:
        # for model prediction issues
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=505, detail="Internal server error.") from e

    return {
        "filename": file.filename,
        "prediction": {"label": label, "confidence": conf},
        "top_k": [{"label": name, "confidence": p} for name, p in top],
    }
