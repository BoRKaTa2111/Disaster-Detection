import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "data" / "train"
CLASSES_PATH = MODELS_DIR / "class_names.json"

DEFAULT_ANSWERS = 3

# registry of available models
MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    "default": {
        "path": MODELS_DIR / "saved_model_disaster_classifier.keras",
        "image_size": (224, 224),
    },
    "32x32": {
        "path": MODELS_DIR / "saved_model_disaster_classifier_32x32.keras",
        "image_size": (32, 32),
    },
    "64x64": {
        "path": MODELS_DIR / "saved_model_disaster_classifier_64x64.keras",
        "image_size": (64, 64),
    },
    "128x128": {
        "path": MODELS_DIR / "saved_model_disaster_classifier_128x128.keras",
        "image_size": (128, 128),
    },
    "224x224": {
        "path": MODELS_DIR / "saved_model_disaster_classifier_224x224.keras",
        "image_size": (224, 224),
    },
    "256x256": {
        "path": MODELS_DIR / "saved_model_disaster_classifier_256x256.keras",
        "image_size": (256, 256),
    },
    "patience4": {
        "path": MODELS_DIR / "saved_model_disaster_classifier_patience=4.keras",
        "image_size": (224, 224),
    },
}

# check if class names file exists and load it
if not CLASSES_PATH.exists():
    raise FileNotFoundError(f"class_names.json not found at: {CLASSES_PATH}")

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names: List[str] = json.load(f)

NUM_CLASSES = len(class_names)

app = FastAPI(title="Disaster Classifier API", version="1.0")

# Cache loaded models
_MODEL_CACHE: Dict[str, tf.keras.Model] = {}


def get_model_and_size(model_name: str) -> Tuple[tf.keras.Model, Tuple[int, int]]:
    # Returns (model, image_size) for the requested model_name.
    # Loads the model once and caches it in memory.
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_name '{model_name}'. Available: {sorted(MODEL_REGISTRY.keys())}",
        )

    meta = MODEL_REGISTRY[model_name]
    model_path = meta["path"]
    image_size = meta["image_size"]

    if not isinstance(model_path, Path) or not model_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found for '{model_name}': {model_path}",
        )

    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = tf.keras.models.load_model(model_path, compile=False)

    return _MODEL_CACHE[model_name], image_size  # type: ignore[return-value]


def preprocess_pil(img: Image.Image, image_size: Tuple[int, int]) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize(image_size)
    x = np.asarray(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    return x


def predict_array(
    model: tf.keras.Model,
    x: np.ndarray,
    top_k: int = DEFAULT_ANSWERS
) -> Tuple[str, float, List[Tuple[str, float]]]:
    # validate input
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"Expected input shape (1,H,W,3). Got: {x.shape}")

    try:
        y = model.predict(x, verbose=0)
    except (tf.errors.InvalidArgumentError, tf.errors.OpError) as e:
        raise RuntimeError(f"TensorFlow predict failed: {e.__class__.__name__}") from e

    if not isinstance(y, np.ndarray):
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


@app.get("/models")
def list_models():
    """
    Quick helper endpoint so you (or your friend) can see which model_name values are valid.
    """
    out = {}
    for name, meta in MODEL_REGISTRY.items():
        out[name] = {
            "file": str(meta["path"]),
            "image_size": meta["image_size"],
            "exists": Path(meta["path"]).exists(),
        }
    return out

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    top_k: int = Query(DEFAULT_ANSWERS, ge=1, le=100),
    model_name: str = Query("default"),
):
    # choose model and required input size
    model, image_size = get_model_and_size(model_name)

    # 1. validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported media type. Upload an image (image/*).")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    # 2. decode image
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not decode image. Try JPG/PNG/WebP.")
    except OSError:
        raise HTTPException(status_code=400, detail="Corrupt or unsupported image data.")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image.") from e

    # 3. preprocess to the selected model's expected size
    try:
        x = preprocess_pil(img, image_size=image_size)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}") from e

    # 4. predict
    try:
        label, conf, top = predict_array(model, x, top_k=top_k)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error.") from e

    return {
        "filename": file.filename,
        "model_name": model_name,
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "prediction": {"label": label, "confidence": conf},
        "top_k": [{"label": name, "confidence": p} for name, p in top],
    }
