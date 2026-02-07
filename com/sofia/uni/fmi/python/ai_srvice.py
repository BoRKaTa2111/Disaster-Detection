import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array

# ---------- Paths (robust) ----------
BASE_DIR = Path(__file__).resolve().parent  # folder of ai_srvice.py

MODEL_PATH = BASE_DIR / "data" / "train" / "saved_model_disaster_classifier.keras"
CLASSES_PATH = BASE_DIR / "data" / "train" / "class_names.json"

# Your dataset folder (only needed for the optional sanity test)
DATASET_DIR = BASE_DIR / "data" / "Comprehensive_Disaster_Dataset(CDD)"

# Folder where YOU put a few hand-picked test images
TEST_IMAGES_DIR = BASE_DIR / "data" / "local_testing"

IMG_SIZE = (224, 224)

# ---------- Load model + classes ----------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

if not CLASSES_PATH.exists():
    raise FileNotFoundError(f"class_names.json not found at: {CLASSES_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

num_classes = len(class_names)
print("Loaded model.")
print("Classes:", class_names)

# ---------- Prediction ----------
def predict_image(image_path: Path, top_k: int = 3):
    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found: {image_path}")

    img = load_img(str(image_path), target_size=IMG_SIZE)
    x = img_to_array(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)

    logits = model.predict(x, verbose=0)
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]

    pred_id = int(np.argmax(probs))
    pred_label = class_names[pred_id]
    pred_conf = float(probs[pred_id])

    top_idx = np.argsort(probs)[-top_k:][::-1]
    top = [(class_names[i], float(probs[i])) for i in top_idx]

    return pred_label, pred_conf, top

def print_prediction(image_path: Path, top_k: int = 3):
    label, conf, top = predict_image(image_path, top_k=top_k)
    print(f"\nImage: {image_path}")
    print(f"Prediction: {label} (confidence {conf:.3f})")
    print(f"Top-{top_k}:")
    for name, p in top:
        print(f"  {name}: {p:.3f}")

# ---------- Optional: sanity test on the dataset itself ----------
def pick_random_image_from_class(class_folder: str) -> Path:
    exts = {".jpg", ".jpeg", ".png"}
    folder = DATASET_DIR / class_folder
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if not files:
        raise RuntimeError(f"No images found in class folder: {folder}")
    return random.choice(files)

def sanity_test_one_per_class():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset dir not found at: {DATASET_DIR}")

    print("\n--- Sanity test: 1 random image from EACH class folder ---")
    for cls in class_names:
        p = pick_random_image_from_class(cls)
        label, conf, top = predict_image(p, top_k=3)
        print(f"True folder: {cls:25s} -> Pred: {label:25s} ({conf:.3f})   file={p.name}")

# ---------- Main ----------
if __name__ == "__main__":
    # 1) Test your own images from data/test_images/
    TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    test_files = [
        "wildfire_1.jpeg",
        "landslide_1.jpeg",
        "flood_1.jpeg",
    ]

    for fname in test_files:
        p = TEST_IMAGES_DIR / fname
        print_prediction(p, top_k=3)

    # 2) Uncomment this to verify model behavior on the dataset itself
    # sanity_test_one_per_class()
