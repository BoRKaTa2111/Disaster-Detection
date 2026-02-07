from pathlib import Path
import tensorflow as tf
import shutil

# deletion of corrupted images from the dataset
# we use tensorflow's image decoding functions to check if the images can be read properly

root = Path("../Comprehensive_Disaster_Dataset(CDD)").resolve()
quarantine = Path("../bad_images_quarantine").resolve()
quarantine.mkdir(parents=True, exist_ok=True)

bad = []
exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

for p in root.rglob("*"):
    if not p.is_file():
        continue
    ext = p.suffix.lower()
    if ext not in exts:
        continue

    try:
        data = tf.io.read_file(str(p))

        if ext in {".jpg", ".jpeg"}:
            _ = tf.io.decode_jpeg(data, channels=3)
        elif ext == ".png":
            _ = tf.io.decode_png(data, channels=3)
        elif ext == ".bmp":
            _ = tf.io.decode_bmp(data)
        elif ext == ".gif":
            _ = tf.io.decode_gif(data)
        elif ext == ".webp":
            _ = tf.io.decode_webp(data)

    except Exception as e:
        bad.append((p, str(e)))

print("Bad images:", len(bad))
for p, e in bad:
    print(p, "->", e)

for p, _ in bad:
    target = quarantine / p.name
    if target.exists():
        target = quarantine / f"{p.stem}_{p.parent.name}{p.suffix}"
    shutil.move(str(p), str(target))

print("Moved bad images to:", quarantine)
