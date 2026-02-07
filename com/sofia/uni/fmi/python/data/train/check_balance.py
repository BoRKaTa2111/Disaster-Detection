from pathlib import Path

# used for checking the balance of the dataset (how many elements each class has) before training the model

root = Path("../Comprehensive_Disaster_Dataset(CDD)").resolve()
exts = {".jpg", ".jpeg", ".png"}

for c in sorted([p for p in root.iterdir() if p.is_dir()]):
    n = sum(1 for f in c.rglob("*") if f.is_file() and f.suffix.lower() in exts)
    print(f"{c.name:25s} {n}")