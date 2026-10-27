"""Buduje listę zdjęć i dzieli ją na train/val/test (zapis do data/splits.json).

Zbiory pobiera się raz (patrz ARCHITEKTURA.md):
  data/flower_photos/        - tf_flowers, 5 klas kwiatów
  data/101_ObjectCategories/ - Caltech-101, źródło klasy "other"
"""
import json
import random
from collections import Counter

from PIL import Image
from sklearn.model_selection import train_test_split

import config as C

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def is_readable(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def collect_items():
    items = []
    for cls in C.FLOWER_CLASSES:
        items += [(p, cls) for p in list_images(C.FLOWERS_DIR / cls)]

    rng = random.Random(C.SEED)
    categories = sorted(d for d in C.CALTECH_DIR.iterdir()
                        if d.is_dir() and d.name not in C.CALTECH_EXCLUDED)
    for cat in categories:
        images = list_images(cat)
        rng.shuffle(images)
        items += [(p, C.OTHER_CLASS) for p in images[:C.OTHER_PER_CATEGORY]]

    return [(p, cls) for p, cls in items if is_readable(p)]


def main():
    items = collect_items()
    paths = [str(p.relative_to(C.DATA_DIR)) for p, _ in items]
    labels = [cls for _, cls in items]

    holdout = C.VAL_FRACTION + C.TEST_FRACTION
    train_p, rest_p, train_y, rest_y = train_test_split(
        paths, labels, test_size=holdout, stratify=labels, random_state=C.SEED)
    val_p, test_p, val_y, test_y = train_test_split(
        rest_p, rest_y, test_size=C.TEST_FRACTION / holdout, stratify=rest_y,
        random_state=C.SEED)

    splits = {
        "classes": C.CLASSES,
        "train": [[p, y] for p, y in zip(train_p, train_y)],
        "val": [[p, y] for p, y in zip(val_p, val_y)],
        "test": [[p, y] for p, y in zip(test_p, test_y)],
    }
    C.SPLITS_FILE.write_text(json.dumps(splits, indent=1))

    for name in ("train", "val", "test"):
        counts = Counter(y for _, y in splits[name])
        print(f"{name:5s} {len(splits[name]):5d}  " +
              "  ".join(f"{c}={counts[c]}" for c in C.CLASSES))
    print(f"Zapisano {C.SPLITS_FILE}")


if __name__ == "__main__":
    main()
