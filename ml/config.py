"""Wspólna konfiguracja całego pipeline'u (dane, trening, eksport)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FLOWERS_DIR = DATA_DIR / "flower_photos"
CALTECH_DIR = DATA_DIR / "101_ObjectCategories"
SPLITS_FILE = DATA_DIR / "splits.json"
OUTPUTS_DIR = ROOT / "outputs"

# Klasy modelu. Kolejność = indeksy wyjść sieci, nie zmieniać po treningu.
FLOWER_CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
OTHER_CLASS = "other"
CLASSES = FLOWER_CLASSES + [OTHER_CLASS]
CLASS_NAMES_PL = {
    "daisy": "Stokrotka",
    "dandelion": "Mniszek lekarski",
    "roses": "Róża",
    "sunflowers": "Słonecznik",
    "tulips": "Tulipan",
    "other": "To nie kwiat",
}

# Klasa "other" = losowe zdjęcia z Caltech-101 bez kategorii roślinnych.
CALTECH_EXCLUDED = {"sunflower", "lotus", "water_lilly", "bonsai", "strawberry",
                    "BACKGROUND_Google", "Faces_easy"}
OTHER_PER_CATEGORY = 8

# Podział danych (stratyfikowany, powtarzalny).
SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# Wejście sieci. Normalizacja ImageNet, bo backbone był trenowany na ImageNet.
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BATCH_SIZE = 32
# 0 = wczytywanie w procesie głównym. Na macOS workery DataLoadera uruchamiają
# torch_shm_manager, który blokuje zakończenie Pythona; zdjęcia są małe, więc to wystarcza.
NUM_WORKERS = 0
