"""Dataset, augmentacja i DataLoadery na podstawie data/splits.json."""
import json

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import config as C


class ImageListDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(C.CLASSES)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel_path, cls = self.items[i]
        img = Image.open(C.DATA_DIR / rel_path).convert("RGB")
        return self.transform(img), self.class_to_idx[cls]


def build_transform(train):
    normalize = T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD)
    if train:
        # Augmentacja symuluje warunki z kamery telefonu:
        # inne kadrowanie, oświetlenie, lekkie rozmycie ruchu.
        return T.Compose([
            T.RandomResizedCrop(C.IMG_SIZE, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
            T.ToTensor(),
            normalize,
        ])
    # Ewaluacja = to samo co Vision na iPhonie (.centerCrop):
    # krótszy bok -> 224, potem wycięcie środka 224x224.
    return T.Compose([
        T.Resize(C.IMG_SIZE),
        T.CenterCrop(C.IMG_SIZE),
        T.ToTensor(),
        normalize,
    ])


def load_splits():
    return json.loads(C.SPLITS_FILE.read_text())


def make_loader(split, train=False, batch_size=C.BATCH_SIZE):
    items = load_splits()[split]
    dataset = ImageListDataset(items, build_transform(train))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train,
                      num_workers=C.NUM_WORKERS, persistent_workers=C.NUM_WORKERS > 0,
                      drop_last=train)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
