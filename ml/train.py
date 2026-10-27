"""Trening modeli.

  python train.py --model baseline    # CNN od zera (punkt odniesienia)
  python train.py --model mobilenet   # etap 1: zamrożony backbone, etap 2: fine-tuning

Najlepszy checkpoint (wg dokładności na walidacji) z każdego etapu trafia do outputs/.
"""
import argparse
import json
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import config as C
from data import get_device, make_loader
from models import build_model, count_params, freeze_batchnorm, set_backbone_trainable


def run_epoch(model, loader, criterion, device, optimizer=None, freeze_bn=False):
    training = optimizer is not None
    model.train(training)
    if training and freeze_bn:
        freeze_batchnorm(model)

    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
    return total_loss / seen, correct / seen


def train_phase(phase, arch, model, loaders, device, epochs, lr, history, freeze_bn=False):
    out_path = C.OUTPUTS_DIR / f"{phase}.pt"
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    total, trainable = count_params(model)
    print(f"\n== {phase}: {epochs} epok, lr={lr}, parametry trenowalne "
          f"{trainable:,} / {total:,}")

    best_acc = -1.0
    for epoch in range(1, epochs + 1):
        start = time.time()
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, device, optimizer, freeze_bn)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step()

        history.append({"phase": phase, "epoch": epoch, "train_loss": tr_loss,
                        "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        marker = ""
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"arch": arch, "phase": phase, "classes": C.CLASSES,
                        "val_acc": va_acc, "state_dict": model.state_dict()}, out_path)
            marker = "  *"
        print(f"[{phase}] epoka {epoch:2d}/{epochs}  train loss {tr_loss:.3f} acc {tr_acc:.3f}"
              f" | val loss {va_loss:.3f} acc {va_acc:.3f}  ({time.time() - start:.0f}s){marker}")

    model.load_state_dict(torch.load(out_path, map_location=device)["state_dict"])
    print(f"Najlepsza dokładność walidacyjna {best_acc:.4f} -> {out_path.name}")
    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "mobilenet"], required=True)
    parser.add_argument("--epochs", type=int, default=None,
                        help="baseline: liczba epok; mobilenet: epoki etapu 2")
    parser.add_argument("--frozen-epochs", type=int, default=8)
    parser.add_argument("--unfreeze-from", type=int, default=10,
                        help="indeks bloku MobileNetV3.features, od którego odmrażamy")
    args = parser.parse_args()

    torch.manual_seed(C.SEED)
    C.OUTPUTS_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Urządzenie: {device}")

    loaders = {"train": make_loader("train", train=True), "val": make_loader("val")}
    model = build_model(args.model, len(C.CLASSES)).to(device)
    history = []

    if args.model == "baseline":
        train_phase("baseline", "baseline", model, loaders, device,
                    epochs=args.epochs or 30, lr=1e-3, history=history)
    else:
        # Etap 1: feature extraction - uczymy tylko klasyfikator na cechach z ImageNet.
        set_backbone_trainable(model, unfreeze_from=None)
        train_phase("mobilenet_frozen", "mobilenet", model, loaders, device,
                    epochs=args.frozen_epochs, lr=1e-3, history=history, freeze_bn=True)
        # Etap 2: fine-tuning - odmrażamy górne bloki, mały learning rate.
        set_backbone_trainable(model, unfreeze_from=args.unfreeze_from)
        train_phase("mobilenet_finetuned", "mobilenet", model, loaders, device,
                    epochs=args.epochs or 12, lr=1e-4, history=history, freeze_bn=True)

    history_path = C.OUTPUTS_DIR / f"history_{args.model}.json"
    history_path.write_text(json.dumps(history, indent=1))
    print(f"Historia treningu -> {history_path.name}")


if __name__ == "__main__":
    main()
