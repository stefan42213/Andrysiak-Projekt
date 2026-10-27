"""Ewaluacja wytrenowanych modeli na zbiorze testowym.

  python evaluate.py

Wyniki: outputs/metrics.json, outputs/RAPORT.md, macierze pomyłek i krzywe uczenia (PNG).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import config as C
from data import get_device, make_loader
from models import build_model, count_params

CHECKPOINTS = [
    ("baseline", "CNN od zera (baseline)"),
    ("mobilenet_frozen", "MobileNetV3 – zamrożony backbone"),
    ("mobilenet_finetuned", "MobileNetV3 – fine-tuning"),
]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
LABELS_PL = [C.CLASS_NAMES_PL[c] for c in C.CLASSES]


def load_checkpoint(name, device):
    ckpt = torch.load(C.OUTPUTS_DIR / f"{name}.pt", map_location=device)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def predict(model, loader, device):
    probs, targets = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
        targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def threshold_table(probs, targets):
    """Jak próg pewności w apce wpływa na odrzucanie niepewnych klatek."""
    other = C.CLASSES.index(C.OTHER_CLASS)
    conf, pred = probs.max(1), probs.argmax(1)
    is_flower = targets != other
    rows = []
    for t in THRESHOLDS:
        accepted = (conf >= t) & (pred != other)
        flower_accepted = accepted & is_flower
        rows.append({
            "threshold": t,
            # odsetek zdjęć kwiatów, dla których apka pokaże wynik
            "flower_coverage": float(flower_accepted.sum() / is_flower.sum()),
            # trafność wśród pokazanych wyników dla kwiatów
            "flower_accuracy_when_shown": float(
                (pred[flower_accepted] == targets[flower_accepted]).mean()),
            # odsetek "nie-kwiatów", które apka błędnie nazwie kwiatem
            "other_false_alarm": float(accepted[~is_flower].mean()),
        })
    return rows


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS_PL)), LABELS_PL, rotation=35, ha="right")
    ax.set_yticks(range(len(LABELS_PL)), LABELS_PL)
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Prawdziwa klasa")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for model_name in ("baseline", "mobilenet"):
        path = C.OUTPUTS_DIR / f"history_{model_name}.json"
        if not path.exists():
            continue
        history = json.loads(path.read_text())
        steps = range(1, len(history) + 1)
        for ax, key in zip(axes, ("acc", "loss")):
            ax.plot(steps, [h[f"train_{key}"] for h in history], "--", label=f"{model_name} train")
            ax.plot(steps, [h[f"val_{key}"] for h in history], label=f"{model_name} val")
        if model_name == "mobilenet":
            switch = sum(h["phase"] == "mobilenet_frozen" for h in history)
            for ax in axes:
                ax.axvline(switch + 0.5, color="gray", lw=0.8, ls=":")
    axes[0].set_title("Dokładność")
    axes[1].set_title("Strata (cross-entropy)")
    for ax in axes:
        ax.set_xlabel("Epoka")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.OUTPUTS_DIR / "krzywe_uczenia.png", dpi=150)
    plt.close(fig)


def write_report(results):
    lines = ["# Wyniki na zbiorze testowym", "",
             f"Zbiór testowy: {results[0]['n_test']} zdjęć, {len(C.CLASSES)} klas.", "",
             "| Model | Accuracy | Macro F1 | Parametry | Rozmiar FP32 |",
             "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['title']} | {r['accuracy']:.2%} | {r['macro_f1']:.3f} | "
                     f"{r['params']/1e6:.2f} M | {r['size_mb']:.1f} MB |")

    best = results[-1]
    lines += ["", f"## Szczegóły: {best['title']}", "",
              "| Klasa | Precision | Recall | F1 | Liczba |", "|---|---|---|---|---|"]
    for cls in C.CLASSES:
        m = best["per_class"][cls]
        lines.append(f"| {C.CLASS_NAMES_PL[cls]} | {m['precision']:.3f} | {m['recall']:.3f} | "
                     f"{m['f1-score']:.3f} | {int(m['support'])} |")

    lines += ["", "## Próg pewności w aplikacji", "",
              "| Próg | Kwiaty z wynikiem | Trafność pokazanych | Nie-kwiaty uznane za kwiat |",
              "|---|---|---|---|"]
    for row in best["thresholds"]:
        lines.append(f"| {row['threshold']:.1f} | {row['flower_coverage']:.1%} | "
                     f"{row['flower_accuracy_when_shown']:.1%} | {row['other_false_alarm']:.1%} |")
    (C.OUTPUTS_DIR / "RAPORT.md").write_text("\n".join(lines) + "\n")


def main():
    device = get_device()
    loader = make_loader("test")
    results = []
    for name, title in CHECKPOINTS:
        if not (C.OUTPUTS_DIR / f"{name}.pt").exists():
            print(f"Brak {name}.pt – pomijam")
            continue
        model = load_checkpoint(name, device)
        probs, targets = predict(model, loader, device)
        preds = probs.argmax(1)
        params, _ = count_params(model)
        report = classification_report(targets, preds, target_names=C.CLASSES,
                                       output_dict=True, zero_division=0)
        cm = confusion_matrix(targets, preds, labels=range(len(C.CLASSES)))
        plot_confusion(cm, title, C.OUTPUTS_DIR / f"macierz_pomylek_{name}.png")
        results.append({
            "name": name, "title": title, "n_test": int(len(targets)),
            "accuracy": float(accuracy_score(targets, preds)),
            "macro_f1": float(f1_score(targets, preds, average="macro")),
            "params": int(params), "size_mb": params * 4 / 1024 ** 2,
            "per_class": {c: report[c] for c in C.CLASSES},
            "confusion_matrix": cm.tolist(),
            "thresholds": threshold_table(probs, targets),
        })
        r = results[-1]
        print(f"{title:38s} acc {r['accuracy']:.4f}  macro-F1 {r['macro_f1']:.4f}")

    plot_history()
    (C.OUTPUTS_DIR / "metrics.json").write_text(json.dumps(results, indent=1))
    write_report(results)
    print("Zapisano outputs/metrics.json, outputs/RAPORT.md i wykresy PNG")


if __name__ == "__main__":
    main()
