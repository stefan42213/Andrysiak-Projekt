"""Eksport modelu do Core ML (.mlpackage) i weryfikacja zgodności z PyTorchem.

  python export_coreml.py [--checkpoint mobilenet_finetuned]

Normalizacja ImageNet i softmax są wbudowane w model, więc aplikacja iOS
podaje surową klatkę z kamery, a dostaje gotowe prawdopodobieństwa klas.
"""
import argparse
import json
import shutil
import time

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T

import config as C
from data import build_transform, load_splits
from evaluate import load_checkpoint

IOS_MODEL_PATH = C.ROOT.parent / "ios" / "FlowerCam" / "FlowerClassifier.mlpackage"


class ExportWrapper(nn.Module):
    """Wejście: obraz RGB w zakresie [0, 1]. Wyjście: prawdopodobieństwa klas."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(C.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(C.IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return torch.softmax(self.model((x - self.mean) / self.std), dim=1)


def convert(model):
    wrapper = ExportWrapper(model).eval()
    example = torch.rand(1, 3, C.IMG_SIZE, C.IMG_SIZE)
    traced = torch.jit.trace(wrapper, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=example.shape, scale=1 / 255.0,
                             color_layout=ct.colorlayout.RGB)],
        classifier_config=ct.ClassifierConfig(C.CLASSES),
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )
    mlmodel.author = "Dawid Stefański"
    mlmodel.short_description = ("Klasyfikator kwiatów (MobileNetV3-Large, transfer learning): "
                                 + ", ".join(C.CLASS_NAMES_PL.values()))
    mlmodel.input_description["image"] = "Zdjęcie RGB 224x224"
    mlmodel.user_defined_metadata["class_names_pl"] = json.dumps(C.CLASS_NAMES_PL, ensure_ascii=False)
    return mlmodel


def parity_check(model, mlmodel):
    """Porównuje predykcje PyTorch (FP32) i Core ML (FP16) na całym zbiorze testowym."""
    torch_tf = build_transform(train=False)
    coreml_tf = T.Compose([T.Resize(C.IMG_SIZE), T.CenterCrop(C.IMG_SIZE)])
    probs_key = mlmodel.get_spec().description.predictedProbabilitiesName

    agree, coreml_correct, torch_correct, max_diff = 0, 0, 0, 0.0
    items = load_splits()["test"]
    with torch.no_grad():
        for rel_path, cls in items:
            img = Image.open(C.DATA_DIR / rel_path).convert("RGB")
            p_torch = torch.softmax(model(torch_tf(img).unsqueeze(0)), dim=1)[0].numpy()
            out = mlmodel.predict({"image": coreml_tf(img)})
            p_coreml = np.array([out[probs_key][c] for c in C.CLASSES])

            target = C.CLASSES.index(cls)
            agree += p_torch.argmax() == p_coreml.argmax()
            torch_correct += p_torch.argmax() == target
            coreml_correct += p_coreml.argmax() == target
            max_diff = max(max_diff, float(np.abs(p_torch - p_coreml).max()))

    n = len(items)
    return {"n_test": n, "top1_agreement": agree / n, "torch_accuracy": torch_correct / n,
            "coreml_accuracy": coreml_correct / n, "max_prob_diff": max_diff}


def benchmark(path, compute_units, runs=100):
    mlmodel = ct.models.MLModel(str(path), compute_units=compute_units)
    img = Image.new("RGB", (C.IMG_SIZE, C.IMG_SIZE), (120, 160, 90))
    for _ in range(10):
        mlmodel.predict({"image": img})
    start = time.perf_counter()
    for _ in range(runs):
        mlmodel.predict({"image": img})
    return (time.perf_counter() - start) / runs * 1000


def append_to_report(result):
    """Dopisuje (lub podmienia) sekcję Core ML w outputs/RAPORT.md z evaluate.py."""
    report = C.OUTPUTS_DIR / "RAPORT.md"
    header = "## Model Core ML (aplikacja iOS)"
    base = report.read_text().split(f"\n{header}")[0].rstrip() if report.exists() else ""
    agree = round(result["top1_agreement"] * result["n_test"])
    lines = ["", "", header, "",
             "| Metryka | Wartość |", "|---|---|",
             f"| Zgodność top-1 PyTorch (FP32) ↔ Core ML (FP16) | "
             f"{result['top1_agreement']:.1%} ({agree}/{result['n_test']}) |",
             f"| Accuracy Core ML na zbiorze testowym | {result['coreml_accuracy']:.2%} |",
             f"| Maks. różnica prawdopodobieństw | {result['max_prob_diff']:.3f} |",
             f"| Rozmiar FlowerClassifier.mlpackage | {result['size_mb']:.1f} MB |",
             f"| Czas inferencji na Macu – tylko CPU | {result['mac_latency_ms']['cpu']:.2f} ms |",
             f"| Czas inferencji na Macu – CPU+GPU+Neural Engine | "
             f"{result['mac_latency_ms']['all_neural_engine']:.2f} ms |"]
    report.write_text(base + "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="mobilenet_finetuned")
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint, "cpu")
    mlmodel = convert(model)
    out_path = C.OUTPUTS_DIR / "FlowerClassifier.mlpackage"
    shutil.rmtree(out_path, ignore_errors=True)
    mlmodel.save(str(out_path))
    size_mb = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file()) / 1024 ** 2
    print(f"Zapisano {out_path.name} ({size_mb:.1f} MB)")

    result = parity_check(model, mlmodel)
    result["size_mb"] = size_mb
    result["mac_latency_ms"] = {
        "cpu": benchmark(out_path, ct.ComputeUnit.CPU_ONLY),
        "all_neural_engine": benchmark(out_path, ct.ComputeUnit.ALL),
    }
    (C.OUTPUTS_DIR / "coreml_parity.json").write_text(json.dumps(result, indent=1))
    append_to_report(result)
    print(json.dumps(result, indent=1))

    shutil.rmtree(IOS_MODEL_PATH, ignore_errors=True)
    shutil.copytree(out_path, IOS_MODEL_PATH)
    print(f"Skopiowano model do {IOS_MODEL_PATH}")


if __name__ == "__main__":
    main()
