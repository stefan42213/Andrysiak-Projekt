# Wyniki na zbiorze testowym

Zbiór testowy: 665 zdjęć, 6 klas.

| Model | Accuracy | Macro F1 | Parametry | Rozmiar FP32 |
|---|---|---|---|---|
| CNN od zera (baseline) | 82.86% | 0.823 | 2.36 M | 9.0 MB |
| MobileNetV3 – zamrożony backbone | 94.29% | 0.942 | 4.21 M | 16.1 MB |
| MobileNetV3 – fine-tuning | 95.49% | 0.953 | 4.21 M | 16.1 MB |

## Szczegóły: MobileNetV3 – fine-tuning

| Klasa | Precision | Recall | F1 | Liczba |
|---|---|---|---|---|
| Stokrotka | 0.938 | 0.958 | 0.948 | 95 |
| Mniszek lekarski | 0.971 | 0.978 | 0.974 | 135 |
| Róża | 0.893 | 0.958 | 0.925 | 96 |
| Słonecznik | 0.980 | 0.924 | 0.951 | 105 |
| Tulipan | 0.948 | 0.908 | 0.928 | 120 |
| To nie kwiat | 0.991 | 1.000 | 0.996 | 114 |

## Próg pewności w aplikacji

| Próg | Kwiaty z wynikiem | Trafność pokazanych | Nie-kwiaty uznane za kwiat |
|---|---|---|---|
| 0.5 | 93.5% | 97.5% | 0.0% |
| 0.6 | 89.1% | 98.6% | 0.0% |
| 0.7 | 84.2% | 99.4% | 0.0% |
| 0.8 | 75.7% | 99.8% | 0.0% |
| 0.9 | 43.9% | 100.0% | 0.0% |

## Model Core ML (aplikacja iOS)

| Metryka | Wartość |
|---|---|
| Zgodność top-1 PyTorch (FP32) ↔ Core ML (FP16) | 100.0% (665/665) |
| Accuracy Core ML na zbiorze testowym | 95.49% |
| Maks. różnica prawdopodobieństw | 0.017 |
| Rozmiar FlowerClassifier.mlpackage | 8.1 MB |
| Czas inferencji na Macu – tylko CPU | 1.93 ms |
| Czas inferencji na Macu – CPU+GPU+Neural Engine | 0.42 ms |
