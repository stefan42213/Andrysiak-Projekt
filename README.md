# FlowerCam – rozpoznawanie kwiatów w czasie rzeczywistym na iPhonie

Projekt z przedmiotu *Zaawansowane techniki sztucznej inteligencji*.
Sieć MobileNetV3 wytrenowana na ImageNet została douczona (transfer learning + fine-tuning)
do rozpoznawania 5 gatunków kwiatów oraz klasy „to nie kwiat”, przekonwertowana do Core ML
i uruchomiona w aplikacji na iPhone'a, która rozpoznaje kwiaty na żywo z kamery – bez internetu.

## Wyniki (zbiór testowy, 665 zdjęć)

| Model | Accuracy | Macro F1 |
|---|---|---|
| CNN od zera (model bazowy) | 82,9% | 0,823 |
| MobileNetV3 – zamrożony backbone | 94,3% | 0,942 |
| **MobileNetV3 – fine-tuning** | **95,5%** | **0,953** |

Model Core ML (8,1 MB) daje te same odpowiedzi co PyTorch dla 665/665 zdjęć testowych.
Na iPhonie 17 aplikacja przetwarza ok. 30 klatek na sekundę.

## Struktura

```
ml/    Python: przygotowanie danych, trening, ewaluacja, eksport do Core ML
ios/   aplikacja iOS (SwiftUI, AVFoundation, Vision, Core ML)
```

## Dokumentacja

- [Dawid_Stefanski_122429_Dokumentacja.pdf](Dawid_Stefanski_122429_Dokumentacja.pdf) – pełna dokumentacja projektu
- [ARCHITEKTURA.md](ARCHITEKTURA.md) – architektura i instrukcja uruchomienia
- [SPECYFIKACKA DLA PROJEKTU.docx](SPECYFIKACKA%20DLA%20PROJEKTU.docx) – specyfikacja projektu
- [ml/outputs/RAPORT.md](ml/outputs/RAPORT.md) – szczegółowe wyniki, wykresy w `ml/outputs/`

## Szybki start

```bash
cd ml
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# pobranie danych – patrz ARCHITEKTURA.md, sekcja 8
./run_all.sh
```

Aplikację uruchamia się z `ios/FlowerCam.xcodeproj` w Xcode (Run na podłączonym iPhonie).

Autor: Dawid Stefański
