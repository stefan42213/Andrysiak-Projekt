# FlowerCam – rozpoznawanie kwiatów w czasie rzeczywistym na iPhonie

Projekt z przedmiotu *Zaawansowane techniki sztucznej inteligencji* (temat nr 8 z listy:
klasyfikator obrazów kwiatów z MobileNet), rozszerzony o działanie na żywo z kamery iPhone'a.

**Idea w jednym zdaniu:** bierzemy sieć MobileNetV3 wytrenowaną na 1,28 mln zdjęć ImageNet,
douczamy ją (transfer learning + fine-tuning) na 5 gatunkach kwiatów i klasie „to nie kwiat”,
konwertujemy do Core ML i uruchamiamy na Neural Engine iPhone'a na każdej klatce z kamery.

---

## 1. Architektura – widok ogólny

```
┌──────────────────────────── ETAP OFFLINE (Mac, Python) ────────────────────────────┐
│                                                                                     │
│  tf_flowers (3670 zdjęć, 5 klas)  ─┐                                                │
│                                    ├─► prepare_data.py ─► splits.json (70/15/15)    │
│  Caltech-101 (760 zdjęć, "inne")  ─┘          │                                     │
│                                               ▼                                     │
│                         data.py (augmentacja, DataLoader)                           │
│                                               │                                     │
│              ┌────────────────────────────────┼─────────────────────────┐           │
│              ▼                                ▼                         │           │
│   train.py --model baseline       train.py --model mobilenet            │           │
│   CNN od zera (punkt odniesienia) etap 1: zamrożony backbone            │           │
│                                   etap 2: fine-tuning górnych bloków    │           │
│              │                                │                         │           │
│              └──────────────┬─────────────────┘                         │           │
│                             ▼                                           │           │
│                 evaluate.py (accuracy, F1, macierz pomyłek, próg)       │           │
│                             │                                           │           │
│                             ▼                                           │           │
│     export_coreml.py: normalizacja + softmax wbudowane w model,         │           │
│     FP16, weryfikacja zgodności PyTorch ↔ Core ML                        │           │
│                             │                                           │           │
└─────────────────────────────┼───────────────────────────────────────────────────────┘
                              ▼  FlowerClassifier.mlpackage (8 MB)
┌──────────────────────────── ETAP ONLINE (iPhone, Swift) ───────────────────────────┐
│                                                                                     │
│  AVCaptureSession (720p, 30 kl/s)                                                   │
│        │ CVPixelBuffer (BGRA)            kolejka "camera.frames"                    │
│        ▼                                                                            │
│  Vision: obrót .right + centerCrop → 224×224                                        │
│        ▼                                                                            │
│  Core ML (Neural Engine): 6 prawdopodobieństw                                       │
│        ▼                                                                            │
│  Wygładzanie w czasie (średnia wykładnicza) + próg pewności                         │
│        ▼                                  wątek główny                              │
│  SwiftUI: podgląd kamery, nazwa kwiatu, pewność, top-3, FPS, ms/klatkę              │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

Cała inferencja odbywa się na telefonie – bez internetu i serwera, obraz nie opuszcza urządzenia.

---

## 2. Dane

| Zbiór | Rola | Liczba zdjęć |
|---|---|---|
| tf_flowers (TensorFlow) | 5 klas kwiatów: stokrotka, mniszek, róża, słonecznik, tulipan | 3670 |
| Caltech-101 | klasa „to nie kwiat” – po 8 losowych zdjęć z 95 kategorii przedmiotów, zwierząt, pojazdów | 760 |

- **Dlaczego klasa „inne”?** Klasyfikator zamkniętego zbioru zawsze wybierze jeden z kwiatów –
  nawet dla kubka czy twarzy. Na żywo z kamery to kluczowe, bo większość klatek kwiatem nie jest.
  Z Caltech-101 wykluczono kategorie roślinne (sunflower, lotus, water_lilly, bonsai, strawberry).
- **Podział:** stratyfikowany 70 / 15 / 15 (train 3101, val 664, test 665), ziarno 42,
  zapisany w `data/splits.json` → wyniki są powtarzalne.
- **Augmentacja (tylko trening)** symuluje warunki z kamery telefonu: losowe kadrowanie
  (RandomResizedCrop 50–100%), odbicie, obrót ±15°, zmiany jasności/kontrastu/nasycenia,
  sporadyczne rozmycie Gaussa (ruch ręki, brak ostrości).
- **Ewaluacja** używa dokładnie tego samego przetwarzania co Vision na iPhonie:
  krótszy bok → 224 px, wycięcie środka 224×224.

---

## 3. Modele i trening

### 3.1 Baseline – CNN od zera
5 bloków `[Conv3×3 → BN → ReLU] ×2 → MaxPool` (32→64→128→256→256 kanałów),
Global Average Pooling, Dropout 0,3, warstwa liniowa. 2,36 mln parametrów, 30 epok.
Rola: pokazać, ile daje transfer learning przy małym zbiorze danych.

### 3.2 MobileNetV3-Large + transfer learning
- Backbone: MobileNetV3-Large z wagami ImageNet (`IMAGENET1K_V2`), 4,2 mln parametrów,
  ok. 0,22 GFLOP na zdjęcie – architektura projektowana pod telefony
  (depthwise separable convolutions, bloki squeeze-and-excitation, aktywacja hard-swish).
- Głowica: oryginalna `Linear(1280 → 1000)` zastąpiona przez `Linear(1280 → 6)`.

**Etap 1 – feature extraction (8 epok, lr 1e-3).** Cały backbone zamrożony, uczymy tylko
klasyfikator (1,24 mln parametrów). Sieć wykorzystuje cechy wyuczone na ImageNet.

**Etap 2 – fine-tuning (12 epok, lr 1e-4).** Odmrażamy bloki `features[10:]` (4,05 mln
parametrów) i douczamy je z 10× mniejszym learning rate, żeby nie zniszczyć wyuczonych cech.

Wspólne ustawienia: AdamW (weight decay 1e-4), harmonogram cosine annealing,
label smoothing 0,1, batch 32, BatchNorm zamrożony (statystyki z ImageNet są stabilniejsze
niż liczone z małych batchy), zapis najlepszego checkpointu wg dokładności walidacyjnej.
Trening na GPU Maca (backend MPS).

---

## 4. Ewaluacja

`evaluate.py` liczy na zbiorze testowym (nieużywanym w treningu):
- accuracy i macro F1 (średnia F1 po klasach – odporna na różne liczności klas),
- precision / recall / F1 dla każdej klasy,
- macierz pomyłek (PNG),
- krzywe uczenia (PNG),
- **analizę progu pewności** – dla progów 0,5…0,9: jaki odsetek kwiatów dostaje wynik,
  jak trafne są pokazane wyniki i jak często „nie-kwiat” jest błędnie nazywany kwiatem.
  Na tej podstawie dobrany jest domyślny próg w aplikacji.

Wyniki: `ml/outputs/RAPORT.md`, `ml/outputs/metrics.json`.

---

## 5. Eksport do Core ML

`export_coreml.py`:
1. Opakowuje sieć w `ExportWrapper`: wejście w zakresie [0, 1] → normalizacja ImageNet →
   MobileNet → softmax. Dzięki temu aplikacja nie musi znać żadnych stałych z treningu.
2. `torch.jit.trace` → `coremltools.convert` jako **ML Program** z wejściem typu obraz
   (`ImageType`, skala 1/255, RGB) i `ClassifierConfig` z etykietami klas.
3. Precyzja **FP16** – 2× mniejszy model i natywny format Neural Engine.
4. **Weryfikacja zgodności**: każde zdjęcie testowe przechodzi przez PyTorch (FP32)
   i Core ML (FP16); porównywana jest klasa top-1 i maksymalna różnica prawdopodobieństw.
5. Pomiar opóźnienia Core ML na Macu (CPU vs Neural Engine).
6. Kopiuje model do projektu iOS (`ios/FlowerCam/FlowerClassifier.mlpackage`).

Wersje są przypięte (`torch 2.7.0` + `coremltools 9.0`), bo nowszy PyTorch nie jest
przetestowany z konwerterem.

---

## 6. Aplikacja iOS (SwiftUI)

| Plik | Odpowiedzialność |
|---|---|
| `FlowerCamApp.swift` | punkt wejścia aplikacji |
| `CameraManager.swift` | sesja AVFoundation: uprawnienia, kamera tylna 720p, wyjście BGRA, `alwaysDiscardsLateVideoFrames` |
| `CameraPreview.swift` | podgląd na żywo (`AVCaptureVideoPreviewLayer` w SwiftUI) |
| `FlowerRecognizer.swift` | Vision + Core ML, wygładzanie wyników, pomiar ms/klatkę i FPS |
| `ContentView.swift` | interfejs: ramka celownika, pasek FPS/ms, karta wyniku, top-3, suwak progu |
| `FlowerClassifier.mlpackage` | model; Xcode generuje z niego klasę Swift `FlowerClassifier` |

**Przepływ jednej klatki:**
1. Kamera dostarcza `CVPixelBuffer` na szeregowej kolejce `camera.frames`.
2. `VNImageRequestHandler` obraca klatkę do pionu (`.right` – sensor jest poziomy),
   `VNCoreMLRequest` z `.centerCrop` skaluje ją do 224×224 – identycznie jak w ewaluacji.
3. Core ML (`computeUnits = .all`) wykonuje sieć, zwykle na Neural Engine.
4. Prawdopodobieństwa są wygładzane średnią wykładniczą (waga nowej klatki 0,35),
   żeby wynik nie „migał” między klatkami.
5. Wynik trafia na wątek główny i do SwiftUI.

**Decyzja wyświetlana użytkownikowi:**
- najwyższa klasa = „inne” → **„To nie kwiat”** (pomarańczowy),
- pewność < próg (domyślnie 70%, regulowany suwakiem) → **„Nie jestem pewien”** (żółty),
- w przeciwnym razie → **nazwa kwiatu + pewność** (zielony).

**Wydajność:** gdy model nie nadąża, stare klatki są odrzucane (brak narastającego opóźnienia).
FPS i czas inferencji są widoczne na ekranie – służą jako metryki wydajności projektu.

---

## 7. Struktura projektu

```
Andrysiak Projekt/
├── ARCHITEKTURA.md              ← ten dokument
├── ml/
│   ├── requirements.txt         zależności (przypięte wersje)
│   ├── run_all.sh               cały pipeline jedną komendą
│   ├── config.py                klasy, ścieżki, hiperparametry danych
│   ├── prepare_data.py          lista zdjęć + podział train/val/test
│   ├── data.py                  Dataset, augmentacja, DataLoader
│   ├── models.py                baseline CNN, MobileNetV3, zamrażanie warstw
│   ├── train.py                 pętla treningowa, etapy, checkpointy
│   ├── evaluate.py              metryki, wykresy, raport
│   ├── export_coreml.py         konwersja + weryfikacja Core ML
│   ├── data/                    zbiory danych (pobierane, niecommitowane)
│   └── outputs/                 checkpointy, logi, wykresy, RAPORT.md, model Core ML
└── ios/
    ├── FlowerCam.xcodeproj
    └── FlowerCam/               kod Swift + model + ikona
```

---

## 8. Jak uruchomić

**Pipeline ML** (Mac z Apple Silicon, ok. 25 min):
```bash
cd ml
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# dane (jednorazowo):
curl -L https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz | tar xz -C data
curl -L -o /tmp/caltech.zip "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"
unzip -q /tmp/caltech.zip -d /tmp && tar xzf /tmp/caltech-101/101_ObjectCategories.tar.gz -C data
./run_all.sh
```

**Aplikacja:**
1. Otwórz `ios/FlowerCam.xcodeproj` w Xcode.
2. Podłącz iPhone'a, wybierz go jako cel, naciśnij **Run** (⌘R).
3. Przy pierwszym uruchomieniu na telefonie: *Ustawienia → Ogólne → VPN i urządzenia →
   zaufaj certyfikatowi deweloperskiemu*, oraz włączony *Tryb dewelopera*.
4. Przy darmowym koncie Apple aplikacja wygasa po 7 dniach – wystarczy ponownie Run.

---

## 9. Wyniki (zbiór testowy, 665 zdjęć, nieużywany w treningu)

| Model | Accuracy | Macro F1 | Parametry |
|---|---|---|---|
| CNN od zera (baseline) | 82,9% | 0,823 | 2,36 M |
| MobileNetV3 – zamrożony backbone (etap 1) | 94,3% | 0,942 | 4,21 M |
| **MobileNetV3 – fine-tuning (etap 2)** | **95,5%** | **0,953** | 4,21 M |

- Transfer learning daje **+11,4 p.p.** względem sieci od zera, fine-tuning dokłada kolejne **+1,2 p.p.**
  Sieć od zera po 30 epokach (19 min) osiąga na walidacji 87,2% – mniej niż MobileNet
  po pierwszej epoce (20 s, 91,1%).
- Najczęstsza pomyłka: **tulipan ↔ róża** (8 + 4 przypadki) – podobny kształt i kolor pąków.
- Klasa „to nie kwiat”: 114/114 poprawnie, 0 fałszywych alarmów.
- **Core ML:** ta sama klasa top-1 co PyTorch dla **665/665** zdjęć (FP16 vs FP32,
  maks. różnica prawdopodobieństw 0,017), model **8,1 MB**, inferencja na Macu 0,42 ms (Neural Engine).
- **Próg pewności 70%** (domyślny w aplikacji): wynik pokazywany dla 84% zdjęć kwiatów,
  z trafnością 99,4%; pozostałe dostają komunikat „Nie jestem pewien”.

Wykresy: `ml/outputs/krzywe_uczenia.png`, `ml/outputs/macierz_pomylek_*.png`.
Pełny raport: `ml/outputs/RAPORT.md`.

---

## 10. Kamienie milowe i metryki (do specyfikacji)

| # | Kamień milowy | Metryka i kryterium | Walidacja | Status |
|---|---|---|---|---|
| 1 | Dane + model bazowy | podział 70/15/15, baseline wytrenowany | stratyfikowany zbiór testowy | ✅ 82,9% |
| 2 | Model z transfer learningiem | accuracy ≥ 90% i macro F1 ≥ 0,90 na teście | `evaluate.py`, macierz pomyłek | ✅ 95,5% / 0,953 |
| 3 | Eksport do Core ML | zgodność top-1 z PyTorch ≥ 99%, rozmiar ≤ 20 MB | `export_coreml.py` na całym teście | ✅ 100% / 8,1 MB |
| 4 | Aplikacja na iPhonie w czasie rzeczywistym | ≥ 15 FPS, inferencja < 30 ms/klatkę, działanie offline | licznik FPS/ms w aplikacji, test na zdjęciach kwiatów pokazanych kamerze, tryb samolotowy | ✅ ok. 30 FPS (iPhone 17) |

---

## 11. Ograniczenia i możliwe rozszerzenia

**Ograniczenia**
- Tylko 5 gatunków – każdy inny kwiat model przypisze do najbliższego z nich albo do „nie wiem”.
- Klasa „inne” pochodzi z Caltech-101 (czyste zdjęcia obiektów), więc wynik 100% na teście jest
  optymistyczny – realne klatki z kamery (ściana, biurko, dłoń) są trudniejsze. Drugą linią obrony
  są próg pewności i wygładzanie w czasie.
- Przesunięcie domeny: zdjęcia z internetu vs obraz z kamery (inne oświetlenie, rozmycie, kadr);
  częściowo kompensowane augmentacją.

**Rozszerzenia**
- Więcej gatunków (np. Oxford 102 Flowers).
- Rozpoznawanie zdjęć z galerii (PhotosPicker).
- Wizualizacja Grad-CAM – na co patrzy sieć.
- Kwantyzacja wag do INT8 (`coremltools.optimize`) – mniejszy model, porównanie dokładności.
