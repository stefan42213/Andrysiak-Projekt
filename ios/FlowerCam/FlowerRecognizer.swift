import CoreML
import QuartzCore
import Vision

struct Prediction: Identifiable, Equatable {
    let id: String          // klucz klasy z modelu, np. "roses"
    let confidence: Float

    var name: String { FlowerRecognizer.namesPL[id] ?? id }
    var isOther: Bool { id == FlowerRecognizer.otherClass }
}

/// Uruchamia model Core ML na klatkach z kamery i wygładza wyniki w czasie.
///
/// Przepływ: CVPixelBuffer → Vision (obrót + centerCrop 224x224) → Core ML (Neural Engine)
/// → prawdopodobieństwa 6 klas → średnia wykładnicza z poprzednich klatek → UI.
final class FlowerRecognizer: ObservableObject {
    static let otherClass = "other"
    static let namesPL: [String: String] = [
        "daisy": "Stokrotka",
        "dandelion": "Mniszek lekarski",
        "roses": "Róża",
        "sunflowers": "Słonecznik",
        "tulips": "Tulipan",
        "other": "To nie kwiat",
    ]

    @Published private(set) var predictions: [Prediction] = []
    @Published private(set) var inferenceMs: Double = 0
    @Published private(set) var fps: Double = 0

    private let request: VNCoreMLRequest

    // Stan poniżej jest używany wyłącznie na (szeregowej) kolejce wideo.
    /// Waga nowej klatki w średniej wykładniczej; mniejsza = stabilniej, ale wolniejsza reakcja.
    private let smoothing: Float = 0.35
    private var smoothed: [String: Float] = [:]
    private var avgInferenceMs: Double = 0
    private var avgFrameInterval: Double = 0
    private var lastFrameTime: CFTimeInterval = 0

    init() {
        let config = MLModelConfiguration()
        config.computeUnits = .all   // CPU + GPU + Neural Engine, wybór należy do Core ML
        do {
            // Klasa FlowerClassifier jest generowana przez Xcode z FlowerClassifier.mlpackage.
            let model = try VNCoreMLModel(for: FlowerClassifier(configuration: config).model)
            request = VNCoreMLRequest(model: model)
            // Tak samo jak przy ewaluacji w Pythonie: krótszy bok → 224, wycięcie środka.
            request.imageCropAndScaleOption = .centerCrop
        } catch {
            fatalError("Nie udało się wczytać modelu Core ML: \(error)")
        }
    }

    /// Klasyfikuje jedną klatkę. Wywoływane synchronicznie na kolejce wideo.
    func classify(_ pixelBuffer: CVPixelBuffer) {
        let start = CACurrentMediaTime()
        // Sensor kamery jest poziomy; .right obraca klatkę do orientacji pionowej telefonu.
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right)
        guard (try? handler.perform([request])) != nil,
              let observations = request.results as? [VNClassificationObservation] else { return }
        let now = CACurrentMediaTime()

        for observation in observations {
            let previous = smoothed[observation.identifier] ?? observation.confidence
            smoothed[observation.identifier] = previous + smoothing * (observation.confidence - previous)
        }
        avgInferenceMs = movingAverage(avgInferenceMs, (now - start) * 1000)
        if lastFrameTime > 0 {
            avgFrameInterval = movingAverage(avgFrameInterval, now - lastFrameTime)
        }
        lastFrameTime = now

        let ranked = smoothed
            .map { Prediction(id: $0.key, confidence: $0.value) }
            .sorted { $0.confidence > $1.confidence }
        let inferenceMs = avgInferenceMs
        let fps = avgFrameInterval > 0 ? 1 / avgFrameInterval : 0
        DispatchQueue.main.async {
            self.predictions = ranked
            self.inferenceMs = inferenceMs
            self.fps = fps
        }
    }

    private func movingAverage(_ average: Double, _ value: Double) -> Double {
        average == 0 ? value : average * 0.9 + value * 0.1
    }
}
