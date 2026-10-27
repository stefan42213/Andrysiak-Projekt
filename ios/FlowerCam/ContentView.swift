import SwiftUI

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @StateObject private var recognizer = FlowerRecognizer()
    /// Minimalna pewność, przy której pokazujemy nazwę kwiatu.
    @State private var threshold: Double = 0.7
    @State private var showDetails = false

    var body: some View {
        ZStack {
            CameraPreview(session: camera.session)
                .ignoresSafeArea()

            Viewfinder()

            VStack(spacing: 12) {
                StatsBar(fps: recognizer.fps, inferenceMs: recognizer.inferenceMs)
                Spacer()
                ResultCard(predictions: recognizer.predictions,
                           threshold: $threshold,
                           showDetails: $showDetails)
            }
            .padding(16)

            if camera.status == .denied || camera.status == .failed {
                CameraUnavailableView(status: camera.status)
            }
        }
        .onAppear {
            camera.onFrame = { [recognizer] buffer in recognizer.classify(buffer) }
            camera.start()
        }
        .onDisappear { camera.stop() }
    }
}

/// Ramka na środku ekranu: wskazówka, gdzie trzymać kwiat. Model widzi kwadrat ze środka
/// klatki (centerCrop), który na ekranie jest mniej więcej tej szerokości.
private struct Viewfinder: View {
    var body: some View {
        GeometryReader { geo in
            let side = geo.size.width
            RoundedRectangle(cornerRadius: 24)
                .strokeBorder(.white.opacity(0.7), style: StrokeStyle(lineWidth: 2, dash: [10, 8]))
                .frame(width: side - 24, height: side - 24)
                .position(x: geo.size.width / 2, y: geo.size.height / 2)
        }
        .ignoresSafeArea()
        .allowsHitTesting(false)
    }
}

private struct StatsBar: View {
    let fps: Double
    let inferenceMs: Double

    var body: some View {
        HStack {
            Label(String(format: "%.0f FPS", fps), systemImage: "speedometer")
            Spacer()
            Label(String(format: "%.1f ms / klatkę", inferenceMs), systemImage: "cpu")
        }
        .font(.footnote.monospacedDigit().weight(.medium))
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial, in: Capsule())
    }
}

private struct ResultCard: View {
    let predictions: [Prediction]
    @Binding var threshold: Double
    @Binding var showDetails: Bool

    private var best: Prediction? { predictions.first }

    private var headline: (title: String, subtitle: String, color: Color) {
        guard let best else { return ("Uruchamianie…", "Skieruj kamerę na kwiat", .gray) }
        if best.isOther {
            return ("To nie kwiat", "Skieruj kamerę na kwiat w ramce", .orange)
        }
        if Double(best.confidence) < threshold {
            return ("Nie jestem pewien", "Podejdź bliżej lub popraw oświetlenie", .yellow)
        }
        return (best.name, String(format: "Pewność %.0f%%", best.confidence * 100), .green)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Circle().fill(headline.color).frame(width: 12, height: 12)
                Text(headline.title).font(.title.bold())
                Spacer()
                Button {
                    withAnimation(.snappy) { showDetails.toggle() }
                } label: {
                    Image(systemName: showDetails ? "chevron.down.circle.fill" : "chevron.up.circle.fill")
                        .font(.title2)
                }
                .tint(.white)
            }
            Text(headline.subtitle)
                .font(.subheadline)
                .foregroundStyle(.secondary)

            if showDetails {
                ForEach(predictions.prefix(3)) { prediction in
                    ProbabilityRow(prediction: prediction)
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text(String(format: "Próg pewności: %.0f%%", threshold * 100))
                        .font(.footnote.monospacedDigit())
                    Slider(value: $threshold, in: 0.4...0.95, step: 0.05)
                }
                .padding(.top, 4)
            }
        }
        .padding(18)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 24))
        .animation(.easeOut(duration: 0.2), value: headline.title)
    }
}

private struct ProbabilityRow: View {
    let prediction: Prediction

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(prediction.name)
                Spacer()
                Text(String(format: "%.1f%%", prediction.confidence * 100)).monospacedDigit()
            }
            .font(.callout)
            ProgressView(value: Double(min(max(prediction.confidence, 0), 1)))
                .tint(prediction.isOther ? .orange : .green)
        }
    }
}

private struct CameraUnavailableView: View {
    let status: CameraManager.Status

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "camera.fill").font(.system(size: 48))
            Text(status == .denied ? "Brak dostępu do kamery" : "Kamera niedostępna")
                .font(.title2.bold())
            if status == .denied {
                Text("Włącz dostęp w Ustawienia → FlowerCam → Aparat.")
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
                Button("Otwórz Ustawienia") {
                    if let url = URL(string: UIApplication.openSettingsURLString) {
                        UIApplication.shared.open(url)
                    }
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.black)
    }
}

#Preview {
    ContentView()
}
