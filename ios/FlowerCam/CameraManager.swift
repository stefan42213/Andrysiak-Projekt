import AVFoundation

/// Zarządza sesją kamery i przekazuje kolejne klatki (CVPixelBuffer) do klasyfikatora.
final class CameraManager: NSObject, ObservableObject {
    enum Status { case unknown, running, denied, failed }

    @Published private(set) var status: Status = .unknown
    let session = AVCaptureSession()

    /// Wywoływane na kolejce wideo dla każdej klatki, która nie została odrzucona.
    var onFrame: ((CVPixelBuffer) -> Void)?

    private let sessionQueue = DispatchQueue(label: "camera.session")
    private let videoQueue = DispatchQueue(label: "camera.frames", qos: .userInitiated)
    private var isConfigured = false

    func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            startSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                if granted { self?.startSession() } else { self?.setStatus(.denied) }
            }
        default:
            setStatus(.denied)
        }
    }

    func stop() {
        sessionQueue.async { [session] in
            if session.isRunning { session.stopRunning() }
        }
    }

    private func startSession() {
        sessionQueue.async { [self] in
            if !isConfigured {
                guard configure() else { return setStatus(.failed) }
                isConfigured = true
            }
            if !session.isRunning { session.startRunning() }
            setStatus(.running)
        }
    }

    private func configure() -> Bool {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        // 720p wystarcza: model i tak dostaje wycinek 224x224.
        session.sessionPreset = .hd1280x720
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else { return false }
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        // Gdy model nie nadąża, stare klatki są odrzucane zamiast tworzyć kolejkę opóźnień.
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: videoQueue)
        guard session.canAddOutput(output) else { return false }
        session.addOutput(output)
        return true
    }

    private func setStatus(_ status: Status) {
        DispatchQueue.main.async { self.status = status }
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        onFrame?(pixelBuffer)
    }
}
