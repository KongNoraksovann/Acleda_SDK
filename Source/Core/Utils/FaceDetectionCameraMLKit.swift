import AVFoundation
import MLKitFaceDetection
import MLKitVision
import UIKit
import os

@available(iOS 12.0, *)
public class FaceDetectionCameraMLKit: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let captureSession = AVCaptureSession()
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private let videoDataOutputQueue = DispatchQueue(label: "com.example.faceDetectionQueue", qos: .userInteractive)
    private var faceDetector: FaceDetector?
    private var currentState: FaceDetectionState = .WAITING
    private var stableFrameCount = 0
    private var inTargetDistanceFrameCount = 0
    private var lastCaptureTime: TimeInterval = 0
    private var qualityScore: Float = 0
    private var distanceStatus: FaceDistanceStatus = .UNKNOWN
    private var faceDistanceCm: Int = 0
    private var facePositionHistory: [CGRect] = []
    private var blinkDetected = false
    private var eyesClosed = false
    private var areEyesOpen = false
    private var isFocusing = false
    private var isFocused = false
    private var focusAttempts = 0
    private var focusStartTime: TimeInterval = 0
    private var lastProcessedTime: TimeInterval = 0
    private let maxHistory = 10
    private let isFrontCamera: Bool
    private let minFaceSize: Float
    private let isBlinkDetectionEnabled: Bool
    private var detectionMessage: String = "Position your face in the frame"
    private lazy var lastDirectionMessage: String = ""
    private let onStateChanged: (FaceDetectionState, String, Float, FaceDistanceStatus, Int, Bool, UIBezierPath?) -> Void
    private let onImageCaptured: (UIImage) -> Void
    private let onError: (Error) -> Void
    private var lastSampleBuffer: CMSampleBuffer?
    private var videoDevice: AVCaptureDevice?

    private var currentImageWidth: CGFloat = 0
    private var currentImageHeight: CGFloat = 0
    
    private let minTargetDistanceCm = 35
    private let maxTargetDistanceCm = 60
    
    private let targetDistanceStableFrames = 10
    private let requiredStableFrames = 8
    private let minIdealFaceSize: Float = 0.15
    private let maxIdealFaceSize: Float = 0.40
    private let optimalFaceSize: Float = 0.25
    private let averageFaceWidthCm = 15.0
    private let focalLengthFactor = 650.0
    private let maxFocusWaitTime: TimeInterval = 3.0
    private let maxFocusAttempts = 3
    private let frameThrottleInterval: TimeInterval = 0.1
    private let centerTolerance: CGFloat = 0.08
    private let maxHeadEulerY: Float = 12.0
    private let maxHeadEulerX: Float = 12.0
    private let maxHeadEulerZ: Float = 10.0

    init(
        previewView: UIView,
        isFrontCamera: Bool = true,
        minFaceSize: Float = 0.1,
        isBlinkDetectionEnabled: Bool = true,
        onStateChanged: @escaping (FaceDetectionState, String, Float, FaceDistanceStatus, Int, Bool, UIBezierPath?) -> Void,
        onImageCaptured: @escaping (UIImage) -> Void,
        onError: @escaping (Error) -> Void
    ) {
        self.isFrontCamera = isFrontCamera
        self.minFaceSize = minFaceSize
        self.isBlinkDetectionEnabled = isBlinkDetectionEnabled
        self.onStateChanged = onStateChanged
        self.onImageCaptured = onImageCaptured
        self.onError = onError
        super.init()

        checkCameraPermissions { [weak self] authorized in
            guard let self = self else { return }
            if authorized {
                self.setupCamera(previewView: previewView)
                self.setupFaceDetector()
            } else {
                let error = NSError(domain: "FaceDetectionCameraMLKit", code: -1, userInfo: [NSLocalizedDescriptionKey: "Camera access denied"])
                self.onError(error)
                os_log("Camera: Camera access denied", log: OSLog.default, type: .error)
            }
        }
    }

    deinit {
        captureSession.stopRunning()
        previewLayer?.removeFromSuperlayer()
        NotificationCenter.default.removeObserver(self)
        lastSampleBuffer = nil
        faceDetector = nil
        os_log("Camera: Capture session stopped and deinitialized", log: OSLog.default, type: .info)
    }

    internal func getCurrentImageWidth() -> CGFloat? {
        return currentImageWidth > 0 ? currentImageWidth : nil
    }

    private func checkCameraPermissions(completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    completion(granted)
                }
            }
        case .denied, .restricted:
            completion(false)
        @unknown default:
            completion(false)
        }
    }

    private func getBestCamera(position: AVCaptureDevice.Position) -> AVCaptureDevice? {
        let deviceTypes: [AVCaptureDevice.DeviceType]
        
        if position == .front {
            if #available(iOS 13.0, *) {
                deviceTypes = [.builtInTrueDepthCamera, .builtInWideAngleCamera]
            } else {
                deviceTypes = [.builtInWideAngleCamera]
            }
        } else {
            if #available(iOS 13.0, *) {
                deviceTypes = [.builtInUltraWideCamera, .builtInWideAngleCamera, .builtInDualCamera]
            } else {
                deviceTypes = [.builtInWideAngleCamera, .builtInDualCamera]
            }
        }
        
        let discoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: deviceTypes,
            mediaType: .video,
            position: position
        )
        
        return discoverySession.devices.first
    }
    
    private func setupCamera(previewView: UIView) {
        captureSession.sessionPreset = .medium

        let position: AVCaptureDevice.Position = isFrontCamera ? .front : .back
        
        guard let videoDevice = getBestCamera(position: position) else {
            let error = NSError(domain: "FaceDetectionCameraMLKit", code: -1, userInfo: [NSLocalizedDescriptionKey: "No suitable camera found for \(isFrontCamera ? "front" : "back") position"])
            onError(error)
            os_log("Camera: No suitable camera found for %@", log: OSLog.default, type: .error, isFrontCamera ? "front" : "back")
            return
        }
        self.videoDevice = videoDevice

        do {
            let videoInput = try AVCaptureDeviceInput(device: videoDevice)
            guard captureSession.canAddInput(videoInput) else {
                let error = NSError(domain: "FaceDetectionCameraMLKit", code: -2, userInfo: [NSLocalizedDescriptionKey: "Failed to add camera input"])
                onError(error)
                os_log("Camera: Failed to add camera input", log: OSLog.default, type: .error)
                return
            }
            captureSession.addInput(videoInput)
            os_log("Camera: Added input for %@ camera", log: OSLog.default, type: .info, videoDevice.localizedName)

            let supportedFocusModes = [AVCaptureDevice.FocusMode.autoFocus, .continuousAutoFocus, .locked]
                .filter { videoDevice.isFocusModeSupported($0) }
            os_log("Camera: Supported focus modes: %@", log: OSLog.default, type: .info, supportedFocusModes.map { "\($0.rawValue)" }.description)
            
            try videoDevice.lockForConfiguration()
            if videoDevice.isFocusModeSupported(.continuousAutoFocus) {
                videoDevice.focusMode = .continuousAutoFocus
                os_log("Camera: Initial focus mode set to continuousAutoFocus", log: OSLog.default, type: .info)
            } else if videoDevice.isFocusModeSupported(.locked) {
                videoDevice.focusMode = .locked
                os_log("Camera: Initial focus mode set to locked", log: OSLog.default, type: .info)
            } else {
                os_log("Camera: No supported focus modes available", log: OSLog.default, type: .info)
            }
            videoDevice.unlockForConfiguration()

            guard captureSession.canAddOutput(videoDataOutput) else {
                let error = NSError(domain: "FaceDetectionCameraMLKit", code: -3, userInfo: [NSLocalizedDescriptionKey: "Failed to add video data output"])
                onError(error)
                os_log("Camera: Failed to add video data output", log: OSLog.default, type: .error)
                return
            }
            captureSession.addOutput(videoDataOutput)
            videoDataOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
            videoDataOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
            videoDataOutput.alwaysDiscardsLateVideoFrames = true
            os_log("Camera: Added video data output", log: OSLog.default, type: .info)

            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.previewLayer = AVCaptureVideoPreviewLayer(session: self.captureSession)
                self.previewLayer?.videoGravity = .resizeAspectFill
                self.previewLayer?.frame = previewView.bounds
                previewView.layer.insertSublayer(self.previewLayer!, at: 0)
                os_log("Camera: Preview layer added to previewView with bounds: %@", log: OSLog.default, type: .info, NSCoder.string(for: previewView.bounds))
                previewView.addObserver(self, forKeyPath: #keyPath(UIView.bounds), options: [.new], context: nil)
            }

            NotificationCenter.default.addObserver(
                self,
                selector: #selector(handleSessionInterruptionEnded),
                name: .AVCaptureSessionInterruptionEnded,
                object: captureSession
            )

            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                guard let self = self else { return }
                self.captureSession.startRunning()
                os_log("Camera: Capture session started with %@ camera", log: OSLog.default, type: .info, videoDevice.localizedName)
            }
        } catch {
            onError(error)
            os_log("Camera: Setup failed: %@", log: OSLog.default, type: .error, error.localizedDescription)
        }
    }

    private func setupFaceDetector() {
        let options = FaceDetectorOptions()
        options.performanceMode = .fast
        options.landmarkMode = .all
        options.classificationMode = .all
        options.minFaceSize = CGFloat(minFaceSize)
        faceDetector = FaceDetector.faceDetector(options: options)
        os_log("Camera: FaceDetector initialized with minFaceSize: %f", log: OSLog.default, type: .info, minFaceSize)
    }

    public override func observeValue(forKeyPath keyPath: String?, of object: Any?, change: [NSKeyValueChangeKey: Any]?, context: UnsafeMutableRawPointer?) {
        if keyPath == #keyPath(UIView.bounds), let view = object as? UIView {
            DispatchQueue.main.async { [weak self] in
                self?.previewLayer?.frame = view.bounds
                os_log("Camera: Updated preview layer frame to: %@", log: OSLog.default, type: .info, NSCoder.string(for: view.bounds))
            }
        } else {
            super.observeValue(forKeyPath: keyPath, of: object, change: change, context: context)
        }
    }

    @objc private func handleSessionInterruptionEnded(notification: Notification) {
        os_log("Camera: Capture session interruption ended", log: OSLog.default, type: .info)
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.captureSession.startRunning()
            os_log("Camera: Capture session restarted", log: OSLog.default, type: .info)
        }
    }

    func captureManually(completion: @escaping (UIImage) -> Void) {
        guard let sampleBuffer = lastSampleBuffer,
              (currentState == .DETECTING || currentState == .STABILIZING || currentState == .BLINK_CHECK),
              CACurrentMediaTime() - lastCaptureTime > 0.5 else {
            os_log("Camera: Manual capture skipped: Invalid state or cooldown", log: OSLog.default, type: .info)
            return
        }

        guard let image = CameraUtilities.toUIImage(sampleBuffer, isFrontCamera: isFrontCamera, normalizeOrientation: true) else {
            os_log("Camera: Manual capture failed: Could not convert sample buffer to UIImage", log: OSLog.default, type: .error)
            return
        }

        lastCaptureTime = CACurrentMediaTime()
        currentState = .CAPTURING
        onStateChanged(currentState, "Capturing...", qualityScore, distanceStatus, faceDistanceCm, areEyesOpen, nil)
        lastSampleBuffer = nil

        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.currentState = .SUCCESS
            self.onStateChanged(self.currentState, "Success!", self.qualityScore, self.distanceStatus, self.faceDistanceCm, self.areEyesOpen, nil)
            let finalImage = self.flipImageIfNeeded(image)
            completion(finalImage)
            os_log("Camera: Manual capture triggered successfully", log: OSLog.default, type: .info)
        }
    }

    private func flipImageIfNeeded(_ image: UIImage) -> UIImage {
        let orientationFixedImage = normalizeImageOrientation(image)
        guard isFrontCamera else { return orientationFixedImage }
        guard let cgImage = orientationFixedImage.cgImage else { return orientationFixedImage }
        let flippedImage = UIImage(cgImage: cgImage, scale: orientationFixedImage.scale, orientation: .leftMirrored)
        return flippedImage
    }

    private func normalizeImageOrientation(_ image: UIImage) -> UIImage {
        if image.imageOrientation == .up {
            return image
        }
        
        UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
        image.draw(in: CGRect(origin: .zero, size: image.size))
        let normalizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return normalizedImage ?? image
    }

    private func triggerFocus(at point: CGPoint, completion: @escaping (Bool) -> Void) {
        guard let device = videoDevice else {
            os_log("Camera: No video device available for focus", log: OSLog.default, type: .info)
            completion(false)
            return
        }
        
        guard device.isFocusPointOfInterestSupported else {
            os_log("Camera: Focus point of interest not supported", log: OSLog.default, type: .info)
            completion(false)
            return
        }

        do {
            try device.lockForConfiguration()
            
            let supportedFocusModes: [AVCaptureDevice.FocusMode] = [.autoFocus, .continuousAutoFocus, .locked]
                .filter { device.isFocusModeSupported($0) }
            
            guard !supportedFocusModes.isEmpty else {
                os_log("Camera: No supported focus modes available", log: OSLog.default, type: .info)
                device.unlockForConfiguration()
                completion(false)
                return
            }
            
            let focusMode: AVCaptureDevice.FocusMode
            if supportedFocusModes.contains(.autoFocus) {
                focusMode = .autoFocus
            } else if supportedFocusModes.contains(.continuousAutoFocus) {
                focusMode = .continuousAutoFocus
            } else {
                focusMode = .locked
            }
            
            device.focusPointOfInterest = point
            device.focusMode = focusMode
            device.unlockForConfiguration()
            
            os_log("Camera: Set focus mode to %d at point (%f, %f)", log: OSLog.default, type: .info, focusMode.rawValue, point.x, point.y)
            
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                completion(true)
            }
        } catch {
            os_log("Camera: Failed to set focus: %@", log: OSLog.default, type: .error, error.localizedDescription)
            device.unlockForConfiguration()
            completion(false)
        }
    }

    private func cancelFocus() {
        guard let device = videoDevice else {
            os_log("Camera: No video device available to cancel focus", log: OSLog.default, type: .info)
            return
        }
        
        do {
            try device.lockForConfiguration()
            if device.isFocusModeSupported(.continuousAutoFocus) {
                device.focusMode = .continuousAutoFocus
                os_log("Camera: Focus cancelled to continuousAutoFocus", log: OSLog.default, type: .info)
            } else if device.isFocusModeSupported(.locked) {
                device.focusMode = .locked
                os_log("Camera: Focus cancelled to locked", log: OSLog.default, type: .info)
            } else {
                os_log("Camera: No supported focus modes to cancel focus", log: OSLog.default, type: .info)
            }
            device.unlockForConfiguration()
        } catch {
            os_log("Camera: Failed to cancel focus: %@", log: OSLog.default, type: .error, error.localizedDescription)
        }
    }

    public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let currentTime = CACurrentMediaTime()
        guard currentTime - lastProcessedTime > frameThrottleInterval else { return }
        lastProcessedTime = currentTime

        lastSampleBuffer = sampleBuffer
        guard currentState != .CAPTURING && currentState != .SUCCESS,
              CACurrentMediaTime() - lastCaptureTime > 0.5 else {
            return
        }

        guard let image = CameraUtilities.toUIImage(sampleBuffer, isFrontCamera: isFrontCamera, normalizeOrientation: true) else {
            os_log("Camera: Failed to convert sample buffer to UIImage", log: OSLog.default, type: .error)
            lastSampleBuffer = nil
            return
        }

        currentImageWidth = image.size.width
        currentImageHeight = image.size.height
        
        let visionImage = VisionImage(image: image)
        visionImage.orientation = imageOrientation(from: connection)

        faceDetector?.process(visionImage) { [weak self] faces, error in
            guard let self = self else { return }

            if let error = error {
                self.onError(error)
                os_log("Camera: Face detection failed: %@", log: OSLog.default, type: .error, error.localizedDescription)
                self.lastSampleBuffer = nil
                return
            }

            self.handleFaceDetection(faces ?? [], image: image, imageWidth: self.currentImageWidth, imageHeight: self.currentImageHeight)
            self.lastSampleBuffer = nil
        }
    }

    private func imageOrientation(from connection: AVCaptureConnection) -> UIImage.Orientation {
        let deviceOrientation = UIDevice.current.orientation
        let isMirrored = isFrontCamera
        
        switch deviceOrientation {
        case .portrait:
            return isMirrored ? .leftMirrored : .up
        case .portraitUpsideDown:
            return isMirrored ? .rightMirrored : .down
        case .landscapeLeft:
            return isMirrored ? .downMirrored : .right
        case .landscapeRight:
            return isMirrored ? .upMirrored : .left
        case .faceUp, .faceDown:
            return isMirrored ? .leftMirrored : .right
        default:
            return isMirrored ? .leftMirrored : .right
        }
    }

    private func estimateDistanceCm(faceWidthPixels: CGFloat) -> Int {
        return Int((averageFaceWidthCm * focalLengthFactor / Double(faceWidthPixels)).rounded())
    }

    private func isAtTargetDistance(_ distanceCm: Int) -> Bool {
        return distanceCm >= minTargetDistanceCm && distanceCm <= maxTargetDistanceCm
    }

    private func analyzeHeadAngles(headEulerY: Float, headEulerX: Float, headEulerZ: Float) -> (isGoodAngle: Bool, message: String?) {
        var messages: [String] = []
        
        if abs(headEulerY) > maxHeadEulerY {
            if headEulerY > maxHeadEulerY {
                messages.append("Turn your head left")
            } else {
                messages.append("Turn your head right")
            }
        }
        
        if abs(headEulerX) > maxHeadEulerX {
            if headEulerX > maxHeadEulerX {
                messages.append("Lower your head")
            } else {
                messages.append("Raise your head")
            }
        }
        
        if abs(headEulerZ) > maxHeadEulerZ {
            if headEulerZ > maxHeadEulerZ {
                messages.append("Straighten your head (tilted left)")
            } else {
                messages.append("Straighten your head (tilted right)")
            }
        }
        
        let isGoodAngle = messages.isEmpty
        let message = messages.isEmpty ? nil : messages.joined(separator: " and ")
        
        return (isGoodAngle, message)
    }

    private func createFaceAndEarFramePath(boundingBox: CGRect) -> UIBezierPath {
        let path = UIBezierPath()
        let faceWidth = boundingBox.width
        let faceHeight = boundingBox.height * 1.3
        let faceX = boundingBox.minX
        let faceY = boundingBox.minY - (faceHeight - boundingBox.height) / 2
        let faceRect = CGRect(x: faceX, y: faceY, width: faceWidth, height: faceHeight)

        path.append(UIBezierPath(ovalIn: faceRect))

        let earWidth = faceWidth * 0.25
        let earHeight = faceHeight * 0.35
        let earY = faceY + faceHeight * 0.15

        let leftEarRect = CGRect(
            x: faceX - earWidth * 0.6,
            y: earY,
            width: earWidth,
            height: earHeight
        )
        let leftEarPath = UIBezierPath()
        leftEarPath.move(to: CGPoint(x: leftEarRect.maxX, y: leftEarRect.minY + earHeight * 0.2))
        leftEarPath.addArc(
            withCenter: CGPoint(x: leftEarRect.midX, y: leftEarRect.midY),
            radius: earWidth * 0.5,
            startAngle: .pi / 2,
            endAngle: .pi * 1.5,
            clockwise: true
        )
        leftEarPath.addLine(to: CGPoint(x: leftEarRect.maxX, y: leftEarRect.minY + earHeight * 0.2))
        leftEarPath.close()
        path.append(leftEarPath)

        let rightEarRect = CGRect(
            x: faceX + faceWidth - earWidth * 0.4,
            y: earY,
            width: earWidth,
            height: earHeight
        )
        let rightEarPath = UIBezierPath()
        rightEarPath.move(to: CGPoint(x: rightEarRect.minX, y: rightEarRect.minY + earHeight * 0.2))
        rightEarPath.addArc(
            withCenter: CGPoint(x: rightEarRect.midX, y: rightEarRect.midY),
            radius: earWidth * 0.5,
            startAngle: -.pi / 2,
            endAngle: .pi / 2,
            clockwise: true
        )
        rightEarPath.addLine(to: CGPoint(x: rightEarRect.minX, y: rightEarRect.minY + earHeight * 0.2))
        rightEarPath.close()
        path.append(rightEarPath)

        return path
    }

    private func handleFaceDetection(_ faces: [Face], image: UIImage, imageWidth: CGFloat, imageHeight: CGFloat) {
        var faceFramePath: UIBezierPath? = nil

        guard !faces.isEmpty else {
            if self.currentState != .WAITING {
                self.currentState = .WAITING
                self.stableFrameCount = 0
                self.inTargetDistanceFrameCount = 0
                self.qualityScore = 0
                self.distanceStatus = .UNKNOWN
                self.faceDistanceCm = 0
                self.facePositionHistory.removeAll()
                self.areEyesOpen = false
                if isBlinkDetectionEnabled {
                    self.blinkDetected = false
                    self.eyesClosed = false
                }
                self.isFocused = false
                self.isFocusing = false
                self.focusAttempts = 0
                self.focusStartTime = 0
                self.cancelFocus()
                self.detectionMessage = "Position your face in the frame"
                self.onStateChanged(self.currentState, self.detectionMessage, self.qualityScore, self.distanceStatus, self.faceDistanceCm, self.areEyesOpen, faceFramePath)
            }
            return
        }

        let face = faces[0]
        let boundingBox = face.frame
        
        let headEulerY = Float(face.headEulerAngleY)
        let headEulerX = Float(face.headEulerAngleX)
        let headEulerZ = Float(face.headEulerAngleZ)
        let (isGoodHeadAngle, headAngleMessage) = analyzeHeadAngles(headEulerY: headEulerY, headEulerX: headEulerX, headEulerZ: headEulerZ)
        let angleQuality = max(0.0, 1.0 - (abs(headEulerY) + abs(headEulerX) + abs(headEulerZ)) / 45.0)

        facePositionHistory.append(boundingBox)
        if facePositionHistory.count > maxHistory {
            facePositionHistory.removeFirst()
        }

        faceFramePath = createFaceAndEarFramePath(boundingBox: boundingBox)

        let movementScore: Float
        if facePositionHistory.count > 1 {
            let movement = CameraUtilities.calculateMovement(facePositionHistory)
            movementScore = max(0.0, 1.0 - movement / 50.0)
        } else {
            movementScore = 0.5
        }

        let faceSize = Float(boundingBox.width * boundingBox.height) / Float(imageWidth * imageHeight)
        faceDistanceCm = estimateDistanceCm(faceWidthPixels: boundingBox.width)
        let atTargetDistance = isAtTargetDistance(faceDistanceCm)
        distanceStatus = faceDistanceCm < minTargetDistanceCm ? .TOO_CLOSE :
                         faceDistanceCm > maxTargetDistanceCm ? .TOO_FAR : .OPTIMAL

        let sizeScore: Float
        if faceSize < minIdealFaceSize {
            sizeScore = (faceSize / minIdealFaceSize) * 0.7
        } else if faceSize > maxIdealFaceSize {
            sizeScore = max(0.0, 1.0 - (faceSize - maxIdealFaceSize) / 0.2) * 0.7
        } else {
            let distanceFromOptimal = abs(faceSize - optimalFaceSize)
            let normalizedDistance = distanceFromOptimal / (maxIdealFaceSize - minIdealFaceSize)
            sizeScore = 0.7 + 0.3 * (1.0 - pow(normalizedDistance, 2))
        }

        qualityScore = (angleQuality * 0.4) + (movementScore * 0.3) + (sizeScore * 0.3)

        let (isCentered, horizontalOffset, verticalOffset, positionMessage) = assessFacePosition(
            face: face,
            imageWidth: imageWidth,
            imageHeight: imageHeight
        )
        
        let faceCenter = boundingBox.midX
        
        os_log("Enhanced positioning - isCentered: %@, H-offset: %f, V-offset: %f", log: OSLog.default, type: .debug, isCentered ? "YES" : "NO", horizontalOffset, verticalOffset)
        os_log("Position message: %@", log: OSLog.default, type: .debug, positionMessage)

        let leftEyeOpen = face.leftEyeOpenProbability ?? 1.0
        let rightEyeOpen = face.rightEyeOpenProbability ?? 1.0
        let isBlink = leftEyeOpen < 0.4 && rightEyeOpen < 0.4
        let isEyesOpen = leftEyeOpen > 0.7 && rightEyeOpen > 0.7
        areEyesOpen = isEyesOpen

        if atTargetDistance && movementScore > 0.7 {
            inTargetDistanceFrameCount += 1
        } else {
            inTargetDistanceFrameCount = max(0, inTargetDistanceFrameCount - 1)
        }

        let isGoodPosition = isGoodHeadAngle &&
                             isCentered &&
                             atTargetDistance &&
                             qualityScore > 0.6

        handleStateWithPositioningData(
            face: face,
            isGoodPosition: isGoodPosition,
            isCentered: isCentered,
            isGoodHeadAngle: isGoodHeadAngle,
            headAngleMessage: headAngleMessage,
            isEyesOpen: isEyesOpen,
            atTargetDistance: atTargetDistance,
            positionMessage: positionMessage,
            faceCenter: faceCenter,
            faceFramePath: faceFramePath
        )
    }

    private func assessFacePosition(face: Face, imageWidth: CGFloat, imageHeight: CGFloat) -> (isCentered: Bool, horizontalOffset: CGFloat, verticalOffset: CGFloat, message: String) {
        var horizontalOffset: CGFloat = 0
        var verticalOffset: CGFloat = 0
        var message = "Center your face"
        
        let boundingBoxCenterX = face.frame.midX
        let boundingBoxCenterY = face.frame.midY
        
        horizontalOffset = (boundingBoxCenterX - imageWidth / 2) / imageWidth
        verticalOffset = (boundingBoxCenterY - imageHeight / 2) / imageHeight
        
        if let noseLandmark = face.landmark(ofType: .noseBase) {
            let noseTip = noseLandmark.position
            horizontalOffset = (noseTip.x - imageWidth / 2) / imageWidth
            verticalOffset = (noseTip.y - imageHeight / 2) / imageHeight
            
            let leftEyePosition = face.landmark(ofType: .leftEye)?.position
            let rightEyePosition = face.landmark(ofType: .rightEye)?.position
            let mouthLeftPosition = face.landmark(ofType: .mouthLeft)?.position
            let mouthRightPosition = face.landmark(ofType: .mouthRight)?.position
            
            if let leftEye = leftEyePosition,
               let rightEye = rightEyePosition,
               let mouthLeft = mouthLeftPosition,
               let mouthRight = mouthRightPosition {
                let eyesMidpointX = (leftEye.x + rightEye.x) / 2
                let eyesMidpointY = (leftEye.y + rightEye.y) / 2
                let mouthMidpointX = (mouthLeft.x + mouthRight.x) / 2
                let mouthMidpointY = (mouthLeft.y + mouthRight.y) / 2
                
                let faceCenterX = (noseTip.x * 0.6 + eyesMidpointX * 0.3 + mouthMidpointX * 0.1)
                let faceCenterY = (noseTip.y * 0.3 + eyesMidpointY * 0.3 + mouthMidpointY * 0.4)
                
                horizontalOffset = (faceCenterX - imageWidth / 2) / imageWidth
                verticalOffset = (faceCenterY - imageHeight / 2) / imageHeight
            }
        }
        
        os_log("Face position assessment - H offset: %f, V offset: %f", log: OSLog.default, type: .debug, horizontalOffset, verticalOffset)
        
        let tightCenterTolerance: CGFloat = 0.04
        let centeringTolerance: CGFloat = 0.08
        
        let isHorizontalPerfect = abs(horizontalOffset) <= tightCenterTolerance
        let isVerticalPerfect = abs(verticalOffset) <= tightCenterTolerance
        let isCentered = isHorizontalPerfect && isVerticalPerfect
        
        if abs(horizontalOffset) > centeringTolerance || abs(verticalOffset) > centeringTolerance {
            var directions: [String] = []
            
            if horizontalOffset > centeringTolerance {
                directions.append("Move Up")
            } else if horizontalOffset < -centeringTolerance {
                directions.append("Move Down")
            }
            
            if verticalOffset > centeringTolerance {
                directions.append("Move Left")
            } else if verticalOffset < -centeringTolerance {
                directions.append("Move Right")
            }
            
            message = directions.joined(separator: " and ")
        } else if !isCentered {
            var directions: [String] = []
            
            if !isHorizontalPerfect {
                if horizontalOffset > 0 {
                    directions.append("Slightly Left")
                } else {
                    directions.append("Slightly Right")
                }
            }
            
            if !isVerticalPerfect {
                if verticalOffset > 0 {
                    directions.append("Slightly Up")
                } else {
                    directions.append("Slightly Down")
                }
            }
            
            message = directions.joined(separator: " and ")
        } else {
            message = "Perfect Position! Hold Still"
        }
        
        return (isCentered, horizontalOffset, verticalOffset, message)
    }

    private func handleStateWithPositioningData(
        face: Face,
        isGoodPosition: Bool,
        isCentered: Bool,
        isGoodHeadAngle: Bool,
        headAngleMessage: String?,
        isEyesOpen: Bool,
        atTargetDistance: Bool,
        positionMessage: String,
        faceCenter: CGFloat,
        faceFramePath: UIBezierPath?
    ) {
        onStateChanged(
            currentState,
            detectionMessage,
            qualityScore,
            distanceStatus,
            faceDistanceCm,
            areEyesOpen,
            faceFramePath
        )
        
        if atTargetDistance && inTargetDistanceFrameCount >= targetDistanceStableFrames &&
           (currentState == .WAITING || currentState == .DETECTING || currentState == .BLINK_CHECK) {
            if (!isBlinkDetectionEnabled || blinkDetected) && isEyesOpen && isCentered && isGoodHeadAngle {
                currentState = .STABILIZING
                stableFrameCount = 0
                isFocused = false
                isFocusing = false
                focusAttempts = 0
                focusStartTime = 0
                detectionMessage = "Perfect Position! Hold Still"
                onStateChanged(currentState, detectionMessage, qualityScore, distanceStatus, faceDistanceCm, areEyesOpen, faceFramePath)
                os_log("ENTERED STABILIZING - Perfect Position!", log: OSLog.default, type: .debug)
            } else if !isEyesOpen {
                detectionMessage = "Please open your eyes for capture"
                onStateChanged(currentState, detectionMessage, qualityScore, distanceStatus, faceDistanceCm, areEyesOpen, faceFramePath)
            } else if !isCentered {
                detectionMessage = positionMessage
                onStateChanged(currentState, detectionMessage, qualityScore, distanceStatus, faceDistanceCm, areEyesOpen, faceFramePath)
                os_log("NOT CENTERED - Showing position guidance", log: OSLog.default, type: .debug)
            } else if !isGoodHeadAngle, let angleMessage = headAngleMessage {
                detectionMessage = angleMessage
                onStateChanged(currentState, detectionMessage, qualityScore, distanceStatus, faceDistanceCm, areEyesOpen, faceFramePath)
            }
        }

        switch currentState {
        case .WAITING:
            let distanceMessage = atTargetDistance ? "Perfect distance!" :
                                 faceDistanceCm < minTargetDistanceCm ? "Move back to \(minTargetDistanceCm)-\(maxTargetDistanceCm)cm" :
                                 "Move closer to \(minTargetDistanceCm)-\(maxTargetDistanceCm)cm"
            currentState = isBlinkDetectionEnabled ? .BLINK_CHECK : .DETECTING
            
            let centerMessage = isCentered ? "Face centered" : positionMessage
            let angleMessage = isGoodHeadAngle ? "" : (headAngleMessage ?? "")
            let mainMessage = !isCentered ? centerMessage :
                             !isGoodHeadAngle ? angleMessage :
                             distanceMessage
            let fullMessage = isBlinkDetectionEnabled ? "\(mainMessage). Please blink to continue." : mainMessage
            detectionMessage = fullMessage
            onStateChanged(currentState, detectionMessage, qualityScore, distanceStatus, faceDistanceCm, areEyesOpen, faceFramePath)
            
        case .DETECTING:
            if isGoodPosition && (!isBlinkDetectionEnabled || blinkDetected) && isEyesOpen {
                currentState = .STABILIZING
                stableFrameCount = 0
                isFocused = false
                isFocusing = false
                focusAttempts = 0
                focusStartTime = 0
                detectionMessage = "Perfect Position! Hold Still"
                onStateChanged(currentState, detectionMessage, qualityScore, distanceStatus, faceDistanceCm, areEyesOpen, faceFramePath)
                os_log("DETECTING -> STABILIZING - Perfect Position!", log: OSLog.default, type: .debug)
            } else {
                let message = !isCentered ? positionMessage :
                              !isEyesOpen ? "Please open your eyes for capture" :
                              !isGoodHeadAngle ? (headAngleMessage ?? "Adjust head position") :
                              !atTargetDistance ? "Move to \(minTargetDistanceCm)-\(maxTargetDistanceCm)cm" :
                              qualityScore <= 0.6 ? "Hold steady for better quality" : "Adjust your position"
                detectionMessage = message
                onStateChanged(currentState, detectionMessage, qualityScore, distanceStatus, faceDistanceCm, areEyesOpen, faceFramePath)
                
                if !isCentered {
                    os_log("DETECTING - NOT CENTERED - Showing: %@", log: OSLog.default, type: .debug, positionMessage)
                }
            }
            
        case .BLINK_CHECK:
            if !isBlinkDetectionEnabled {
                currentState = .DETECTING
                let centerMessage = isCentered ? "Face centered" : positionMessage
                let angleMessage = isGoodHeadAngle ? "" : (headAngleMessage ?? "")
                let mainMessage = !isCentered ? centerMessage :
                                 !isGoodHeadAngle ? angleMessage :
                                 "Keep at \(minTargetDistanceCm)-\(maxTargetDistanceCm)cm"
                detectionMessage = mainMessage
                onStateChanged(currentState, detectionMessage, qualityScore, distanceStatus, faceDistanceCm, areEyesOpen, faceFramePath)
            } else if isGoodPosition {
                let leftEyeOpen = face.leftEyeOpenProbability
                let rightEyeOpen = face.rightEyeOpenProbability
                let isBlink = leftEyeOpen < 0.4 && rightEyeOpen < 0.4
                
                if isBlink && !eyesClosed {
                    eyesClosed = true
                    detectionMessage = "Blink detected, open your eyes"
                    onStateChanged(currentState, detectionMessage, qualityScore, distanceStatus, faceDistanceCm, areEyesOpen, faceFramePath)
                } else if eyesClosed && isEyesOpen {
                    blinkDetected = true
                    currentState = .DETECTING
                    let centerMessage = isCentered ? "Face centered" : positionMessage
                    let angleMessage = isGoodHeadAngle ? "" : (headAngleMessage ?? "")
                    let mainMessage = !isCentered ? centerMessage :
                                     !isGoodHeadAngle ? angleMessage :
                                     "Keep at \(minTargetDistanceCm)-\(maxTargetDistanceCm)cm"
                    detectionMessage = mainMessage
                    onStateChanged(currentState, detectionMessage, qualityScore, distanceStatus, faceDistanceCm, areEyesOpen, faceFramePath)
                } else {
                    detectionMessage = "Please blink to continue"
                    onStateChanged(currentState, detectionMessage, qualityScore, distanceStatus, faceDistanceCm, areEyesOpen, faceFramePath)
                }
            } else {
                let centerMessage = isCentered ? "Face centered" : positionMessage
                let angleMessage = isGoodHeadAngle ? "" : (headAngleMessage ?? "")
                let message = !isCentered ? centerMessage :
                              !isGoodHeadAngle ? angleMessage :
                              !atTargetDistance ? "Move to \(minTargetDistanceCm)-\(maxTargetDistanceCm)cm" :
                              qualityScore <= 0.6 ? "Hold steady for better quality" : "Adjust your position"
                detectionMessage = message
                onStateChanged(currentState, detectionMessage, qualityScore, distanceStatus, faceDistanceCm, areEyesOpen, faceFramePath)
            }
            
        case .STABILIZING:
            if !isFocused {
                if !isFocusing && CACurrentMediaTime() - focusStartTime < maxFocusWaitTime {
                    isFocusing = true
                    focusStartTime = CACurrentMediaTime()
                    let focusPoint = CGPoint(x: face.frame.midX / currentImageWidth, y: face.frame.midY / currentImageHeight)
                    triggerFocus(at: focusPoint) { success in
                        self.isFocusing = false
                        if success && self.currentState == .STABILIZING {
                            self.isFocused = true
                            self.stableFrameCount = 0
                        } else {
                            self.focusAttempts += 1
                            if self.focusAttempts >= self.maxFocusAttempts {
                                self.isFocused = true
                                os_log("Camera: Proceeding without perfect focus after %d attempts", log: OSLog.default, type: .info, self.maxFocusAttempts)
                            }
                        }
                    }
                    detectionMessage = "Focusing... Hold still"
                    onStateChanged(currentState, detectionMessage, qualityScore, distanceStatus, faceDistanceCm, areEyesOpen, faceFramePath)
                } else if CACurrentMediaTime() - focusStartTime >= maxFocusWaitTime {
                    isFocused = true
                    os_log("Camera: Proceeding without focus due to timeout", log: OSLog.default, type: .info)
                    detectionMessage = "Perfect Position! Stay still (0/\(requiredStableFrames))"
                    onStateChanged(currentState, detectionMessage, qualityScore, distanceStatus, faceDistanceCm, areEyesOpen, faceFramePath)
                }
            } else {
                if isGoodPosition && isEyesOpen {
                    stableFrameCount += 1
                    detectionMessage = "Perfect Position! Stay still (\(stableFrameCount)/\(requiredStableFrames))"
                    onStateChanged(currentState, detectionMessage, qualityScore, distanceStatus, faceDistanceCm, areEyesOpen, faceFramePath)
                    if stableFrameCount >= requiredStableFrames {
                        currentState = .CAPTURING
                        detectionMessage = "Capturing..."
                        onStateChanged(currentState, detectionMessage, qualityScore, distanceStatus, faceDistanceCm, areEyesOpen, faceFramePath)
                        guard let sampleBuffer = self.lastSampleBuffer,
                              let image = CameraUtilities.toUIImage(sampleBuffer, isFrontCamera: self.isFrontCamera, normalizeOrientation: true) else {
                            os_log("Camera: No sample buffer or image for capture", log: OSLog.default, type: .error)
                            self.currentState = .WAITING
                            self.detectionMessage = "Position your face in the frame"
                            self.onStateChanged(.WAITING, self.detectionMessage, 0, .UNKNOWN, 0, false, nil)
                            return
                        }
                        self.lastSampleBuffer = nil
                        DispatchQueue.main.async { [weak self] in
                            guard let self = self else { return }
                            self.currentState = .SUCCESS
                            self.lastCaptureTime = CACurrentMediaTime()
                            self.detectionMessage = "Success!"
                            self.onStateChanged(.SUCCESS, self.detectionMessage, self.qualityScore, self.distanceStatus, self.faceDistanceCm, self.areEyesOpen, nil)
                            let finalImage = self.flipImageIfNeeded(image)
                            self.onImageCaptured(finalImage)
                            os_log("Camera: Automatic capture triggered after %d stable frames", log: OSLog.default, type: .info, self.requiredStableFrames)
                        }
                    }
                } else if !isEyesOpen {
                    stableFrameCount = 0
                    detectionMessage = "Please open your eyes for capture"
                    onStateChanged(currentState, detectionMessage, qualityScore, distanceStatus, faceDistanceCm, areEyesOpen, faceFramePath)
                } else if !isCentered {
                    stableFrameCount = 0
                    detectionMessage = positionMessage
                    onStateChanged(currentState, detectionMessage, qualityScore, distanceStatus, faceDistanceCm, areEyesOpen, faceFramePath)
                    os_log("STABILIZING - Lost centering", log: OSLog.default, type: .debug)
                } else if !isGoodHeadAngle {
                    stableFrameCount = 0
                    let angleMessage = headAngleMessage ?? "Adjust your head position"
                    detectionMessage = angleMessage
                    onStateChanged(currentState, detectionMessage, qualityScore, distanceStatus, faceDistanceCm, areEyesOpen, faceFramePath)
                } else {
                    stableFrameCount = max(0, stableFrameCount - 2)
                    let message = !atTargetDistance ? "Keep at \(minTargetDistanceCm)-\(maxTargetDistanceCm)cm" :
                                  qualityScore <= 0.6 ? "Hold steady for better quality" : "Stay still, keep your position"
                    detectionMessage = message
                    onStateChanged(currentState, detectionMessage, qualityScore, distanceStatus, faceDistanceCm, areEyesOpen, faceFramePath)
                }
            }
            
        case .CAPTURING, .SUCCESS, .COUNTDOWN, .LIVENESS_CHECK, .LIVENESS_FAILED:
            break
        }
    }
}
