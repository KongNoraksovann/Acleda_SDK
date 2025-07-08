import Foundation
import CoreGraphics
import MLKitVision
import MLKitFaceDetection
import os

@available(iOS 13.0, *)
public class MLKitFaceDetector {
    // MARK: - Logging
    private static let TAG = "FaceDetector"
    private let log = OSLog(subsystem: "com.example.faceverification.mlkit", category: "FaceDetector")

    private let faceQualityChecker = FaceQualityChecker()
    private let faceDetectorOptions = FaceDetectorOptions()
    private var detector: FaceDetector?

    struct FaceExtractionResult {
        let faceImage: CGImage?
        let qualityResult: FaceQualityResult
    }

    init() {
        faceDetectorOptions.performanceMode = .accurate
        faceDetectorOptions.landmarkMode = .all
        faceDetectorOptions.classificationMode = .all
        faceDetectorOptions.minFaceSize = 0.15
        faceDetectorOptions.isTrackingEnabled = true
        self.detector = FaceDetector.faceDetector(options: faceDetectorOptions)
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "hh:mm a z, MMMM dd, yyyy"
        let currentTime = dateFormatter.string(from: Date())
        os_log(.debug, log: log, "%{public}s: Initialized detector with performanceMode: ACCURATE, minFaceSize: 0.15 at %{public}s", Self.TAG, currentTime)
    }

    // Make this method internal so it can be called from other files if needed
    func processVisionImage(_ visionImage: VisionImage) async throws -> [Face] {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "hh:mm a z, MMMM dd, yyyy"
        let currentTime = dateFormatter.string(from: Date())

        guard let detector = detector else {
            os_log(.error, log: log, "%{public}s: Detector not initialized at %{public}s", Self.TAG, currentTime)
            throw NSError(domain: "FaceDetector", code: -1, userInfo: [NSLocalizedDescriptionKey: "Detector not initialized"])
        }

        return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<[Face], Error>) in
            detector.process(visionImage) { faces, error in
                if let error = error {
                    let currentTimeError = dateFormatter.string(from: Date())
                    os_log(.error, log: self.log, "%{public}s: Error detecting faces: %{public}s at %{public}s", Self.TAG, error.localizedDescription, currentTimeError)
                    continuation.resume(throwing: error)
                    return
                }
                let currentTimeSuccess = dateFormatter.string(from: Date())
                os_log(.debug, log: self.log, "%{public}s: Detected %d faces in the image at %{public}s", Self.TAG, faces?.count ?? 0, currentTimeSuccess)
                continuation.resume(returning: faces ?? [])
            }
        }
    }

    func detectFaces(in image: CGImage) async -> [Face] {
        let uiImage = UIImage(cgImage: image)
        let visionImage = VisionImage(image: uiImage)
        visionImage.orientation = .up

        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "hh:mm a z, MMMM dd, yyyy"
        let currentTime = dateFormatter.string(from: Date())
        os_log(.debug, log: log, "%{public}s: Detecting faces in image at %{public}s", Self.TAG, currentTime)

        do {
            return try await processVisionImage(visionImage)
        } catch {
            let currentTime = dateFormatter.string(from: Date())
            os_log(.error, log: log, "%{public}s: Error during face detection: %{public}s at %{public}s", Self.TAG, error.localizedDescription, currentTime)
            return []
        }
    }

    private func attemptFaceDetection(in image: CGImage) async -> [Face] {
        var faces = await detectFaces(in: image)
        if faces.isEmpty {
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "hh:mm a z, MMMM dd, yyyy"
            let currentTime = dateFormatter.string(from: Date())
            os_log(.debug, log: log, "%{public}s: No face found at original quality, trying with compressed image at %{public}s", Self.TAG, currentTime)

            let uiImage = UIImage(cgImage: image)
            if let compressed = compressBitmap(uiImage, quality: 80), let cgCompressed = compressed.cgImage {
                faces = await detectFaces(in: cgCompressed)
            } else {
                let currentTimeCompressFail = dateFormatter.string(from: Date())
                os_log(.error, log: log, "%{public}s: Failed to compress image at %{public}s", Self.TAG, currentTimeCompressFail)
            }
        }
        return faces
    }

    private func processFallbackImage(_ uiImage: UIImage) -> FaceExtractionResult {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "hh:mm a z, MMMM dd, yyyy"
        let currentTime = dateFormatter.string(from: Date())
        os_log(.debug, log: log, "%{public}s: No faces detected, using fallback resize and crop at %{public}s", Self.TAG, currentTime)

        let noFaceQualityResult = FaceQualityResult(
            isGoodQuality: false,
            qualityScore: 0.0,
            issues: [.noFaceDetected],
            failureReason: "No face detected in the image"
        )
        if let fallbackImage = applyResizeAndCenterCrop(uiImage), let cgFallbackImage = fallbackImage.cgImage {
            return FaceExtractionResult(faceImage: cgFallbackImage, qualityResult: noFaceQualityResult)
        }
        return FaceExtractionResult(faceImage: nil, qualityResult: noFaceQualityResult)
    }

    func cropFace(from image: CGImage) async -> FaceExtractionResult {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "hh:mm a z, MMMM dd, yyyy"   
        let currentTime = dateFormatter.string(from: Date())
        os_log(.debug, log: log, "%{public}s: Loaded image size: %dx%d at %{public}s", Self.TAG, image.width, image.height, currentTime)

        let uiImage = UIImage(cgImage: image)
        let faces = await attemptFaceDetection(in: image)

        guard let largestFace = faces.max(by: { $0.frame.width * $0.frame.height < $1.frame.width * $1.frame.height }) else {
            return processFallbackImage(uiImage)
        }

        let qualityResult = faceQualityChecker.checkFaceQuality(image: image)
        let currentTimeQuality = dateFormatter.string(from: Date())
        os_log(.debug, log: log, "%{public}s: Face quality check result: isGoodQuality=%{public}s, score=%f at %{public}s",
               Self.TAG, qualityResult.isGoodQuality ? "true" : "false", qualityResult.qualityScore, currentTimeQuality)

        let box = largestFace.frame
        let faceImage = cropFaceTightly(uiImage, boundingBox: box)
        let processedImage = faceImage ?? uiImage
        if let finalImage = applyResizeAndCenterCrop(processedImage), let cgProcessedImage = finalImage.cgImage {
            let currentTimeSuccess = dateFormatter.string(from: Date())
            os_log(.debug, log: log, "%{public}s: Face cropped and processed successfully: %dx%d at %{public}s",
                   Self.TAG, Int(finalImage.size.width), Int(finalImage.size.height), currentTimeSuccess)
            return FaceExtractionResult(faceImage: cgProcessedImage, qualityResult: qualityResult)
        }

        return processFallbackImage(uiImage)
    }

    func assessFaceQuality(image: CGImage) async -> FaceQualityResult {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "hh:mm a z, MMMM dd, yyyy"
        let currentTime = dateFormatter.string(from: Date())
        os_log(.debug, log: log, "%{public}s: Checking face quality for image (%dx%d) at %{public}s", Self.TAG, image.width, image.height, currentTime)

        let faces = await detectFaces(in: image)

        if faces.isEmpty {
            let currentTimeNoFace = dateFormatter.string(from: Date())
            os_log(.debug, log: log, "%{public}s: No faces detected during quality check at %{public}s", Self.TAG, currentTimeNoFace)
            return FaceQualityResult(
                isGoodQuality: false,
                qualityScore: 0.0,
                issues: [.noFaceDetected],
                failureReason: "No face detected in the image"
            )
        }

        guard let largestFace = faces.max(by: { $0.frame.width * $0.frame.height < $1.frame.width * $1.frame.height }) else {
            let currentTimeNoLargest = dateFormatter.string(from: Date())
            os_log(.debug, log: log, "%{public}s: No largest face found during quality check at %{public}s", Self.TAG, currentTimeNoLargest)
            return FaceQualityResult(
                isGoodQuality: false,
                qualityScore: 0.0,
                issues: [.noFaceDetected],
                failureReason: "No face detected in the image"
            )
        }

        let qualityResult = faceQualityChecker.checkFaceQuality(image: image)
        let currentTimeQuality = dateFormatter.string(from: Date())
        os_log(.debug, log: log, "%{public}s: Face quality check result: isGoodQuality=%{public}s, score=%f at %{public}s",
               Self.TAG, qualityResult.isGoodQuality ? "true" : "false", qualityResult.qualityScore, currentTimeQuality)
        return qualityResult
    }

    private func cropFaceTightly(_ image: UIImage, boundingBox: CGRect) -> UIImage? {
        let normalizedImage = normalizeImage(image)

        let left = Int(max(0, boundingBox.origin.x))
        let top = Int(max(0, boundingBox.origin.y))
        let right = Int(min(boundingBox.maxX, normalizedImage.size.width))
        let bottom = Int(min(boundingBox.maxY, normalizedImage.size.height))
        let width = right - left
        let height = bottom - top

        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "hh:mm a z, MMMM dd, yyyy"
        let currentTime = dateFormatter.string(from: Date())
        os_log(.debug, log: log, "%{public}s: Crop box: left=%d, top=%d, width=%d, height=%d at %{public}s",
               Self.TAG, left, top, width, height, currentTime)

        guard width > 0, height > 0 else {
            os_log(.error, log: log, "%{public}s: Invalid crop dimensions: width=%d, height=%d at %{public}s",
                   Self.TAG, width, height, currentTime)
            return nil
        }

        let cropRect = CGRect(x: CGFloat(left), y: CGFloat(top), width: CGFloat(width), height: CGFloat(height))
        guard let cgImage = normalizedImage.cgImage else {
            os_log(.error, log: log, "%{public}s: Failed to get CGImage at %{public}s", Self.TAG, currentTime)
            return nil
        }

        guard let croppedCGImage = cgImage.cropping(to: cropRect) else {
            os_log(.error, log: log, "%{public}s: Failed to crop image to rect: x=%d, y=%d, w=%d, h=%d at %{public}s",
                   Self.TAG, left, top, width, height, currentTime)
            return nil
        }

        let croppedImage = UIImage(cgImage: croppedCGImage, scale: 1.0, orientation: .up)
        os_log(.debug, log: log, "%{public}s: Cropped image: %dx%d, orientation=%d at %{public}s",
               Self.TAG, Int(croppedImage.size.width), Int(croppedImage.size.height), croppedImage.imageOrientation.rawValue, currentTime)
        return croppedImage
    }

    private func applyResizeAndCenterCrop(_ image: UIImage) -> UIImage? {
        let targetSize = CGSize(width: 256, height: 256)
        UIGraphicsBeginImageContextWithOptions(targetSize, false, 1.0)
        image.draw(in: CGRect(origin: .zero, size: targetSize))
        guard let resizedImage = UIGraphicsGetImageFromCurrentImageContext() else {
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "hh:mm a z, MMMM dd, yyyy"
            let currentTime = dateFormatter.string(from: Date())
            os_log(.error, log: log, "%{public}s: Failed to resize image to 256x256 at %{public}s", Self.TAG, currentTime)
            UIGraphicsEndImageContext()
            return nil
        }
        UIGraphicsEndImageContext()
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "hh:mm a z, MMMM dd, yyyy"
        let currentTimeResize = dateFormatter.string(from: Date())
        os_log(.debug, log: log, "%{public}s: Resized image: %dx%d at %{public}s",
               Self.TAG, Int(resizedImage.size.width), Int(resizedImage.size.height), currentTimeResize)

        let cropSize = CGSize(width: 224, height: 224)
        let x = (resizedImage.size.width - cropSize.width) / 2
        let y = (resizedImage.size.height - cropSize.height) / 2
        let cropRect = CGRect(x: x, y: y, width: cropSize.width, height: cropSize.height)

        guard let croppedCGImage = resizedImage.cgImage?.cropping(to: cropRect) else {
            let currentTimeCrop = dateFormatter.string(from: Date())
            os_log(.error, log: log, "%{public}s: Failed to crop image to 224x224 at %{public}s", Self.TAG, currentTimeCrop)
            return nil
        }

        let croppedImage = UIImage(cgImage: croppedCGImage, scale: 1.0, orientation: .up)
        let currentTimeCropSuccess = dateFormatter.string(from: Date())
        os_log(.debug, log: log, "%{public}s: Cropped image: %dx%d at %{public}s",
               Self.TAG, Int(croppedImage.size.width), Int(croppedImage.size.height), currentTimeCropSuccess)
        return croppedImage
    }

    private func normalizeImage(_ image: UIImage) -> UIImage {
        if image.imageOrientation == .up {
            return image
        }

        UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
        image.draw(in: CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height))
        let normalizedImage = UIGraphicsGetImageFromCurrentImageContext() ?? image
        UIGraphicsEndImageContext()

        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "hh:mm a z, MMMM dd, yyyy"
        let currentTime = dateFormatter.string(from: Date())
        os_log(.debug, log: log, "%{public}s: Normalized image orientation from %d to up at %{public}s",
               Self.TAG, image.imageOrientation.rawValue, currentTime)
        return normalizedImage
    }

    private func compressBitmap(_ image: UIImage, quality: Int) -> UIImage? {
        let compressionQuality = CGFloat(quality) / 100.0
        guard let data = image.jpegData(compressionQuality: compressionQuality),
              let compressedImage = UIImage(data: data, scale: 1.0) else {
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "hh:mm a z, MMMM dd, yyyy"
            let currentTime = dateFormatter.string(from: Date())
            os_log(.error, log: log, "%{public}s: Failed to compress image with quality %d at %{public}s", Self.TAG, quality, currentTime)
            return nil
        }
        let normalizedCompressed = normalizeImage(compressedImage)
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "hh:mm a z, MMMM dd, yyyy"
        let currentTime = dateFormatter.string(from: Date())
        os_log(.debug, log: log, "%{public}s: Compressed image (quality=%d): %dx%d at %{public}s",
               Self.TAG, quality, Int(normalizedCompressed.size.width), Int(normalizedCompressed.size.height), currentTime)
        return normalizedCompressed
    }

    func close() {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "hh:mm a z, MMMM dd, yyyy"
        let currentTime = dateFormatter.string(from: Date())
        os_log(.debug, log: log, "%{public}s: Detector closed at %{public}s", Self.TAG, currentTime)
    }
}
@available(iOS 13.0, *)
extension MLKitFaceDetector {
    struct LandmarkPoint {
        let x: Float
        let y: Float
    }
    
    func detectFace(in image: CGImage) async -> (CGImage?, [[Float]])? {
        let uiImage = UIImage(cgImage: image)
        let visionImage = VisionImage(image: uiImage)
        visionImage.orientation = .up
        
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "hh:mm a z, MMMM dd, yyyy"
        let currentTime = dateFormatter.string(from: Date())
        os_log(.debug, log: log, "%{public}s: Detecting face for cropping and landmarks at %{public}s", MLKitFaceDetector.TAG, currentTime)
        
        do {
            let faces = try await processVisionImage(visionImage)
            
            if faces.isEmpty {
                os_log(.debug, log: log, "%{public}s: No faces detected at %{public}s", MLKitFaceDetector.TAG, currentTime)
                return nil
            }
            
            let face = faces[0]
            let boundingBox = face.frame
            let x = max(0, Int(boundingBox.origin.x))
            let y = max(0, Int(boundingBox.origin.y))
            let width = min(Int(boundingBox.width), image.width - x)
            let height = min(Int(boundingBox.height), image.height - y)
            
            guard width > 0, height > 0 else {
                let currentTimeError = dateFormatter.string(from: Date())
                os_log(.error, log: log, "%{public}s: Invalid bounding box dimensions: width=%d, height=%d at %{public}s", MLKitFaceDetector.TAG, width, height, currentTimeError)
                return nil
            }
            
            let cropRect = CGRect(x: CGFloat(x), y: CGFloat(y), width: CGFloat(width), height: CGFloat(height))
            guard let cgImage = uiImage.cgImage else {
                let currentTimeError = dateFormatter.string(from: Date())
                os_log(.error, log: log, "%{public}s: Failed to get CGImage at %{public}s", MLKitFaceDetector.TAG, currentTimeError)
                return nil
            }
            
            guard let croppedCGImage = cgImage.cropping(to: cropRect) else {
                let currentTimeError = dateFormatter.string(from: Date())
                os_log(.error, log: log, "%{public}s: Failed to crop face to rect: x=%d, y=%d, w=%d, h=%d at %{public}s", MLKitFaceDetector.TAG, x, y, width, height, currentTimeError)
                return nil
            }
            
            let croppedImage = UIImage(cgImage: croppedCGImage, scale: 1.0, orientation: .up)
            let currentTimeSuccess = dateFormatter.string(from: Date())
            os_log(.debug, log: log, "%{public}s: Cropped face: %dx%d at %{public}s", MLKitFaceDetector.TAG, Int(croppedImage.size.width), Int(croppedImage.size.height), currentTimeSuccess)
            
            let landmarks = extractLandmarks(face: face, offsetX: x, offsetY: y, width: width, height: height)
            
            return (croppedCGImage, landmarks)
        } catch {
            let currentTimeError = dateFormatter.string(from: Date())
            os_log(.error, log: log, "%{public}s: Face detection error: %{public}s at %{public}s", MLKitFaceDetector.TAG, error.localizedDescription, currentTimeError)
            return nil
        }
    }
    
    private func extractLandmarks(face: Face, offsetX: Int, offsetY: Int, width: Int, height: Int) -> [[Float]] {
        let leftEye = face.landmark(ofType: .leftEye)
        let rightEye = face.landmark(ofType: .rightEye)
        let noseBase = face.landmark(ofType: .noseBase)
        let leftMouth = face.landmark(ofType: .mouthLeft)
        let rightMouth = face.landmark(ofType: .mouthRight)
        
        // Helper function to compute coordinates with fallback
        func getCoordinate(landmark: FaceLandmark?, xFactor: Float, yFactor: Float, offsetX: Int, offsetY: Int, width: Int, height: Int) -> [Float] {
            let x = landmark.map { Float($0.position.x) } ?? (Float(offsetX) + Float(width) * xFactor)
            let y = landmark.map { Float($0.position.y) } ?? (Float(offsetY) + Float(height) * yFactor)
            return [x, y]
        }
        
        // Construct the landmarks array step-by-step
        let leftEyeCoords = getCoordinate(landmark: leftEye, xFactor: 0.3, yFactor: 0.3, offsetX: offsetX, offsetY: offsetY, width: width, height: height)
        let rightEyeCoords = getCoordinate(landmark: rightEye, xFactor: 0.7, yFactor: 0.3, offsetX: offsetX, offsetY: offsetY, width: width, height: height)
        let noseBaseCoords = getCoordinate(landmark: noseBase, xFactor: 0.5, yFactor: 0.5, offsetX: offsetX, offsetY: offsetY, width: width, height: height)
        let leftMouthCoords = getCoordinate(landmark: leftMouth, xFactor: 0.4, yFactor: 0.7, offsetX: offsetX, offsetY: offsetY, width: width, height: height)
        let rightMouthCoords = getCoordinate(landmark: rightMouth, xFactor: 0.6, yFactor: 0.7, offsetX: offsetX, offsetY: offsetY, width: width, height: height)
        
        return [
            leftEyeCoords,
            rightEyeCoords,
            noseBaseCoords,
            leftMouthCoords,
            rightMouthCoords
        ]
    }
}
