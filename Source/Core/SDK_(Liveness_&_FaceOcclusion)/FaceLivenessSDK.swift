
import Foundation
import UIKit
import MLKitFaceDetection
import MLKitVision
import os.log

// MARK: - AutoCloseable Protocol
@objc public protocol AutoCloseable {
    func close()
}

// MARK: - FaceLivenessSDK
@available(iOS 13.0, *)
@objc public class FaceLivenessSDK: NSObject, AutoCloseable {
    private let TAG = "FaceLivenessSDK"
    private let bundle: Bundle
    private let config: Config
    private let log = OSLog(subsystem: "com.acleda.facelivenesssdk", category: "FaceLivenessSDK")
    private let faceDetector = MLKitFaceDetector()

    // Lazy initialization of components
    private lazy var livenessDetector: LivenessDetector? = {
        do {
            return try LivenessDetector(bundle: bundle)
        } catch {
            LogUtils.e(TAG, "Failed to initialize LivenessDetector: \(error.localizedDescription)", error)
            return nil
        }
    }()
    
    private lazy var occlusionDetector: FaceOcclusionDetector? = {
        do {
            return try FaceOcclusionDetector(bundle: bundle)
        } catch {
            LogUtils.e(TAG, "Failed to initialize FaceOcclusionDetector: \(error.localizedDescription)", error)
            return nil
        }
    }()
    
    private lazy var albedoDetector: AlbedoDetector? = {
        return AlbedoDetector()
    }()
    
    private lazy var faceQualityChecker: FaceQualityChecker? = {
        return FaceQualityChecker()
    }()

    // MARK: - Config
    @objc public class Config: NSObject {
        public let enableDebugLogging: Bool
        public let skipOcclusionCheck: Bool
        public let skipAlbedoCheck: Bool
        public let skipFaceCropping: Bool

        private init(
            enableDebugLogging: Bool,
            skipOcclusionCheck: Bool,
            skipAlbedoCheck: Bool,
            skipFaceCropping: Bool
        ) {
            self.enableDebugLogging = enableDebugLogging
            self.skipOcclusionCheck = skipOcclusionCheck
            self.skipAlbedoCheck = skipAlbedoCheck
            self.skipFaceCropping = skipFaceCropping
        }

        @objc public class Builder: NSObject {
            private var enableDebugLogging = false
            private var skipOcclusionCheck = false
            private var skipAlbedoCheck = false
            private var skipFaceCropping = false

            @objc public func setDebugLoggingEnabled(_ enabled: Bool) -> Builder {
                self.enableDebugLogging = enabled
                return self
            }

            @objc public func setSkipOcclusionCheck(_ skip: Bool) -> Builder {
                self.skipOcclusionCheck = skip
                return self
            }

            @objc public func setSkipAlbedoCheck(_ skip: Bool) -> Builder {
                self.skipAlbedoCheck = skip
                return self
            }

            @objc public func setSkipFaceCropping(_ skip: Bool) -> Builder {
                self.skipFaceCropping = skip
                return self
            }

            @objc public func build() -> Config {
                return Config(
                    enableDebugLogging: enableDebugLogging,
                    skipOcclusionCheck: skipOcclusionCheck,
                    skipAlbedoCheck: skipAlbedoCheck,
                    skipFaceCropping: skipFaceCropping
                )
            }
        }
    }

    // MARK: - Initialization
    private init(bundle: Bundle, config: Config) {
        self.bundle = bundle
        self.config = config
        super.init()

        LogUtils.setDebugEnabled(config.enableDebugLogging)
       31
        LogUtils.i(
            TAG,
            "FaceLivenessSDK initialized with config: debugLogging=\(config.enableDebugLogging), skipOcclusionCheck=\(config.skipOcclusionCheck), skipAlbedoCheck=\(config.skipAlbedoCheck), skipFaceCropping=\(config.skipFaceCropping)"
        )
    }

    // MARK: - Factory Methods
    @objc public static func create(_ bundle: Bundle = Bundle.main) -> FaceLivenessSDK {
        return create(bundle, config: Config.Builder().build())
    }

    @objc public static func create(_ bundle: Bundle, config: Config) -> FaceLivenessSDK {
        return FaceLivenessSDK(bundle: bundle, config: config)
    }

    // MARK: - Public Methods
    public func detectLiveness(_ image: UIImage) async throws -> (FaceLivenessModel, [String: Float]?, [String: Float]?) {
        LogUtils.d(TAG, "Starting face liveness detection process with detailed scores")

        // Validate input using BitmapUtils
        guard BitmapUtils.validateImage(image) else {
            LogUtils.e(TAG, "Invalid input image")
            throw InvalidImageException("Invalid input image")
        }

        do {
            var processingImage = image
            var occlusionScores: [String: Float]? = nil
            var livenessScores: [String: Float]? = nil

            // Step 0: Crop face if enabled
            if !config.skipFaceCropping {
                LogUtils.i(TAG, "Cropping face before analysis")
                guard let cgImage = image.cgImage else {
                    LogUtils.e(TAG, "Failed to get CGImage from UIImage")
                    throw FaceLivenessException("Invalid image format")
                }

                let extractionResult = await faceDetector.cropFace(from: cgImage)
                if let croppedCGImage = extractionResult.faceImage {
                    processingImage = UIImage(cgImage: croppedCGImage)
                    LogUtils.i(TAG, "Successfully cropped face: \(processingImage.size.width)x\(processingImage.size.height)")
                } else {
                    LogUtils.w(TAG, "Face cropping failed, using original image")
                }
            }

            // Step 1: Face quality check
            guard let faceQualityChecker = faceQualityChecker else {
                throw FaceLivenessException("Face quality checker not initialized")
            }
            
            guard let cgImage = processingImage.cgImage else {
                LogUtils.e(TAG, "Failed to get CGImage for quality check")
                throw FaceLivenessException("Invalid image format")
            }

            let qualityResult = faceQualityChecker.checkFaceQuality(image: cgImage)
            if !qualityResult.isGoodQuality {
                LogUtils.i(TAG, "Laplacian/blur check failed: \(qualityResult.failureReason ?? "Face quality check failed")")
                return (
                    FaceLivenessModel(
                        prediction: "Spoof",
                        confidence: 1.0,
                        failureReason: qualityResult.failureReason ?? "Face quality check failed"
                    ),
                    nil,
                    nil
                )
            }

            // Step 2: Albedo check
            if !config.skipAlbedoCheck {
                guard let albedoDetector = albedoDetector else {
                    throw FaceLivenessException("Albedo detector not initialized")
                }
                LogUtils.d(TAG, "Performing albedo detection")
                let albedoResult = try albedoDetector.detectSpoof(bitmap: processingImage)

                LogUtils.i(TAG, "Albedo detection result: \(albedoResult.prediction) (Green: \(albedoResult.greenOutliers), Blue: \(albedoResult.blueOutliers))")
                LogUtils.i(TAG, "Albedo values - R: \(albedoResult.albedoValueR), G: \(albedoResult.albedoValueG), B: \(albedoResult.albedoValueB)")

                if !albedoResult.isLive {
                    return (
                        FaceLivenessModel(
                            prediction: "Spoof",
                            confidence: 1.0,
                            failureReason: "Albedo check failed: Image is spoof"
                        ),
                        nil,
                        nil
                    )
                }
            } else {
                LogUtils.d(TAG, "Albedo check skipped as per configuration")
            }

            // Step 3: Occlusion check
            if !config.skipOcclusionCheck {
                guard let occlusionDetector = occlusionDetector else {
                    throw FaceLivenessException("Occlusion detector not initialized")
                }
                LogUtils.d(TAG, "Performing occlusion detection")
                occlusionScores = try occlusionDetector.getDetailedScores(bitmap: processingImage)
                let occlusionResult = try occlusionDetector.detectFaceMaskWithAveraging(bitmap: processingImage, iterations: 3)

                LogUtils.i(TAG, "Final occlusion detection decision: \(occlusionResult.label) (confidence: \(String(format: "%.4f", occlusionResult.confidence)))")
                LogUtils.i(TAG, "Detailed occlusion scores: \(occlusionScores?.description ?? "null")")

                if occlusionResult.label != "normal" {
                    return (
                        FaceLivenessModel(
                            prediction: "Spoof",
                            confidence: occlusionResult.confidence,
                            failureReason: "Face is occluded: \(occlusionResult.label)"
                        ),
                        nil,
                        occlusionScores
                    )
                }
            } else {
                LogUtils.d(TAG, "Face occlusion check skipped as per configuration")
            }

            // Step 4: Liveness check
            guard let livenessDetector = livenessDetector else {
                throw FaceLivenessException("Liveness detector not initialized")
            }
            LogUtils.d(TAG, "Performing liveness detection")
            livenessScores = try livenessDetector.getDetailedScores(bitmap: processingImage)
            let livenessResult = try livenessDetector.runInferenceWithAveraging(bitmap: processingImage, iterations: 3)

            LogUtils.i(TAG, "Final liveness detection decision: \(livenessResult.label) (confidence: \(String(format: "%.4f", livenessResult.confidence)))")
            LogUtils.i(TAG, "Detailed liveness scores: \(livenessScores?.description ?? "null")")

            return (
                FaceLivenessModel(
                    prediction: livenessResult.label,
                    confidence: livenessResult.confidence,
                    failureReason: livenessResult.label == "Live" ? nil : "Liveness check failed"
                ),
                livenessScores,
                occlusionScores
            )
        } catch {
            LogUtils.e(TAG, "Detection pipeline error: \(error.localizedDescription)", error)
            throw FaceLivenessException("SDK error: \(error.localizedDescription)", error)
        }
    }

    @objc public func getVersion() -> String {
        return "1.0.0"
    }

    @objc public func close() {
        LogUtils.d(TAG, "Closing FaceLivenessSDK resources")
        if !config.skipOcclusionCheck {
            do {
                occlusionDetector?.close()
                occlusionDetector = nil
            } catch {
                LogUtils.e(TAG, "Error closing occlusionDetector: \(error.localizedDescription)", error)
            }
        }

        do {
            livenessDetector?.close()
            livenessDetector = nil
        } catch {
            LogUtils.e(TAG, "Error closing livenessDetector: \(error.localizedDescription)", error)
        }

        faceDetector.close()
        faceQualityChecker = nil
        albedoDetector = nil
    }
}


@available(iOS 13.0, *)
public class FaceEmbedding {
    private let TAG = "FaceEmbedding"
    private let model: FaceEmbeddingModel
    private let faceDetector: MLKitFaceDetector
    private let bundle: Bundle
    private let log = OSLog(subsystem: "com.acleda.facelivenesssdk", category: "FaceEmbedding")

    public init(bundle: Bundle = .main) {
        self.bundle = bundle
        self.model = FaceEmbeddingModel(bundle: bundle)
        self.faceDetector = MLKitFaceDetector()
    }

    public func initialize() async -> Bool {
        do {
            let modelLoaded = try await model.loadModel()
            return modelLoaded
        } catch {
            LogUtils.e(TAG, "Failed to initialize model: \(error.localizedDescription)")
            return false
        }
    }

    private func normalizeImage(_ image: UIImage) -> UIImage? {
        guard image.imageOrientation != .up else { return image }
        
        UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
        defer { UIGraphicsEndImageContext() }
        image.draw(in: CGRect(origin: .zero, size: image.size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }

    private func preprocessImage(_ image: UIImage, maxSize: CGFloat = 640.0) -> UIImage? {
        let aspectRatio = image.size.width / image.size.height
        var newSize: CGSize
        if image.size.width > image.size.height {
            newSize = CGSize(width: maxSize, height: maxSize / aspectRatio)
        } else {
            newSize = CGSize(width: maxSize * aspectRatio, height: maxSize)
        }

        UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
        defer { UIGraphicsEndImageContext() }
        image.draw(in: CGRect(origin: .zero, size: newSize))
        return UIGraphicsGetImageFromCurrentImageContext()
    }

    public func getFaceEmbedding(faceImage: UIImage, source: String = "unknown") async -> ([Double]?, UIImage?) {
        guard BitmapUtils.validateImage(faceImage) else {
            LogUtils.e(TAG, "Invalid image provided")
            return (nil, nil)
        }

        guard let normalizedImage = normalizeImage(faceImage) else {
            LogUtils.e(TAG, "Failed to normalize image orientation")
            return (nil, nil)
        }

        guard let preprocessedImage = preprocessImage(normalizedImage) else {
            LogUtils.e(TAG, "Failed to preprocess image")
            return (nil, nil)
        }

        LogUtils.d(TAG, "Image properties: size=\(preprocessedImage.size), orientation=\(preprocessedImage.imageOrientation.rawValue), scale=\(preprocessedImage.scale)")

        guard let cgImage = preprocessedImage.cgImage else {
            LogUtils.e(TAG, "Failed to get CGImage from UIImage")
            return (nil, nil)
        }

        guard let detectionResult = await faceDetector.detectFace(in: cgImage) else {
            LogUtils.e(TAG, "No face detected by ML Kit")
            return (nil, nil)
        }
        
        let (croppedCGImage, landmarks) = detectionResult
        
        guard let croppedCGImage = croppedCGImage, landmarks.count == 5 else {
            LogUtils.e(TAG, "Invalid detection result: face or landmarks missing")
            return (nil, nil)
        }

        let croppedUIImage = UIImage(cgImage: croppedCGImage)

        let alignedBitmap = await alignFace(faceImage: croppedUIImage, facialLandmarks: landmarks, source: source)

        let embedding = alignedBitmap != nil ? try? await model.extractFaceEmbedding(faceImage: alignedBitmap!, source: source) : nil

        return (embedding, alignedBitmap)
    }

    public func alignFace(faceImage: UIImage, facialLandmarks: [[Float]], source: String = "unknown") async -> UIImage? {
        guard BitmapUtils.validateImage(faceImage) else {
            LogUtils.e(TAG, "Invalid image provided")
            return nil
        }

        guard facialLandmarks.count == 5 else {
            LogUtils.e(TAG, "Invalid number of landmarks: \(facialLandmarks.count)")
            return nil
        }

        return try? await model.alignFace(image: faceImage, facialLandmarks: facialLandmarks, source: source)
    }

    public func cosineSimilarity(embedding1: [Double], embedding2: [Double]) -> Float {
        return model.cosineSimilarity(embedding1: embedding1, embedding2: embedding2)
    }

    public func verifyFaces(embedding1: [Double], embedding2: [Double]) -> Bool {
        let similarity = model.cosineSimilarity(embedding1: embedding1, embedding2: embedding2)
        let threshold = model.getCosineThreshold()
        LogUtils.d(TAG, "Similarity: \(similarity), Threshold: \(threshold), Pass: \(similarity > threshold)")
        return similarity > threshold
    }

    public func setCosineThreshold(threshold: Float) {
        model.setCosineThreshold(threshold)
    }

    public func getCosineThreshold() -> Float {
        return model.getCosineThreshold()
    }

    public func release() {
        model.close()
        faceDetector.close()
    }

    deinit {
        release()
    }
}
