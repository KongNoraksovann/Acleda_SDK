//import Foundation
//import UIKit
//import CoreImage
//import MLKitVision
//import MLKitFaceDetection
//
///**
// * A utility class to assess the quality of an image for face liveness detection within the FaceLivenessSDK.
// * Evaluates brightness, sharpness, and face presence using predefined thresholds.
// * Designed for external use in iOS applications requiring face liveness verification.
// */
//@objc public class ImageQualityChecker: NSObject {
//    private let TAG = "ImageQualityChecker"
//
//    // MARK: - Public Thresholds
//    @objc public var brightnessTooLow: Float = 40.0
//    @objc public var brightnessSomewhatLow: Float = 80.0
//    @objc public var brightnessGoodUpper: Float = 180.0
//    @objc public var brightnessSomewhatHigh: Float = 220.0
//
//    @objc public var sharpnessVeryBlurry: Float = 5.0
//    @objc public var sharpnessSomewhatBlurry: Float = 10.0
//    @objc public var sharpnessGoodUpper: Float = 50.0
//    @objc public var sharpnessTooDetailed: Float = 100.0
//
//    // MARK: - Private Properties
//    private var faceDetector: MLKitFaceDetection.FaceDetector // Changed to var to allow updates
//    
//    // MARK: - Public Properties
//    @objc public static let shared = ImageQualityChecker()
//    
//    @objc public var minFaceSize: Float = 0.2 {
//        didSet {
//            updateFaceDetector()
//        }
//    }
//    
//    @objc public var faceDetectionPerformanceMode: FaceDetectorPerformanceMode = .fast {
//        didSet {
//            updateFaceDetector()
//        }
//    }
//
//    // MARK: - Initialization
//    @objc public override init() {
//        let options = FaceDetectorOptions()
//        options.performanceMode = .fast
//        options.minFaceSize = 0.2
//        options.landmarkMode = .none
//        options.classificationMode = .none
//        self.faceDetector = MLKitFaceDetection.FaceDetector.faceDetector(options: options)
//        super.init()
//    }
//    
//    @objc public convenience init(minFaceSize: Float, performanceMode: FaceDetectorPerformanceMode) {
//        self.init()
//        self.minFaceSize = minFaceSize
//        self.faceDetectionPerformanceMode = performanceMode
//        updateFaceDetector()
//    }
//
//    // MARK: - Public Methods
//    @objc public func checkImageQuality(image: UIImage, completion: @escaping (ImageQualityResult?, Error?) -> Void) {
//        LogUtils.d(self.TAG, "Checking image quality for image: \(Int(image.size.width))x\(Int(image.size.height))")
//
//        guard BitmapUtils.validateImage(image) else {
//            LogUtils.e(self.TAG, "Invalid image provided")
//            completion(nil, QualityCheckException("Invalid image provided"))
//            return
//        }
//
//        let result = ImageQualityResult()
//
//        result.brightnessScore = checkBrightness(image)
//        result.sharpnessScore = checkSharpness(image)
//
//        checkFacePresence(image) { hasFace, error in
//            if let error = error {
//                LogUtils.e(self.TAG, "Error in quality check: \(error.localizedDescription)", error)
//                completion(nil, QualityCheckException("Error in quality check: \(error.localizedDescription)", error))
//                return
//            }
//
//            result.hasFace = hasFace
//            result.faceScore = hasFace ? 1.0 : 0.0
//            result.calculateOverallScore()
//            LogUtils.d(self.TAG, "Quality check completed. Overall score: \(result.overallScore)")
//            completion(result, nil)
//        }
//    }
//    
//    @objc public func getBrightnessScore(_ image: UIImage) -> Float {
//        return checkBrightness(image) // Fixed: Changed FMcheckBrightness to checkBrightness
//    }
//    
//    @objc public func getSharpnessScore(_ image: UIImage) -> Float {
//        return checkSharpness(image)
//    }
//    
//    @objc public func detectFacePresence(in image: UIImage, completion: @escaping (Bool, Error?) -> Void) {
//        checkFacePresence(image, completion: completion)
//    }
//
//    @objc public func close() {
//        LogUtils.d(self.TAG, "ImageQualityChecker resources released")
//    }
//    
//    @objc public func updateFaceDetector() {
//        let options = FaceDetectorOptions()
//        options.performanceMode = faceDetectionPerformanceMode
//        options.minFaceSize = CGFloat(Float(minFaceSize))
//        options.landmarkMode = .none
//        options.classificationMode = .none
//        self.faceDetector = MLKitFaceDetection.FaceDetector.faceDetector(options: options)
//        LogUtils.d(self.TAG, "Face detector settings updated")
//    }
//
//    // MARK: - Private Methods
//    private func checkBrightness(_ image: UIImage) -> Float {
//        let avgBrightness = BitmapUtils.calculateAverageBrightness(image)
//        LogUtils.d(self.TAG, "Average brightness: \(avgBrightness)")
//
//        let score: Float
//        switch avgBrightness {
//        case ..<brightnessTooLow:
//            score = avgBrightness / brightnessTooLow
//        case brightnessTooLow..<brightnessSomewhatLow:
//            score = 0.5 + (avgBrightness - brightnessTooLow) / (brightnessSomewhatLow - brightnessTooLow)
//        case brightnessSomewhatLow..<brightnessGoodUpper:
//            score = 1.0
//        case brightnessGoodUpper..<brightnessSomewhatHigh:
//            score = 1.0 - (avgBrightness - brightnessGoodUpper) / (brightnessSomewhatHigh - brightnessGoodUpper)
//        default:
//            score = 0.5 - (avgBrightness - brightnessSomewhatHigh) / (255.0 - brightnessSomewhatHigh)
//        }
//
//        return max(0.0, min(1.0, score))
//    }
//
//    private func checkSharpness(_ image: UIImage) -> Float {
//        guard let cgImage = image.cgImage else {
//            LogUtils.w(self.TAG, "No CGImage available for sharpness check")
//            return 0.5
//        }
//
//        let width = cgImage.width
//        let height = cgImage.height
//
//        if width < 10 || height < 10 {
//            LogUtils.w(self.TAG, "Image too small for reliable sharpness calculation: \(width)x\(height)")
//            return 0.5
//        }
//
//        let avgGrad = calculateLaplacianVariance(image)
//        LogUtils.d(self.TAG, "Laplacian variance (sharpness): \(avgGrad)")
//
//        let score: Float
//        switch avgGrad {
//        case ..<sharpnessVeryBlurry:
//            score = avgGrad / sharpnessVeryBlurry
//        case sharpnessVeryBlurry..<sharpnessSomewhatBlurry:
//            score = 0.5 + (avgGrad - sharpnessVeryBlurry) / (sharpnessSomewhatBlurry - sharpnessVeryBlurry)
//        case sharpnessSomewhatBlurry..<sharpnessGoodUpper:
//            score = 1.0
//        case sharpnessGoodUpper..<sharpnessTooDetailed:
//            score = 1.0 - (avgGrad - sharpnessGoodUpper) / (sharpnessTooDetailed - sharpnessGoodUpper)
//        default:
//            score = 0.5
//        }
//
//        return max(0.0, min(1.0, score))
//    }
//
//    private func calculateLaplacianVariance(_ image: UIImage) -> Float {
//        guard let cgImage = image.cgImage else {
//            LogUtils.w(self.TAG, "No CGImage for Laplacian variance calculation")
//            return 0.0
//        }
//
//        let ciImage = CIImage(cgImage: cgImage)
//        let context = CIContext(options: nil)
//
//        guard let edgeFilter = CIFilter(name: "CIEdges") else {
//            LogUtils.e(self.TAG, "Failed to create CIEdges filter")
//            return 0.0
//        }
//
//        edgeFilter.setValue(ciImage, forKey: kCIInputImageKey)
//
//        guard let outputImage = edgeFilter.outputImage,
//              let outputCGImage = context.createCGImage(outputImage, from: outputImage.extent) else {
//            LogUtils.e(self.TAG, "Failed to process edge filter output")
//            return 0.0
//        }
//
//        let edgeUIImage = UIImage(cgImage: outputCGImage)
//        return BitmapUtils.calculateAverageBrightness(edgeUIImage) * 0.5
//    }
//
//    private func checkFacePresence(_ image: UIImage, completion: @escaping (Bool, Error?) -> Void) {
//        let visionImage = VisionImage(image: image)
//        visionImage.orientation = image.imageOrientation
//        faceDetector.process(visionImage) { faces, error in
//            if let error = error {
//                LogUtils.e(self.TAG, "MLKit face detection failed: \(error.localizedDescription)")
//                completion(false, FaceDetectionException("MLKit face detection failed: \(error.localizedDescription)", error))
//                return
//            }
//
//            let faceDetected = !(faces?.isEmpty ?? true)
//            LogUtils.d(self.TAG, "Face detection result: \(faceDetected ? "Face detected" : "No face detected")")
//            completion(faceDetected, nil)
//        }
//    }
//
//    // MARK: - Deinitialization
//    deinit {
//        close()
//    }
//}
//
