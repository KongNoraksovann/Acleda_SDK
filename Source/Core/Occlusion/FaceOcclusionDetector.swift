import Foundation
import UIKit
import onnxruntime_objc

public class FaceOcclusionDetector: NSObject {
    private let tag = "FaceOcclusionDetector"
    public static let modelName = "occlusion_model_epoch60.onnx.enc"
    public static let imageSize = 224
    public static let defaultIterations = 3
    public static let confidenceThreshold: Float = 0.7
    public static let classNames = [0: "occluded", 1: "normal"]
    
    private var ortSession: ORTSession?
    private var ortEnv: ORTEnv?
    private let isModelLoaded = AtomicBoolean(false)
    private let bundle: Bundle
    
    public init(bundle: Bundle = .main) throws {
        self.bundle = bundle
        super.init()
        LogUtils.d(tag, "Initializing with bundle: \(bundle.bundlePath)")
        try loadModel()
    }
    
    private func loadModel() throws {
        ortEnv = try ORTEnv(loggingLevel: .warning)
        
        let modelData = try loadAndDecryptModel(resource: Self.modelName)
        let tempModelURL = try saveTempModel(data: modelData, name: "occlusion_model.onnx")
        defer { try? FileManager.default.removeItem(at: tempModelURL) }
        
        let sessionOptions = try ORTSessionOptions()
        try sessionOptions.setIntraOpNumThreads(1)
        try sessionOptions.setGraphOptimizationLevel(.all)
        
        ortSession = try ORTSession(
            env: ortEnv!,
            modelPath: tempModelURL.path,
            sessionOptions: sessionOptions
        )
        
        isModelLoaded.set(true)
    }
    
    public func getDetailedScores(bitmap: UIImage) throws -> [String: Float] {
        let probabilities = try runInference(on: bitmap)
        return ["occludedScore": probabilities[0], "normalScore": probabilities[1]]
    }
    
    public func detectFaceMask(bitmap: UIImage, threshold: Float = confidenceThreshold) throws -> DetectionResult {
        let probabilities = try runInference(on: bitmap)
        
        let normalProb = probabilities[1]
        let occludedProb = probabilities[0]
        let label = normalProb > threshold ? "normal" : "occluded"
        
        LogUtils.i(tag, "Occlusion scores - Occluded: \(occludedProb.formatted), Normal: \(normalProb.formatted)")
        LogUtils.i(tag, "Predicted class: \(label) with confidence: \(normalProb.formatted)")
        
        return DetectionResult(label: label, confidence: normalProb)
    }
    
    public func detectFaceMaskWithAveraging(
        bitmap: UIImage,
        iterations: Int = defaultIterations,
        threshold: Float = confidenceThreshold
    ) throws -> DetectionResult {
        guard validateBitmap(bitmap) else {
            throw OcclusionDetectionException("Cannot process invalid image")
        }
        
        if !isModelLoaded.get() || ortSession == nil {
            return DetectionResult(label: "normal", confidence: 1.0)
        }
        
        var totalProbs: [Float] = [0.0, 0.0]
        
        for i in 0..<iterations {
            let detailedScores = try getDetailedScores(bitmap: bitmap)
            let occludedScore = detailedScores["occludedScore"] ?? 0.0
            let normalScore = detailedScores["normalScore"] ?? 0.0
            
            totalProbs[0] += occludedScore
            totalProbs[1] += normalScore
            
            LogUtils.i(tag, "Occlusion run #\(i + 1): \(normalScore > threshold ? "normal" : "occluded") (Occluded: \(occludedScore.formatted), Normal: \(normalScore.formatted))")
        }
        
        let avgProbs = totalProbs.map { $0 / Float(iterations) }
        let finalLabel = avgProbs[1] > threshold ? "normal" : "occluded"
        let finalConfidence = avgProbs[1]
        
        LogUtils.i(tag, "Average scores: Occluded: \(avgProbs[0].formatted), Normal: \(avgProbs[1].formatted), Label: \(finalLabel)")
        
        return DetectionResult(label: finalLabel, confidence: finalConfidence)
    }
    
    @objc public func reloadModel() -> Bool {
        if isModelLoaded.get() { return true }
        
        ortSession = nil
        do {
            try loadModel()
            return isModelLoaded.get()
        } catch {
            LogUtils.e(tag, "Model reload failed: \(error.localizedDescription)")
            return false
        }
    }
    
    @objc public func close() {
        ortSession = nil
        isModelLoaded.set(false)
    }
    
    deinit { close() }
    
    private func runInference(on bitmap: UIImage) throws -> [Float] {
        guard let session = ortSession else {
            throw OcclusionDetectionException("Session is null")
        }
        
        guard validateBitmap(bitmap) else {
            throw OcclusionDetectionException("Cannot process invalid image")
        }
        
        let inputTensor = try preprocessImage(bitmap: bitmap)
        
        do {
            let inputNames = ["input"]
            let outputNames: Set<String> = [try session.outputNames().first ?? "output"]
            
            let outputs = try session.run(
                withInputs: [inputNames[0]: inputTensor],
                outputNames: outputNames,
                runOptions: nil
            )
            
            guard let outputTensor = outputs[outputNames.first!],
                  let tensorData = try outputTensor.tensorData() as? NSData else {
                throw OcclusionDetectionException("Failed to extract tensor data")
            }
            
            let floatBuffer = tensorData.bytes.assumingMemoryBound(to: Float.self)
            let logits = Array(UnsafeBufferPointer(start: floatBuffer, count: 2))
            
            return softmax(logits: logits)
        } catch {
            LogUtils.e(tag, "Error during inference: \(error.localizedDescription)", error)
            throw OcclusionDetectionException("Error during inference: \(error.localizedDescription)")
        }
    }
    
    private func preprocessImage(bitmap: UIImage) throws -> ORTValue {
        guard let resizedImage = bitmap.resize(to: CGSize(width: Self.imageSize, height: Self.imageSize)),
              let cgImage = resizedImage.cgImage else {
            throw OcclusionDetectionException("Failed to process image")
        }
        
        let width = Self.imageSize
        let height = Self.imageSize
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        
        guard let context = CGContext(
            data: &pixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: 4 * width,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw OcclusionDetectionException("Failed to create context")
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        let mean: [Float] = [0.485, 0.456, 0.406]
        let std: [Float] = [0.229, 0.224, 0.225]
        var floatPixels = [Float](repeating: 0.0, count: 3 * width * height)
        
        for i in 0..<width * height {
            let pixelOffset = i * 4
            let channels = pixels[pixelOffset..<(pixelOffset + 3)].map { Float($0) / 255.0 }
            
            for c in 0..<3 {
                floatPixels[i + c * width * height] = (channels[c] - mean[c]) / std[c]
            }
        }
        
        let shape: [NSNumber] = [1, 3, NSNumber(value: width), NSNumber(value: height)]
        let tensorData = NSMutableData(bytes: &floatPixels, length: floatPixels.count * MemoryLayout<Float>.size)
        
        return try ORTValue(
            tensorData: tensorData,
            elementType: .float,
            shape: shape
        )
    }
    
    private func loadAndDecryptModel(resource: String) throws -> Data {
        guard let url = bundle.url(forResource: resource, withExtension: nil),
              FileManager.default.fileExists(atPath: url.path) else {
            throw OcclusionDetectionException("Model file \(resource) not found")
        }
        return try ModelUtils.loadModelDataFromBundle(bundle, modelName: resource)
    }
    
    private func saveTempModel(data: Data, name: String) throws -> URL {
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(name)
        try data.write(to: tempURL)
        return tempURL
    }
    
    private func softmax(logits: [Float]) -> [Float] {
        let expValues = logits.map { expf($0) }
        let sumExp = expValues.reduce(0, +)
        return expValues.map { $0 / sumExp }
    }
    
    private func validateBitmap(_ image: UIImage) -> Bool {
        guard let cgImage = image.cgImage, cgImage.width > 0, cgImage.height > 0 else {
            return false
        }
        return true
    }
}

private class AtomicBoolean {
    private var value: Bool
    private let lock = NSLock()
    
    init(_ initialValue: Bool) {
        self.value = initialValue
    }
    
    func get() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
    
    func set(_ newValue: Bool) {
        lock.lock()
        defer { lock.unlock() }
        value = newValue
    }
}

extension UIImage {
    func resize(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        defer { UIGraphicsEndImageContext() }
        draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }
}

private extension Float {
    var formatted: String {
        return String(format: "%.4f", self)
    }
}
