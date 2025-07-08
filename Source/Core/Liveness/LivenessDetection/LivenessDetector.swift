import Foundation
import UIKit
import onnxruntime_objc

public class LivenessDetector: NSObject, AutoCloseable {
    private let TAG = "LivenessDetector"

    private struct Constants {
        static let model1Path = "shufflenetv2_liveness_1.0x.onnx.enc" // Updated to .enc
        static let model2Path = "shufflenetv2_liveness_0.5x.onnx.enc" // Updated to .enc
        static let inputSize = 224
        static let batchSize = 1
        static let rgbChannels = 3
    }

    private let mean: [Float] = [0.485, 0.456, 0.406]
    private let std: [Float] = [0.229, 0.224, 0.225]
    private let threshold: Float
    private let modelWeights: (Float, Float)
    private var ortEnv: ORTEnv?
    private var session1: ORTSession?
    private var session2: ORTSession?
    private var inputName1: String = ""
    private var inputName2: String = ""
    private var isModelLoaded: Bool = false

    private var inputBuffer: UnsafeMutableBufferPointer<Float>
    private let bufferLock = NSLock()
    private var inputFloatArray: [Float]

    public struct LivenessScores {
        let liveScore: Float
        let spoofScore: Float
    }

    public init(bundle: Bundle = .main, threshold: Float = 0.75, modelWeights: (Float, Float) = (0.5, 0.5)) throws {
        self.threshold = threshold
        self.modelWeights = modelWeights
        let bufferSize = Constants.batchSize * Constants.rgbChannels * Constants.inputSize * Constants.inputSize
        self.inputBuffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: bufferSize)
        self.inputBuffer.initialize(repeating: 0.0)
        self.inputFloatArray = [Float](repeating: 0.0, count: bufferSize)
        super.init()
        
        try loadModels(bundle: bundle)
        LogUtils.d(TAG, "Liveness models loaded. Model1 input: \(inputName1), Model2 input: \(inputName2)")
    }

    deinit {
        close()
        inputBuffer.deallocate()
    }

    public func getDetailedScores(bitmap: UIImage) throws -> [String: Float] {
        guard isModelLoaded else {
            throw LivenessException("LivenessDetector initialization failed")
        }
        guard BitmapUtils.validateImage(bitmap) else {
            throw LivenessException("Cannot process invalid bitmap")
        }
        let (_, scores) = try runInference(bitmap: bitmap)
        return [
            "liveScore": scores.liveScore,
            "spoofScore": scores.spoofScore
        ]
    }

    public func runInference(bitmap: UIImage) throws -> (DetectionResult, LivenessScores) {
        guard isModelLoaded else {
            throw LivenessException("LivenessDetector initialization failed")
        }
        guard BitmapUtils.validateImage(bitmap) else {
            throw LivenessException("Cannot process invalid bitmap")
        }

        LogUtils.d(TAG, "Starting combined liveness inference on image: \(bitmap.size.width)x\(bitmap.size.height)")

        let (_, model1Scores) = try runInferenceInternal(bitmap, session: session1, inputName: inputName1)
        let (_, model2Scores) = try runInferenceInternal(bitmap, session: session2, inputName: inputName2)

        let combinedLiveScore = model1Scores.liveScore * modelWeights.0 + model2Scores.liveScore * modelWeights.1
        let combinedSpoofScore = model1Scores.spoofScore * modelWeights.0 + model2Scores.spoofScore * modelWeights.1

        let finalLabel: String
        let finalConfidence: Float

        if combinedLiveScore > threshold {
            finalLabel = "Live"
            finalConfidence = combinedLiveScore
        } else {
            finalLabel = "Spoof"
            finalConfidence = combinedSpoofScore
        }

        LogUtils.i(TAG, "Combined liveness scores - Model1 Live: \(String(format: "%.4f", model1Scores.liveScore)), Model1 Spoof: \(String(format: "%.4f", model1Scores.spoofScore)), Model2 Live: \(String(format: "%.4f", model2Scores.liveScore)), Model2 Spoof: \(String(format: "%.4f", model2Scores.spoofScore)), Final Live: \(String(format: "%.4f", combinedLiveScore)), Final Spoof: \(String(format: "%.4f", combinedSpoofScore))")

        return (DetectionResult(label: finalLabel, confidence: finalConfidence), LivenessScores(liveScore: combinedLiveScore, spoofScore: combinedSpoofScore))
    }

    public func runInferenceWithAveraging(bitmap: UIImage, iterations: Int = 3) throws -> DetectionResult {
        guard isModelLoaded else {
            throw LivenessException("LivenessDetector initialization failed")
        }
        guard BitmapUtils.validateImage(bitmap) else {
            throw LivenessException("Cannot process invalid bitmap")
        }

        var totalModel1LiveScore: Float = 0
        var totalModel1SpoofScore: Float = 0
        var totalModel2LiveScore: Float = 0
        var totalModel2SpoofScore: Float = 0
        var votes: [String: Int] = [:]
        var individualLogs: [String] = []

        for i in 0..<iterations {
            let (_, model1Scores) = try runInferenceInternal(bitmap, session: session1, inputName: inputName1)
            let (_, model2Scores) = try runInferenceInternal(bitmap, session: session2, inputName: inputName2)

            totalModel1LiveScore += model1Scores.liveScore
            totalModel1SpoofScore += model1Scores.spoofScore
            totalModel2LiveScore += model2Scores.liveScore
            totalModel2SpoofScore += model2Scores.spoofScore

            let combinedLiveScore = model1Scores.liveScore * modelWeights.0 + model2Scores.liveScore * modelWeights.1
            let combinedSpoofScore = model1Scores.spoofScore * modelWeights.0 + model2Scores.spoofScore * modelWeights.1

            let finalLabel = combinedLiveScore > threshold ? "Live" : "Spoof"
            votes[finalLabel] = (votes[finalLabel] ?? 0) + 1

            let log = "Liveness run #\(i + 1): \(finalLabel) (Model1 Live: \(String(format: "%.4f", model1Scores.liveScore)), Model1 Spoof: \(String(format: "%.4f", model1Scores.spoofScore)), Model2 Live: \(String(format: "%.4f", model2Scores.liveScore)), Model2 Spoof: \(String(format: "%.4f", model2Scores.spoofScore)), Combined Live: \(String(format: "%.4f", combinedLiveScore)), Combined Spoof: \(String(format: "%.4f", combinedSpoofScore)))"
            individualLogs.append(log)
            LogUtils.i(TAG, log)
        }

        let avgModel1LiveScore = totalModel1LiveScore / Float(iterations)
        let avgModel1SpoofScore = totalModel1SpoofScore / Float(iterations)
        let avgModel2LiveScore = totalModel2LiveScore / Float(iterations)
        let avgModel2SpoofScore = totalModel2SpoofScore / Float(iterations)

        let finalCombinedLiveScore = avgModel1LiveScore * modelWeights.0 + avgModel2LiveScore * modelWeights.1
        let finalCombinedSpoofScore = avgModel1SpoofScore * modelWeights.0 + avgModel2SpoofScore * modelWeights.1

        let finalLabel = votes.max(by: { $0.value < $1.value })?.key ?? "Unknown"
        let finalConfidence = finalLabel == "Live" ? finalCombinedLiveScore : finalCombinedSpoofScore

        LogUtils.i(TAG, "Liveness detection - Average scores after \(iterations) iterations:")
        LogUtils.i(TAG, "- Model1 Live: \(String(format: "%.4f", avgModel1LiveScore))")
        LogUtils.i(TAG, "- Model1 Spoof: \(String(format: "%.4f", avgModel1SpoofScore))")
        LogUtils.i(TAG, "- Model2 Live: \(String(format: "%.4f", avgModel2LiveScore))")
        LogUtils.i(TAG, "- Model2 Spoof: \(String(format: "%.4f", avgModel2SpoofScore))")
        LogUtils.i(TAG, "- Combined Live: \(String(format: "%.4f", finalCombinedLiveScore))")
        LogUtils.i(TAG, "- Combined Spoof: \(String(format: "%.4f", finalCombinedSpoofScore))")
        LogUtils.i(TAG, "- Final label: \(finalLabel) with majority votes: \(votes[finalLabel] ?? 0)/\(iterations)")

        return DetectionResult(label: finalLabel, confidence: finalConfidence)
    }

    private func runInferenceInternal(_ bitmap: UIImage, session: ORTSession?, inputName: String) throws -> (DetectionResult, LivenessScores) {
        guard let session = session else {
            throw LivenessException("Model not initialized")
        }

        preprocessImage(bitmap)

        bufferLock.lock()
        defer { bufferLock.unlock() }
        
        // Transfer preprocessed data to input array
        inputFloatArray = Array(inputBuffer)

        // Prepare input tensor shape
        let shape = [NSNumber(value: Constants.batchSize),
                     NSNumber(value: Constants.rgbChannels),
                     NSNumber(value: Constants.inputSize),
                     NSNumber(value: Constants.inputSize)]
        
        // Create input tensor
        guard let inputTensor = try? ORTValue(
            tensorData: NSMutableData(bytes: inputFloatArray, length: inputFloatArray.count * MemoryLayout<Float>.size),
            elementType: .float,
            shape: shape
        ) else {
            throw LivenessException("Failed to create input tensor")
        }

        // Get the output name for the model
        let outputName: String
        do {
            outputName = try session.outputNames().first ?? "output"
        } catch {
            throw LivenessException("Failed to get model output names", error)
        }
        
        // Run inference
        let outputs: [String: ORTValue]
        do {
            outputs = try session.run(withInputs: [inputName: inputTensor],
                                     outputNames: [outputName],
                                     runOptions: nil)
        } catch {
            throw LivenessException("Error during liveness detection: \(error.localizedDescription)", error)
        }

        // Process output tensor
        guard let outputTensor = outputs[outputName] else {
            throw LivenessException("Missing output tensor")
        }
        
        do {
            guard let outputData = try outputTensor.tensorData() as? NSData,
                  outputData.length / MemoryLayout<Float>.size == 2 else {
                throw LivenessException("Invalid output tensor format")
            }
            
            let outputBuffer = outputData.bytes.assumingMemoryBound(to: Float.self)
            let probabilities = Array(UnsafeBufferPointer(start: outputBuffer, count: 2))
            let liveScore = probabilities[0]
            let spoofScore = probabilities[1]

            let label = liveScore > threshold ? "Live" : "Spoof"
            let confidence = liveScore > threshold ? liveScore : spoofScore

            return (DetectionResult(label: label, confidence: confidence),
                    LivenessScores(liveScore: liveScore, spoofScore: spoofScore))
        } catch {
            throw LivenessException("Failed to process output tensor data", error)
        }
    }

    private func preprocessImage(_ bitmap: UIImage) {
        bufferLock.lock()
        defer { bufferLock.unlock() }

        guard let resizedImage = BitmapUtils.resizeImage(bitmap, width: Constants.inputSize, height: Constants.inputSize),
              let cgImage = resizedImage.cgImage else {
            LogUtils.e(TAG, "Failed to resize image or get CGImage")
            return
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: Constants.inputSize,
            height: Constants.inputSize,
            bitsPerComponent: 8,
            bytesPerRow: Constants.inputSize * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            LogUtils.e(TAG, "Failed to create CGContext for image resizing")
            return
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: Constants.inputSize, height: Constants.inputSize))
        guard let pixelBuffer = context.data else {
            LogUtils.e(TAG, "Failed to get pixel buffer")
            return
        }

        let pixelData = pixelBuffer.assumingMemoryBound(to: UInt8.self)
        var index = 0

        for channel in 0..<Constants.rgbChannels {
            for i in 0..<(Constants.inputSize * Constants.inputSize) {
                let pixelIndex = i * 4
                let value: Float
                switch channel {
                case 0: // R
                    value = Float(pixelData[pixelIndex]) / 255.0
                case 1: // G
                    value = Float(pixelData[pixelIndex + 1]) / 255.0
                default: // B
                    value = Float(pixelData[pixelIndex + 2]) / 255.0
                }
                inputBuffer[index] = (value - mean[channel]) / std[channel]
                index += 1
            }
        }

        let expectedFloats = Constants.rgbChannels * Constants.inputSize * Constants.inputSize
        if index != expectedFloats {
            LogUtils.e(TAG, "Buffer write count mismatch: \(index) vs \(expectedFloats)")
        } else {
            LogUtils.d(TAG, "Image preprocessing completed. Floats written: \(index)")
        }
    }

    public func close() {
        session1 = nil
        session2 = nil
        ortEnv = nil
        isModelLoaded = false
        LogUtils.d(TAG, "LivenessDetector resources released")
    }

    private func loadModels(bundle: Bundle) throws {
        ortEnv = try ORTEnv(loggingLevel: .warning)
        let sessionOptions = try ORTSessionOptions()
        try sessionOptions.setIntraOpNumThreads(1)
        try sessionOptions.setGraphOptimizationLevel(.all)

        // Load model 1
        let model1Data = try ModelUtils.loadModelDataFromBundle(bundle, modelName: Constants.model1Path)
        let tempModel1URL = try saveTempModel(data: model1Data, name: "shufflenetv2_liveness_1.0x.onnx")
        defer { try? FileManager.default.removeItem(at: tempModel1URL) }
        session1 = try ORTSession(env: ortEnv!, modelPath: tempModel1URL.path, sessionOptions: sessionOptions)
        guard let inputNames1 = try? session1?.inputNames(), let firstInputName1 = inputNames1.first else {
            throw LivenessException("No input name found for model 1")
        }
        inputName1 = firstInputName1
        LogUtils.d(TAG, "Model 1 loaded successfully from \(tempModel1URL.path)")

        // Load model 2
        let model2Data = try ModelUtils.loadModelDataFromBundle(bundle, modelName: Constants.model2Path)
        let tempModel2URL = try saveTempModel(data: model2Data, name: "shufflenetv2_liveness_0.5x.onnx")
        defer { try? FileManager.default.removeItem(at: tempModel2URL) }
        session2 = try ORTSession(env: ortEnv!, modelPath: tempModel2URL.path, sessionOptions: sessionOptions)
        guard let inputNames2 = try? session2?.inputNames(), let firstInputName2 = inputNames2.first else {
            throw LivenessException("No input name found for model 2")
        }
        inputName2 = firstInputName2
        LogUtils.d(TAG, "Model 2 loaded successfully from \(tempModel2URL.path)")

        isModelLoaded = true
    }

    private func saveTempModel(data: Data, name: String) throws -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let tempURL = tempDir.appendingPathComponent(name)
        try data.write(to: tempURL)
        return tempURL
    }
}
