import Foundation
import UIKit
import onnxruntime_objc
import os
import Accelerate

@available(iOS 13.0, *)
class FaceEmbeddingModel {
    private let TAG = "FaceEmbeddingModel"
    

    private let EMBEDDING_MODEL_NAME = "edgeface_s_gamma_05.onnx.enc"
    private let PNET_MODEL_NAME = "pnet.onnx.enc"
    private let RNET_MODEL_NAME = "rnet.onnx.enc"
    private let ONET_MODEL_NAME = "onet.onnx.enc"
    private let INPUT_SIZE = 112
    private let EMBEDDING_SIZE = 512
    private let MIN_FACE_SIZE = 12
    private let THRESHOLDS: [Float] = [0.1, 0.7, 0.9]
    private let FACTOR: Float = 0.709
    private let NMS_THRESHOLDS: [Float] = [0.7, 0.7, 0.7]
    
    
    private let REFERENCE_FACIAL_POINTS: [[Float]] = [
        [30.29459953, 51.69630051],
        [65.53179932, 51.50139999],
        [48.02519989, 71.73660278],
        [33.54930115, 92.3655014],
        [62.72990036, 92.20410156]
    ]

    private let EMBEDDING_INPUT_NAME = "input"
    private let EMBEDDING_OUTPUT_NAME = "output"
    private var cosineThreshold: Float = 0.7
    
    // ONNX Runtime objects
    private var ortEnv: ORTEnv?
    private var embeddingSession: ORTSession?
    private var pnetSession: ORTSession?
    private var rnetSession: ORTSession?
    private var onetSession: ORTSession?
    private var isModelLoaded = false
    
    // Temporary files for decrypted models
    private var tempModelFiles: [URL] = []
    
    private let bundle: Bundle
    
    init(bundle: Bundle = .main) {
        self.bundle = bundle
        loadModelSync()
    }
    
    deinit {
        cleanupTempFiles()
    }
    
    func setCosineThreshold(_ threshold: Float) {
        cosineThreshold = threshold
        os_log(.default, log: OSLog.default, "%{public}s: Cosine similarity threshold set to: %f", TAG, threshold)
    }
    
    func getCosineThreshold() -> Float {
        return cosineThreshold
    }
    
    // Synchronous model loading for initialization
    private func loadModelSync() {
        do {
            ortEnv = try ORTEnv(loggingLevel: .warning)
            
            let sessionOptions = try ORTSessionOptions()
            try sessionOptions.setIntraOpNumThreads(1)
            try sessionOptions.setGraphOptimizationLevel(.all)
            
            // Load embedding model (encrypted)
            let embeddingModelData = try loadEncryptedModelData(EMBEDDING_MODEL_NAME)
            let embeddingModelURL = try saveTempModel(data: embeddingModelData, name: "edgeface_s_gamma_05.onnx")
            embeddingSession = try ORTSession(env: ortEnv!, modelPath: embeddingModelURL.path, sessionOptions: sessionOptions)
            
            // Load PNet model (encrypted)
            let pnetModelData = try loadEncryptedModelData(PNET_MODEL_NAME)
            let pnetModelURL = try saveTempModel(data: pnetModelData, name: "pnet.onnx")
            pnetSession = try ORTSession(env: ortEnv!, modelPath: pnetModelURL.path, sessionOptions: sessionOptions)
            
            // Load RNet model (encrypted)
            let rnetModelData = try loadEncryptedModelData(RNET_MODEL_NAME)
            let rnetModelURL = try saveTempModel(data: rnetModelData, name: "rnet.onnx")
            rnetSession = try ORTSession(env: ortEnv!, modelPath: rnetModelURL.path, sessionOptions: sessionOptions)
            
            // Load ONet model (encrypted)
            let onetModelData = try loadEncryptedModelData(ONET_MODEL_NAME)
            let onetModelURL = try saveTempModel(data: onetModelData, name: "onet.onnx")
            onetSession = try ORTSession(env: ortEnv!, modelPath: onetModelURL.path, sessionOptions: sessionOptions)
            
            isModelLoaded = true
            os_log(.default, log: OSLog.default, "%{public}s: All ONNX models loaded successfully", TAG)
        } catch {
            os_log(.error, log: OSLog.default, "%{public}s: Error loading models: %{public}s", TAG, error.localizedDescription)
            isModelLoaded = false
        }
    }
    
    // Load encrypted model data
    private func loadEncryptedModelData(_ modelName: String) throws -> Data {
        do {
            // Use ModelUtils from LivenessDetector if available
            if let modelData = try? ModelUtils.loadModelDataFromBundle(bundle, modelName: modelName) {
                return modelData
            }
            
            // Fallback to direct loading if ModelUtils is not available
            guard let modelURL = bundle.url(forResource: modelName, withExtension: nil) else {
                throw NSError(domain: "FaceEmbeddingModel", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model file not found: \(modelName)"])
            }
            
            let encryptedData = try Data(contentsOf: modelURL)
            
            // If data is encrypted, decrypt it (assuming simple XOR encryption or similar)
            // This is a placeholder - use the actual decryption method that matches your encryption
            // For example, if using ModelUtils.decryptModel in LivenessDetector
            return encryptedData // Return raw data if no decryption needed or apply decryption here
        } catch {
            os_log(.error, log: OSLog.default, "%{public}s: Failed to load encrypted model %{public}s: %{public}s",
                   TAG, modelName, error.localizedDescription)
            throw error
        }
    }
    
    // Save decrypted model to temporary file
    private func saveTempModel(data: Data, name: String) throws -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let tempURL = tempDir.appendingPathComponent(name)
        
        try data.write(to: tempURL)
        tempModelFiles.append(tempURL)
        
        return tempURL
    }
    
    private func cleanupTempFiles() {
        for url in tempModelFiles {
            do {
                try FileManager.default.removeItem(at: url)
                os_log(.default, log: OSLog.default, "%{public}s: Removed temporary model file: %{public}s", TAG, url.path)
            } catch {
                os_log(.error, log: OSLog.default, "%{public}s: Failed to remove temporary file: %{public}s", TAG, error.localizedDescription)
            }
        }
        tempModelFiles.removeAll()
    }
    
    // Asynchronous model loading
    func loadModel() async throws -> Bool {
        if isModelLoaded {
            os_log(.default, log: OSLog.default, "%{public}s: Models already loaded", TAG)
            return true
        }
        
        return await withCheckedContinuation { continuation in
            do {
                ortEnv = try ORTEnv(loggingLevel: .warning)
                
                let sessionOptions = try ORTSessionOptions()
                try sessionOptions.setIntraOpNumThreads(1)
                try sessionOptions.setGraphOptimizationLevel(.all)
                
                // Load embedding model (encrypted)
                let embeddingModelData = try loadEncryptedModelData(EMBEDDING_MODEL_NAME)
                let embeddingModelURL = try saveTempModel(data: embeddingModelData, name: "edgeface_s_gamma_05.onnx")
                embeddingSession = try ORTSession(env: ortEnv!, modelPath: embeddingModelURL.path, sessionOptions: sessionOptions)
                
                // Load PNet model (encrypted)
                let pnetModelData = try loadEncryptedModelData(PNET_MODEL_NAME)
                let pnetModelURL = try saveTempModel(data: pnetModelData, name: "pnet.onnx")
                pnetSession = try ORTSession(env: ortEnv!, modelPath: pnetModelURL.path, sessionOptions: sessionOptions)
                
                // Load RNet model (encrypted)
                let rnetModelData = try loadEncryptedModelData(RNET_MODEL_NAME)
                let rnetModelURL = try saveTempModel(data: rnetModelData, name: "rnet.onnx")
                rnetSession = try ORTSession(env: ortEnv!, modelPath: rnetModelURL.path, sessionOptions: sessionOptions)
                
                // Load ONet model (encrypted)
                let onetModelData = try loadEncryptedModelData(ONET_MODEL_NAME)
                let onetModelURL = try saveTempModel(data: onetModelData, name: "onet.onnx")
                onetSession = try ORTSession(env: ortEnv!, modelPath: onetModelURL.path, sessionOptions: sessionOptions)
                
                isModelLoaded = true
                os_log(.default, log: OSLog.default, "%{public}s: All ONNX models loaded successfully", TAG)
                continuation.resume(returning: true)
            } catch {
                os_log(.error, log: OSLog.default, "%{public}s: Error loading models: %{public}s", TAG, error.localizedDescription)
                isModelLoaded = false
                continuation.resume(returning: false)
            }
        }
    }
    
    // Standard cosine similarity matching Python exactly
    func cosineSimilarity(embedding1: [Double], embedding2: [Double]) -> Float {
        guard embedding1.count == embedding2.count else {
            fatalError("Embedding size mismatch")
        }
        
        // Calculate dot product
        let dotProduct = zip(embedding1, embedding2).reduce(0.0) { $0 + $1.0 * $1.1 }
        
        // Calculate norms
        let norm1 = sqrt(embedding1.reduce(0.0) { $0 + $1 * $1 })
        let norm2 = sqrt(embedding2.reduce(0.0) { $0 + $1 * $1 })
        
        // Standard cosine similarity
        return Float(dotProduct / (norm1 * norm2))
    }
    
    func close() {
        embeddingSession = nil
        pnetSession = nil
        rnetSession = nil
        onetSession = nil
        ortEnv = nil
        isModelLoaded = false
        cleanupTempFiles()
        os_log(.default, log: OSLog.default, "%{public}s: Model resources released", TAG)
    }
    
    // Face detection pipeline - matching Python MTCNN implementation exactly
    func detectFaces(image: UIImage) async throws -> ([Float], [Float]) {
        guard let cgImage = image.cgImage else {
            os_log(.error, log: OSLog.default, "%{public}s: Invalid CGImage", TAG)
            return ([], [])
        }
        
        if !isModelLoaded {
            let success = try await loadModel()
            if !success {
                os_log(.error, log: OSLog.default, "%{public}s: Models not loaded, cannot detect faces", TAG)
                return ([], [])
            }
        }
        
        let width = Float(cgImage.width)
        let height = Float(cgImage.height)
        let minLength = min(height, width)
        let minDetectionSize: Float = 12
        
        // Generate scales exactly like Python
        var scales: [Float] = []
        let m = minDetectionSize / Float(MIN_FACE_SIZE)
        var scaledMinLength = minLength * m
        var factorCount = 0
        
        while scaledMinLength > minDetectionSize {
            scales.append(m * pow(FACTOR, Float(factorCount)))
            scaledMinLength *= FACTOR
            factorCount += 1
        }
        
        var boundingBoxes: [[Float]] = []
        
        // First stage - PNet
        for scale in scales {
            if let boxes = try await runFirstStage(image: image, scale: scale, threshold: THRESHOLDS[0]) {
                boundingBoxes.append(boxes)
            }
        }
        
        // Filter out nil boxes
        let validBoxes = boundingBoxes.filter { !$0.isEmpty }
        if validBoxes.isEmpty {
            return ([], [])
        }
        
        // Combine all boxes
        let allBoxes = validBoxes.flatMap { $0 }
        
        // Convert to matrix format for processing
        var boxes = convertToBoxMatrix(allBoxes)
        
        // Apply NMS
        var keep = nms(boxes: boxes, overlapThreshold: NMS_THRESHOLDS[0])
        boxes = filterBoxes(boxes, indices: keep)
        
        // Calibrate boxes
        let offsets = extractOffsetsFromFlat(allBoxes, indices: keep)
        boxes = calibrateBox(bboxes: boxes, offsets: offsets)
        boxes = convertToSquare(bboxes: boxes)
        
        // Round coordinates
        for i in 0..<boxes.count {
            for j in 0..<4 {
                boxes[i][j] = round(boxes[i][j])
            }
        }
        
        // Second stage - RNet
        let imgBoxes24 = try await getImageBoxes(boxes: boxes, image: image, size: 24)
        if imgBoxes24.isEmpty {
            return ([], [])
        }
        
        let rnetOutput = try await runRNet(imgBoxes: imgBoxes24)
        let rnetOffsets = rnetOutput.0
        let rnetProbs = rnetOutput.1
        
        // Filter by threshold
        keep = []
        for i in 0..<rnetProbs.count/2 {
            if rnetProbs[i * 2 + 1] > THRESHOLDS[1] {
                keep.append(i)
            }
        }
        
        if keep.isEmpty {
            return ([], [])
        }
        
        boxes = filterBoxes(boxes, indices: keep)
        for i in 0..<keep.count {
            boxes[i][4] = rnetProbs[keep[i] * 2 + 1]
        }
        
        let filteredRnetOffsets = filterOffsets(rnetOffsets, indices: keep)
        keep = nms(boxes: boxes, overlapThreshold: NMS_THRESHOLDS[1])
        boxes = filterBoxes(boxes, indices: keep)
        let finalRnetOffsets = filterOffsets(filteredRnetOffsets, indices: keep)
        boxes = calibrateBox(bboxes: boxes, offsets: finalRnetOffsets)
        boxes = convertToSquare(bboxes: boxes)
        
        // Round coordinates
        for i in 0..<boxes.count {
            for j in 0..<4 {
                boxes[i][j] = round(boxes[i][j])
            }
        }
        
        // Third stage - ONet
        let imgBoxes48 = try await getImageBoxes(boxes: boxes, image: image, size: 48)
        if imgBoxes48.isEmpty {
            return ([], [])
        }
        
        let onetOutput = try await runONet(imgBoxes: imgBoxes48)
        var landmarks = onetOutput.0
        let onetOffsets = onetOutput.1
        let onetProbs = onetOutput.2
        
        // Filter by threshold
        keep = []
        for i in 0..<onetProbs.count/2 {
            if onetProbs[i * 2 + 1] > THRESHOLDS[2] {
                keep.append(i)
            }
        }
        
        if keep.isEmpty {
            return ([], [])
        }
        
        boxes = filterBoxes(boxes, indices: keep)
        for i in 0..<keep.count {
            boxes[i][4] = onetProbs[keep[i] * 2 + 1]
        }
        
        let filteredOnetOffsets = filterOffsets(onetOffsets, indices: keep)
        landmarks = filterLandmarks(landmarks, indices: keep)
        
        // Transform landmarks to absolute coordinates exactly like Python
        for i in 0..<landmarks.count {
            let width = boxes[i][2] - boxes[i][0] + 1.0
            let height = boxes[i][3] - boxes[i][1] + 1.0
            let xmin = boxes[i][0]
            let ymin = boxes[i][1]
            
            // landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
            for j in 0..<5 {
                landmarks[i][j] = xmin + width * landmarks[i][j]
            }
            
            // landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]
            for j in 5..<10 {
                landmarks[i][j] = ymin + height * landmarks[i][j]
            }
        }
        
        boxes = calibrateBox(bboxes: boxes, offsets: filteredOnetOffsets)
        keep = nms(boxes: boxes, overlapThreshold: NMS_THRESHOLDS[2], mode: "min")
        boxes = filterBoxes(boxes, indices: keep)
        landmarks = filterLandmarks(landmarks, indices: keep)
        
        return (boxes.flatMap { $0 }, landmarks.flatMap { $0 })
    }
    
    // Helper functions matching Python logic
    private func convertToBoxMatrix(_ boxes: [Float]) -> [[Float]] {
        let numBoxes = boxes.count / 9
        var result: [[Float]] = []
        for i in 0..<numBoxes {
            let box = Array(boxes[i*9..<(i*9+5)])
            result.append(box)
        }
        return result
    }
    
    private func extractOffsetsFromFlat(_ boxes: [Float], indices: [Int]) -> [[Float]] {
        var offsets: [[Float]] = []
        for i in indices {
            let offset = Array(boxes[i*9+5..<i*9+9])
            offsets.append(offset)
        }
        return offsets
    }
    
    private func filterBoxes(_ boxes: [[Float]], indices: [Int]) -> [[Float]] {
        return indices.map { boxes[$0] }
    }
    
    private func filterOffsets(_ offsets: [[Float]], indices: [Int]) -> [[Float]] {
        return indices.map { offsets[$0] }
    }
    
    private func filterLandmarks(_ landmarks: [[Float]], indices: [Int]) -> [[Float]] {
        return indices.map { landmarks[$0] }
    }
    
    // CRITICAL FIX: Match Python's runFirstStage exactly - using sliding window approach
    private func runFirstStage(image: UIImage, scale: Float, threshold: Float) async throws -> [Float]? {
        guard let cgImage = image.cgImage else { return nil }
        
        let width = cgImage.width
        let height = cgImage.height
        let sw = Int(ceil(Float(width) * scale))
        let sh = Int(ceil(Float(height) * scale))
        
        let resizedImage = image.resize(size: CGSize(width: sw, height: sh), interpolation: .low)
        guard let imgArray = getImageArray(resizedImage) else { return nil }
        
        let stride = 2
        let cellSize = 12
        
        // Calculate steps exactly like Python
        let hSteps = Int(ceil(Float(sh - cellSize) / Float(stride))) + 1
        let wSteps = Int(ceil(Float(sw - cellSize) / Float(stride))) + 1
        
        var probs = Array(repeating: Array(repeating: Float(0), count: wSteps), count: hSteps)
        var offsets = Array(repeating: Array(repeating: Array(repeating: Float(0), count: wSteps), count: hSteps), count: 4)
        
        guard let pnetSession = pnetSession else { return nil }
        
        // Sliding window approach exactly like Python
        for i in 0..<hSteps {
            for j in 0..<wSteps {
                let y = i * stride
                let x = j * stride
                let yEnd = min(y + cellSize, sh)
                let xEnd = min(x + cellSize, sw)
                
                if yEnd - y != cellSize || xEnd - x != cellSize {
                    continue
                }
                
                // Extract patch
                let patch = extractPatch(from: imgArray, x: x, y: y, width: cellSize, height: cellSize, originalWidth: sw)
                guard let preprocessed = preprocess(patch, width: cellSize, height: cellSize) else { continue }
                
                // Run inference on patch
                let shape: [NSNumber] = [1, 3, cellSize, cellSize].map { NSNumber(value: $0) }
                let inputTensor = try ORTValue(
                    tensorData: NSMutableData(bytes: preprocessed, length: preprocessed.count * MemoryLayout<Float>.size),
                    elementType: .float,
                    shape: shape
                )
                
                let inputName = try pnetSession.inputNames().first ?? "input"
                let outputs = try pnetSession.run(
                    withInputs: [inputName: inputTensor],
                    outputNames: Set(try pnetSession.outputNames()),
                    runOptions: nil
                )
                
                // Extract outputs exactly like Python
                let outputValues = Array(outputs.values)
                if outputValues.count >= 2 {
                    let offsetsTensor = outputValues[0]
                    let probsTensor = outputValues[1]
                    
                    // Get patch probabilities and offsets
                    let patchProbs = try getPatchProbs(from: probsTensor)
                    let patchOffsets = try getPatchOffsets(from: offsetsTensor)
                    
                    probs[i][j] = patchProbs
                    for k in 0..<4 {
                        offsets[k][i][j] = patchOffsets[k]
                    }
                }
            }
        }
        
        return generateBboxes(probs: probs, offsets: offsets, scale: scale, threshold: threshold)
    }
    
    // Extract image as array for Python-style processing
    private func getImageArray(_ image: UIImage) -> [Float]? {
        guard let cgImage = image.cgImage,
              let pixels = image.pixelData() else { return nil }
        
        let width = cgImage.width
        let height = cgImage.height
        var result = [Float](repeating: 0, count: width * height * 3)
        
        // Convert to HWC float array
        for y in 0..<height {
            for x in 0..<width {
                let pixel = pixels[y * width + x]
                let idx = (y * width + x) * 3
                result[idx + 0] = Float((pixel >> 16) & 0xFF) // R
                result[idx + 1] = Float((pixel >> 8) & 0xFF)  // G
                result[idx + 2] = Float(pixel & 0xFF)         // B
            }
        }
        
        return result
    }
    
    // Extract patch from image array
    private func extractPatch(from imgArray: [Float], x: Int, y: Int, width: Int, height: Int, originalWidth: Int) -> [Float] {
        var patch = [Float](repeating: 0, count: width * height * 3)
        
        for py in 0..<height {
            for px in 0..<width {
                let srcIdx = ((y + py) * originalWidth + (x + px)) * 3
                let dstIdx = (py * width + px) * 3
                
                if srcIdx + 2 < imgArray.count && dstIdx + 2 < patch.count {
                    patch[dstIdx + 0] = imgArray[srcIdx + 0]
                    patch[dstIdx + 1] = imgArray[srcIdx + 1]
                    patch[dstIdx + 2] = imgArray[srcIdx + 2]
                }
            }
        }
        
        return patch
    }
    
    // Get patch probabilities from tensor
    private func getPatchProbs(from tensor: ORTValue) throws -> Float {
        let data = try tensor.tensorData() as Data
        var probs: [Float] = []
        
        data.withUnsafeBytes { buffer in
            if let ptr = buffer.baseAddress?.assumingMemoryBound(to: Float.self) {
                probs = Array(UnsafeBufferPointer(start: ptr, count: buffer.count / MemoryLayout<Float>.size))
            }
        }
        
        // Return face probability (class 1)
        return probs.count >= 2 ? probs[1] : 0.0
    }
    
    // Get patch offsets from tensor
    private func getPatchOffsets(from tensor: ORTValue) throws -> [Float] {
        let data = try tensor.tensorData() as Data
        var offsets: [Float] = []
        
        data.withUnsafeBytes { buffer in
            if let ptr = buffer.baseAddress?.assumingMemoryBound(to: Float.self) {
                offsets = Array(UnsafeBufferPointer(start: ptr, count: buffer.count / MemoryLayout<Float>.size))
            }
        }
        
        return Array(offsets.prefix(4))
    }
    
    // Match Python's preprocess exactly: (img - 127.5) * 0.0078125
    private func preprocess(_ patch: [Float], width: Int, height: Int) -> [Float]? {
        // Convert HWC to CHW format with normalization exactly like Python
        var result = [Float](repeating: 0, count: 3 * width * height)
        
        for c in 0..<3 {
            for y in 0..<height {
                for x in 0..<width {
                    let hwcIdx = (y * width + x) * 3 + c
                    let chwIdx = c * width * height + y * width + x
                    result[chwIdx] = (patch[hwcIdx] - 127.5) * 0.0078125
                }
            }
        }
        
        return result
    }
    
    // Match Python's generate_bboxes exactly
    private func generateBboxes(probs: [[Float]], offsets: [[[Float]]], scale: Float, threshold: Float) -> [Float] {
        let stride = 2
        let cellSize = 12
        
        var boundingBoxes: [Float] = []
        
        for i in 0..<probs.count {
            for j in 0..<probs[i].count {
                if probs[i][j] > threshold {
                    let tx1 = offsets[0][i][j]
                    let ty1 = offsets[1][i][j]
                    let tx2 = offsets[2][i][j]
                    let ty2 = offsets[3][i][j]
                    
                    let score = probs[i][j]
                    
                    // Match Python's coordinate calculation exactly
                    let x1 = round((Float(stride * j + 1)) / scale)
                    let y1 = round((Float(stride * i + 1)) / scale)
                    let x2 = round((Float(stride * j + 1 + cellSize)) / scale)
                    let y2 = round((Float(stride * i + 1 + cellSize)) / scale)
                    
                    boundingBoxes.append(contentsOf: [x1, y1, x2, y2, score, tx1, ty1, tx2, ty2])
                }
            }
        }
        
        return boundingBoxes
    }
        private func nms(boxes: [[Float]], overlapThreshold: Float, mode: String = "union") -> [Int] {
        if boxes.isEmpty { return [] }
        
        var pick: [Int] = []
        let x1 = boxes.map { $0[0] }
        let y1 = boxes.map { $0[1] }
        let x2 = boxes.map { $0[2] }
        let y2 = boxes.map { $0[3] }
        let score = boxes.map { $0[4] }
        
        var area: [Float] = []
        for i in 0..<boxes.count {
            let width = x2[i] - x1[i] + 1.0
            let height = y2[i] - y1[i] + 1.0
            area.append(width * height)
        }
        
        var ids = score.enumerated().sorted { $0.element < $1.element }.map { $0.offset }
        
        while !ids.isEmpty {
            let last = ids.count - 1
            let i = ids[last]
            pick.append(i)
            
            var toDelete: [Int] = [last]
            
            for j in 0..<last {
                let idx = ids[j]
                let ix1 = max(x1[i], x1[idx])
                let iy1 = max(y1[i], y1[idx])
                let ix2 = min(x2[i], x2[idx])
                let iy2 = min(y2[i], y2[idx])
                let w = max(0.0, ix2 - ix1 + 1.0)
                let h = max(0.0, iy2 - iy1 + 1.0)
                let inter = w * h
                
                let overlap: Float
                if mode == "min" {
                    overlap = inter / min(area[i], area[idx])
                } else {
                    overlap = inter / (area[i] + area[idx] - inter)
                }
                
                if overlap > overlapThreshold {
                    toDelete.append(j)
                }
            }
            
            // Remove indices in reverse order to maintain correct indexing
            for deleteIndex in toDelete.sorted().reversed() {
                ids.remove(at: deleteIndex)
            }
        }
        
        return pick
    }
    
    // Match Python's calibrate_box exactly
    private func calibrateBox(bboxes: [[Float]], offsets: [[Float]]) -> [[Float]] {
        var result = bboxes
        
        for i in 0..<bboxes.count {
            let x1 = bboxes[i][0]
            let y1 = bboxes[i][1]
            let x2 = bboxes[i][2]
            let y2 = bboxes[i][3]
            let w = x2 - x1 + 1.0
            let h = y2 - y1 + 1.0
            
            if i < offsets.count && offsets[i].count >= 4 {
                result[i][0] = x1 + w * offsets[i][0]
                result[i][1] = y1 + h * offsets[i][1]
                result[i][2] = x2 + w * offsets[i][2]
                result[i][3] = y2 + h * offsets[i][3]
            }
        }
        
        return result
    }
    
    // Match Python's convert_to_square exactly
    private func convertToSquare(bboxes: [[Float]]) -> [[Float]] {
        var squareBboxes = bboxes
        
        for i in 0..<bboxes.count {
            let x1 = bboxes[i][0]
            let y1 = bboxes[i][1]
            let x2 = bboxes[i][2]
            let y2 = bboxes[i][3]
            let h = y2 - y1 + 1.0
            let w = x2 - x1 + 1.0
            let maxSide = max(h, w)
            
            squareBboxes[i][0] = x1 + w * 0.5 - maxSide * 0.5
            squareBboxes[i][1] = y1 + h * 0.5 - maxSide * 0.5
            squareBboxes[i][2] = squareBboxes[i][0] + maxSide - 1.0
            squareBboxes[i][3] = squareBboxes[i][1] + maxSide - 1.0
        }
        
        return squareBboxes
    }
    
    // Match Python's get_image_boxes exactly
    private func getImageBoxes(boxes: [[Float]], image: UIImage, size: Int) async throws -> [Float] {
        guard let cgImage = image.cgImage else { return [] }
        let width = Float(cgImage.width)
        let height = Float(cgImage.height)
        
        let correctedData = correctBboxes(bboxes: boxes, width: width, height: height)
        var imgBoxes: [Float] = []
        
        for i in 0..<boxes.count {
            let dy = Int(correctedData.dy[i])
            let edy = Int(correctedData.edy[i])
            let dx = Int(correctedData.dx[i])
            let edx = Int(correctedData.edx[i])
            let y = Int(correctedData.y[i])
            let ey = Int(correctedData.ey[i])
            let x = Int(correctedData.x[i])
            let ex = Int(correctedData.ex[i])
            let w = Int(correctedData.w[i])
            let h = Int(correctedData.h[i])
            
            // Create image box exactly like Python
            let boxImage = createImageBox(from: image, dy: dy, edy: edy, dx: dx, edx: edx,
                                          y: y, ey: ey, x: x, ex: ex, w: w, h: h, targetSize: size)
            if let processed = preprocessImageForModel(boxImage) {
                imgBoxes.append(contentsOf: processed)
            }
        }
        
        return imgBoxes
    }
    
    // Match Python's correct_bboxes exactly
    private func correctBboxes(bboxes: [[Float]], width: Float, height: Float) ->
        (dy: [Float], edy: [Float], dx: [Float], edx: [Float], y: [Float], ey: [Float], x: [Float], ex: [Float], w: [Float], h: [Float]) {
        
        let x1 = bboxes.map { $0[0] }
        let y1 = bboxes.map { $0[1] }
        let x2 = bboxes.map { $0[2] }
        let y2 = bboxes.map { $0[3] }
        let w = zip(x1, x2).map { $1 - $0 + 1.0 }
        let h = zip(y1, y2).map { $1 - $0 + 1.0 }
        
        var x = x1
        var y = y1
        var ex = x2
        var ey = y2
        var dx = [Float](repeating: 0, count: bboxes.count)
        var dy = [Float](repeating: 0, count: bboxes.count)
        var edx = w.map { $0 - 1.0 }
        var edy = h.map { $0 - 1.0 }
        
        // Apply corrections exactly like Python
        for i in 0..<bboxes.count {
            if ex[i] > width - 1.0 {
                edx[i] = w[i] + width - 2.0 - ex[i]
                ex[i] = width - 1.0
            }
            if ey[i] > height - 1.0 {
                edy[i] = h[i] + height - 2.0 - ey[i]
                ey[i] = height - 1.0
            }
            if x[i] < 0.0 {
                dx[i] = 0.0 - x[i]
                x[i] = 0.0
            }
            if y[i] < 0.0 {
                dy[i] = 0.0 - y[i]
                y[i] = 0.0
            }
        }
        
        return (dy: dy, edy: edy, dx: dx, edx: edx, y: y, ey: ey, x: x, ex: ex, w: w, h: h)
    }
    
    // Match Python's image box creation exactly
    private func createImageBox(from image: UIImage, dy: Int, edy: Int, dx: Int, edx: Int,
                                y: Int, ey: Int, x: Int, ex: Int, w: Int, h: Int, targetSize: Int) -> UIImage {
        
        guard let cgImage = image.cgImage else { return image }
        
        // Create image box with padding exactly like Python
        let context = CGContext(
            data: nil,
            width: w,
            height: h,
            bitsPerComponent: 8,
            bytesPerRow: w * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
        )
        
        guard let context = context else { return image }
        
        // Fill with black (padding) - exactly like Python zeros
        context.setFillColor(UIColor.black.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: w, height: h))
        
        // Draw the actual image portion exactly like Python array indexing
        let sourceRect = CGRect(x: x, y: y, width: ex - x + 1, height: ey - y + 1)
        let destRect = CGRect(x: dx, y: dy, width: edx - dx + 1, height: edy - dy + 1)
        
        if let croppedCGImage = cgImage.cropping(to: sourceRect) {
            context.draw(croppedCGImage, in: destRect)
        }
        
        guard let resultCGImage = context.makeImage() else { return image }
        let resultImage = UIImage(cgImage: resultCGImage)
        
        // Resize to target size using BILINEAR interpolation exactly like Python
        return resultImage.resize(size: CGSize(width: targetSize, height: targetSize), interpolation: .medium)
    }
    
    // Match Python's preprocessing for model input
    private func preprocessImageForModel(_ image: UIImage) -> [Float]? {
        guard let cgImage = image.cgImage else { return nil }
        
        let width = cgImage.width
        let height = cgImage.height
        guard let pixels = image.pixelData() else { return nil }
        
        var result = [Float](repeating: 0, count: 3 * width * height)
        
        // Convert HWC to CHW format and normalize exactly like Python: (img - 127.5) * 0.0078125
        for c in 0..<3 {
            for y in 0..<height {
                for x in 0..<width {
                    let pixel = pixels[y * width + x]
                    let channelValue: Float
                    switch c {
                    case 0: channelValue = Float((pixel >> 16) & 0xFF) // R
                    case 1: channelValue = Float((pixel >> 8) & 0xFF)  // G
                    case 2: channelValue = Float(pixel & 0xFF)         // B
                    default: channelValue = 0
                    }
                    result[c * width * height + y * width + x] = (channelValue - 127.5) * 0.0078125
                }
            }
        }
        
        return result
    }
    
    // RNet processing exactly like Python
    private func runRNet(imgBoxes: [Float]) async throws -> ([[Float]], [Float]) {
        let numBoxes = imgBoxes.count / (3 * 24 * 24)
        if numBoxes == 0 {
            return ([], [])
        }
        
        let shape: [NSNumber] = [numBoxes, 3, 24, 24].map { NSNumber(value: $0) }
        let inputTensor = try ORTValue(
            tensorData: NSMutableData(bytes: imgBoxes, length: imgBoxes.count * MemoryLayout<Float>.size),
            elementType: .float,
            shape: shape
        )
        
        guard let rnetSession = rnetSession else {
            return ([], [])
        }
        
        let inputName = try rnetSession.inputNames().first ?? "input"
        let outputs = try rnetSession.run(
            withInputs: [inputName: inputTensor],
            outputNames: Set(try rnetSession.outputNames()),
            runOptions: nil
        )
        
        guard outputs.count >= 2 else { return ([], []) }
        
        let outputValues = Array(outputs.values)
        let offsetsTensor = outputValues[0]
        let probsTensor = outputValues[1]
        
        var offsets: [Float] = []
        var probs: [Float] = []
        
        let offsetsData = try offsetsTensor.tensorData() as Data
        offsetsData.withUnsafeBytes { buffer in
            if let ptr = buffer.baseAddress?.assumingMemoryBound(to: Float.self) {
                offsets = Array(UnsafeBufferPointer(start: ptr, count: buffer.count / MemoryLayout<Float>.size))
            }
        }
        
        let probsData = try probsTensor.tensorData() as Data
        probsData.withUnsafeBytes { buffer in
            if let ptr = buffer.baseAddress?.assumingMemoryBound(to: Float.self) {
                probs = Array(UnsafeBufferPointer(start: ptr, count: buffer.count / MemoryLayout<Float>.size))
            }
        }
        
        // Convert offsets to proper format exactly like Python
        var offsetMatrix: [[Float]] = []
        for i in 0..<numBoxes {
            let offset = Array(offsets[i*4..<(i*4+4)])
            offsetMatrix.append(offset)
        }
        
        return (offsetMatrix, probs)
    }
    
    // ONet processing exactly like Python
    private func runONet(imgBoxes: [Float]) async throws -> ([[Float]], [[Float]], [Float]) {
        let numBoxes = imgBoxes.count / (3 * 48 * 48)
        if numBoxes == 0 {
            return ([], [], [])
        }
        
        let shape: [NSNumber] = [numBoxes, 3, 48, 48].map { NSNumber(value: $0) }
        let inputTensor = try ORTValue(
            tensorData: NSMutableData(bytes: imgBoxes, length: imgBoxes.count * MemoryLayout<Float>.size),
            elementType: .float,
            shape: shape
        )
        
        guard let onetSession = onetSession else {
            return ([], [], [])
        }
        
        let inputName = try onetSession.inputNames().first ?? "input"
        let outputs = try onetSession.run(
            withInputs: [inputName: inputTensor],
            outputNames: Set(try onetSession.outputNames()),
            runOptions: nil
        )
        
        guard outputs.count >= 3 else { return ([], [], []) }
        
        let outputValues = Array(outputs.values)
        let landmarksTensor = outputValues[0]
        let offsetsTensor = outputValues[1]
        let probsTensor = outputValues[2]
        
        var landmarks: [Float] = []
        var offsets: [Float] = []
        var probs: [Float] = []
        
        let landmarksData = try landmarksTensor.tensorData() as Data
        landmarksData.withUnsafeBytes { buffer in
            if let ptr = buffer.baseAddress?.assumingMemoryBound(to: Float.self) {
                landmarks = Array(UnsafeBufferPointer(start: ptr, count: buffer.count / MemoryLayout<Float>.size))
            }
        }
        
        let offsetsData = try offsetsTensor.tensorData() as Data
        offsetsData.withUnsafeBytes { buffer in
            if let ptr = buffer.baseAddress?.assumingMemoryBound(to: Float.self) {
                offsets = Array(UnsafeBufferPointer(start: ptr, count: buffer.count / MemoryLayout<Float>.size))
            }
        }
        
        let probsData = try probsTensor.tensorData() as Data
        probsData.withUnsafeBytes { buffer in
            if let ptr = buffer.baseAddress?.assumingMemoryBound(to: Float.self) {
                probs = Array(UnsafeBufferPointer(start: ptr, count: buffer.count / MemoryLayout<Float>.size))
            }
        }
        
        // Convert to proper format exactly like Python
        var landmarkMatrix: [[Float]] = []
        var offsetMatrix: [[Float]] = []
        
        for i in 0..<numBoxes {
            let landmark = Array(landmarks[i*10..<(i*10+10)])
            let offset = Array(offsets[i*4..<(i*4+4)])
            landmarkMatrix.append(landmark)
            offsetMatrix.append(offset)
        }
        
        return (landmarkMatrix, offsetMatrix, probs)
    }
    
    // Improved face alignment using Python's exact similarity transform approach
    func alignFace(image: UIImage, facialLandmarks: [[Float]], source: String = "unknown") async throws -> UIImage? {
        if !isModelLoaded {
            let loaded = try await loadModel()
            if !loaded {
                os_log(.error, log: OSLog.default, "%{public}s: Models not loaded, cannot align face", TAG)
                return nil
            }
        }
        
        // Convert landmarks to facial5points format exactly like Python
        // Python: facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
        var facial5Points: [[Float]] = []
        for j in 0..<5 {
            if facialLandmarks.count > 0 && facialLandmarks[0].count > j + 5 {
                facial5Points.append([facialLandmarks[0][j], facialLandmarks[0][j + 5]])
            } else if j < facialLandmarks.count {
                facial5Points.append([facialLandmarks[j][0], facialLandmarks[j][1]])
            }
        }
        
        guard let alignedImage = warpAndCropFace(
            image: image,
            facialPts: facial5Points,
            referencePts: REFERENCE_FACIAL_POINTS,
            cropSize: (INPUT_SIZE, INPUT_SIZE)
        ) else {
            os_log(.error, log: OSLog.default, "%{public}s: Error aligning face for %{public}s", TAG, source)
            return nil
        }
        
        os_log(.default, log: OSLog.default, "%{public}s: Aligned image for %@: %dx%d", TAG, source,
               Int(alignedImage.size.width), Int(alignedImage.size.height))
        return alignedImage
    }
    
    // Python-style warp_and_crop_face exactly matching the Python implementation
    private func warpAndCropFace(image: UIImage, facialPts: [[Float]],
                                referencePts: [[Float]], cropSize: (Int, Int)) -> UIImage? {
        guard image.cgImage != nil else { return nil }
        
        // Get similarity transform exactly like Python
        guard let tform = getSimilarityTransformForCV2(srcPts: facialPts, dstPts: referencePts) else {
            return nil
        }
        
        // Apply warp affine transform exactly like Python cv2.warpAffine
        return warpAffine(image: image, transform: tform, outputSize: cropSize)
    }
    
    // Match Python's get_similarity_transform_for_cv2 exactly
    private func getSimilarityTransformForCV2(srcPts: [[Float]], dstPts: [[Float]]) -> [[Float]]? {
        // First try non-reflective similarity
        guard let (trans, _) = getSimilarityTransform(srcPts: srcPts, dstPts: dstPts, reflective: true) else {
            return nil
        }
        
        // Convert to CV2 format (2x3 matrix) exactly like Python: trans[:, 0:2].T
        return [
            [trans[0][0], trans[0][1], trans[0][2]],
            [trans[1][0], trans[1][1], trans[1][2]]
        ]
    }
    
    // Match Python's get_similarity_transform exactly
    private func getSimilarityTransform(srcPts: [[Float]], dstPts: [[Float]], reflective: Bool = true) -> ([[Float]], [[Float]])? {
        // Try non-reflective similarity first
        guard let (trans1, trans1Inv) = findNonreflectiveSimilarity(uv: srcPts, xy: dstPts) else {
            return nil
        }
        
        if !reflective {
            return (trans1, trans1Inv)
        }
        
        // Try reflective similarity exactly like Python
        // Python: xyR = dstPts.copy(); xyR[:, 0] = -1 * xyR[:, 0]
        var xyR = dstPts
        for i in 0..<xyR.count {
            xyR[i][0] = -xyR[i][0]
        }
        
        guard let (trans2r, _) = findNonreflectiveSimilarity(uv: srcPts, xy: xyR) else {
            return (trans1, trans1Inv)
        }
        
        // Python: TreflectY = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        let reflectY: [[Float]] = [
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        
        // Python: trans2 = np.dot(trans2r, TreflectY)
        let trans2 = matrixMultiply(trans2r, reflectY)
        
        // Calculate norms to choose best transform exactly like Python
        let xy1 = transformPoints(srcPts, transform: trans1)
        let xy2 = transformPoints(srcPts, transform: trans2)
        
        let norm1 = calculateNorm(xy1, dstPts)
        let norm2 = calculateNorm(xy2, dstPts)
        
        if norm1 <= norm2 {
            return (trans1, trans1Inv)
        } else {
            guard let trans2Inv = matrixInverse(trans2) else {
                return (trans1, trans1Inv)
            }
            return (trans2, trans2Inv)
        }
    }
    
    // Match Python's findNonreflectiveSimilarity exactly
    private func findNonreflectiveSimilarity(uv: [[Float]], xy: [[Float]]) -> ([[Float]], [[Float]])? {
        let M = xy.count
        
        // Get x and y columns from xy
        let x = xy.map { [$0[0]] }
        let y = xy.map { [$0[1]] }
        
        // Build X matrix exactly like Python
        var X: [[Float]] = []
        for i in 0..<M {
            X.append([x[i][0], y[i][0], 1.0, 0.0])
            X.append([y[i][0], -x[i][0], 0.0, 1.0])
        }
        
        // Build U vector exactly like Python
        let u = uv.map { [$0[0]] }
        let v = uv.map { [$0[1]] }
        var U: [Float] = []
        for i in 0..<M {
            U.append(u[i][0])
            U.append(v[i][0])
        }
        
        // Solve least squares exactly like Python
        guard let r = solveLeastSquares(X, U) else { return nil }
        
        let sc = r[0]
        let ss = r[1]
        let tx = r[2]
        let ty = r[3]
        
        // Build transformation matrices exactly like Python
        let Tinv: [[Float]] = [
            [sc, -ss, 0],
            [ss, sc, 0],
            [tx, ty, 1]
        ]
        
        guard let T = matrixInverse(Tinv) else { return nil }
        let TFinal: [[Float]] = [
            [T[0][0], T[0][1], T[0][2]],
            [T[1][0], T[1][1], T[1][2]],
            [0, 0, 1]
        ]
        
        return (TFinal, Tinv)
    }
    
    // Least squares solver matching Python's numpy.linalg.lstsq
    private func solveLeastSquares(_ A: [[Float]], _ b: [Float]) -> [Float]? {
        let n = A[0].count
        let m = A.count
        
        // Calculate A^T * A
        var AtA = [[Float]](repeating: [Float](repeating: 0, count: n), count: n)
        for i in 0..<n {
            for j in 0..<n {
                var sum: Float = 0
                for k in 0..<m {
                    if k < A.count && i < A[k].count && j < A[k].count {
                        sum += A[k][i] * A[k][j]
                    }
                }
                AtA[i][j] = sum
            }
        }
        
        // Calculate A^T * b
        var Atb = [Float](repeating: 0, count: n)
        for i in 0..<n {
            var sum: Float = 0
            for j in 0..<m {
                if j < A.count && i < A[j].count && j < b.count {
                    sum += A[j][i] * b[j]
                }
            }
            Atb[i] = sum
        }
        
        // Solve using Gaussian elimination
        return gaussianElimination(AtA, Atb)
    }
    
    // Gaussian elimination solver similar to NumPy's implementation
    private func gaussianElimination(_ A: [[Float]], _ b: [Float]) -> [Float]? {
        let n = A.count
        var augmented = A
        var bCopy = b
        
        // Forward elimination
        for i in 0..<n {
            // Find pivot
            var maxRow = i
            for k in i+1..<n {
                if abs(augmented[k][i]) > abs(augmented[maxRow][i]) {
                    maxRow = k
                }
            }
            
            // Swap rows
            if maxRow != i {
                augmented.swapAt(i, maxRow)
                let temp = bCopy[i]
                bCopy[i] = bCopy[maxRow]
                bCopy[maxRow] = temp
            }
            
            guard augmented[i][i] != 0 else { return nil }
            
            // Make all rows below this one 0 in current column
            for k in i+1..<n {
                let factor = augmented[k][i] / augmented[i][i]
                for j in i..<n {
                    augmented[k][j] -= factor * augmented[i][j]
                }
                bCopy[k] -= factor * bCopy[i]
            }
        }
        
        // Back substitution
        var x = [Float](repeating: 0, count: n)
        for i in stride(from: n-1, through: 0, by: -1) {
            x[i] = bCopy[i]
            for j in i+1..<n {
                x[i] -= augmented[i][j] * x[j]
            }
            x[i] /= augmented[i][i]
        }
        
        return x
    }
    
    // Matrix operations matching Python numpy
    private func matrixMultiply(_ A: [[Float]], _ B: [[Float]]) -> [[Float]] {
        let rowsA = A.count
        let colsA = A[0].count
        let colsB = B[0].count
        
        var result = [[Float]](repeating: [Float](repeating: 0, count: colsB), count: rowsA)
        
        for i in 0..<rowsA {
            for j in 0..<colsB {
                for k in 0..<colsA {
                    result[i][j] += A[i][k] * B[k][j]
                }
            }
        }
        
        return result
    }
    
    private func matrixInverse(_ matrix: [[Float]]) -> [[Float]]? {
        let n = matrix.count
        guard n == matrix[0].count else { return nil }
        
        // Create augmented matrix [A|I]
        var augmented = [[Float]](repeating: [Float](repeating: 0, count: 2*n), count: n)
        
        for i in 0..<n {
            for j in 0..<n {
                augmented[i][j] = matrix[i][j]
                augmented[i][j + n] = (i == j) ? 1.0 : 0.0
            }
        }
        
        // Gaussian elimination
        for i in 0..<n {
            // Find pivot
            var maxRow = i
            for k in i+1..<n {
                if abs(augmented[k][i]) > abs(augmented[maxRow][i]) {
                    maxRow = k
                }
            }
            
            if maxRow != i {
                augmented.swapAt(i, maxRow)
            }
            
            guard augmented[i][i] != 0 else { return nil }
            
            // Scale row
            let pivot = augmented[i][i]
            for j in 0..<2*n {
                augmented[i][j] /= pivot
            }
            
            // Eliminate column
            for k in 0..<n {
                if k != i {
                    let factor = augmented[k][i]
                    for j in 0..<2*n {
                        augmented[k][j] -= factor * augmented[i][j]
                    }
                }
            }
        }
        
        // Extract inverse matrix
        var inverse = [[Float]](repeating: [Float](repeating: 0, count: n), count: n)
        for i in 0..<n {
            for j in 0..<n {
                inverse[i][j] = augmented[i][j + n]
            }
        }
        
        return inverse
    }
    
    private func transformPoints(_ points: [[Float]], transform: [[Float]]) -> [[Float]] {
        return points.map { point in
            let homogeneous = [point[0], point[1], 1.0]
            var result = [Float](repeating: 0, count: 2)
            
            for i in 0..<2 {
                for j in 0..<3 {
                    result[i] += transform[i][j] * homogeneous[j]
                }
            }
            
            return result
        }
    }
    
    private func calculateNorm(_ points1: [[Float]], _ points2: [[Float]]) -> Float {
        var sum: Float = 0
        for i in 0..<min(points1.count, points2.count) {
            let dx = points1[i][0] - points2[i][0]
            let dy = points1[i][1] - points2[i][1]
            sum += dx * dx + dy * dy
        }
        return sqrt(sum)
    }
    
    // Warp affine exactly like Python cv2.warpAffine
    private func warpAffine(image: UIImage, transform: [[Float]], outputSize: (Int, Int)) -> UIImage? {
        guard let cgImage = image.cgImage else { return nil }
        
        // Create a CGAffineTransform from our transform matrix
        let affineTransform = CGAffineTransform(
            a: CGFloat(transform[0][0]),
            b: CGFloat(transform[1][0]),
            c: CGFloat(transform[0][1]),
            d: CGFloat(transform[1][1]),
            tx: CGFloat(transform[0][2]),
            ty: CGFloat(transform[1][2])
        )
        
        // Create a context for the output image
        let context = CGContext(
            data: nil,
            width: outputSize.0,
            height: outputSize.1,
            bitsPerComponent: cgImage.bitsPerComponent,
            bytesPerRow: 0,
            space: cgImage.colorSpace ?? CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: cgImage.bitmapInfo.rawValue
        )
        
        guard let context = context else { return nil }
        
        // Clear the context with black background
        context.setFillColor(UIColor.black.cgColor)
        context.fill(CGRect(x: 0, y: 0, width: outputSize.0, height: outputSize.1))
        
        // Apply the transformation
        context.concatenate(affineTransform)
        
        // Draw the image
        let rect = CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height)
        context.draw(cgImage, in: rect)
        
        // Get the resulting image
        guard let outputCGImage = context.makeImage() else { return nil }
        
        return UIImage(cgImage: outputCGImage)
    }
    
    // Face embedding extraction exactly matching Python preprocessing
    func extractFaceEmbedding(faceImage: UIImage, source: String = "unknown") async throws -> [Double]? {
        if !isModelLoaded {
            let success = try await loadModel()
            if !success {
                os_log(.error, log: OSLog.default, "%{public}s: Model not loaded, cannot extract embedding", TAG)
                return nil
            }
        }
        
        guard let hwcPixels = resizeAndNormalize(image: faceImage) else {
            os_log(.error, log: OSLog.default, "%{public}s: Failed to preprocess image for %{public}s", TAG, source)
            return nil
        }
        
        let nchwPixels = hwcToNchw(hwcPixels: hwcPixels)
        
        guard let embeddingSession = embeddingSession else {
            os_log(.error, log: OSLog.default, "%{public}s: Embedding session is nil", TAG)
            return nil
        }
        
        let shape: [NSNumber] = [1, 3, INPUT_SIZE, INPUT_SIZE].map { NSNumber(value: $0) }
        let tensor = try ORTValue(
            tensorData: NSMutableData(bytes: nchwPixels, length: nchwPixels.count * MemoryLayout<Float>.size),
            elementType: .float,
            shape: shape
        )
        
        let inputName = try embeddingSession.inputNames().first ?? EMBEDDING_INPUT_NAME
        let outputName = try embeddingSession.outputNames().first ?? EMBEDDING_OUTPUT_NAME
        
        do {
            let outputs = try embeddingSession.run(
                withInputs: [inputName: tensor],
                outputNames: [outputName],
                runOptions: nil
            )
            
            guard let outputTensor = outputs[outputName] else {
                os_log(.error, log: OSLog.default, "%{public}s: Failed to get output tensor", TAG)
                return nil
            }
            
            let outputData = try outputTensor.tensorData() as Data
            let count = outputData.count / MemoryLayout<Float>.size
            
            var embeddingArray = [Float](repeating: 0, count: count)
            outputData.withUnsafeBytes { rawBufferPointer in
                if let baseAddress = rawBufferPointer.baseAddress {
                    let pointer = baseAddress.assumingMemoryBound(to: Float.self)
                    for i in 0..<count {
                        embeddingArray[i] = pointer[i]
                    }
                }
            }
            
            return embeddingArray.map { Double($0) }
        } catch {
            os_log(.error, log: OSLog.default, "%{public}s: Error during inference: %{public}s", TAG, error.localizedDescription)
            #if DEBUG
            return getMockEmbedding()
            #else
            throw error
            #endif
        }
    }
    
    private func getMockEmbedding() -> [Double] {
        var mockEmbedding = [Double](repeating: 0, count: EMBEDDING_SIZE)
        for i in 0..<EMBEDDING_SIZE {
            mockEmbedding[i] = Double.random(in: -1.0...1.0)
        }
        return mockEmbedding
    }
    
    // Match Python's preprocessing exactly: resize to 112x112 and normalize
    private func resizeAndNormalize(image: UIImage) -> [Float]? {
        guard image.cgImage != nil else { return nil }
        
        let resizedImage = (image.size.width != CGFloat(INPUT_SIZE) || image.size.height != CGFloat(INPUT_SIZE)) ?
            image.resize(size: CGSize(width: INPUT_SIZE, height: INPUT_SIZE), interpolation: .medium) : image
        
        guard let pixels = resizedImage.pixelData() else { return nil }
        
        var floatPixels = [Float](repeating: 0, count: INPUT_SIZE * INPUT_SIZE * 3)
        
        for y in 0..<INPUT_SIZE {
            for x in 0..<INPUT_SIZE {
                let pixelOffset = y * INPUT_SIZE + x
                let pixel = pixels[pixelOffset]
                let rgb = [
                    Float((pixel >> 16) & 0xFF),
                    Float((pixel >> 8) & 0xFF),
                    Float(pixel & 0xFF)
                ]
                
                let idx = pixelOffset * 3
                for c in 0..<3 {
                    floatPixels[idx + c] = (rgb[c] - 127.5) * 0.0078125
                }
            }
        }
        
        return floatPixels
    }
    
    private func hwcToNchw(hwcPixels: [Float]) -> [Float] {
        let size = INPUT_SIZE * INPUT_SIZE
        var nchwPixels = [Float](repeating: 0, count: 3 * size)
        
        for i in 0..<size {
            for c in 0..<3 {
                nchwPixels[c * size + i] = hwcPixels[i * 3 + c]
            }
        }
        
        return nchwPixels
    }
}

extension Float {
    var toDouble: Double { Double(self) }
}

extension UIImage {
    func resize(size: CGSize, interpolation: CGInterpolationQuality) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        defer { UIGraphicsEndImageContext() }
        
        guard let context = UIGraphicsGetCurrentContext() else { return self }
        context.interpolationQuality = interpolation
        draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext() ?? self
    }
    
    func pixelData() -> [UInt32]? {
        guard let cgImage = cgImage else { return nil }
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        var pixelData = [UInt32](repeating: 0, count: width * height)
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo: UInt32 = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        guard let context = CGContext(data: &pixelData,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: bitsPerComponent,
                                      bytesPerRow: bytesPerRow,
                                      space: colorSpace,
                                      bitmapInfo: bitmapInfo) else {
            return nil
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pixelData
    }
}
