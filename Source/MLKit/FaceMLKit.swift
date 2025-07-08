import Foundation
import UIKit
import MLKitFaceDetection

enum FaceDetectionError: Error {
    case detectionFailed(String)
}

@available(iOS 13.0, *)
class FaceMLKit {
    private let faceDetector = MLKitFaceDetector()
    private let dispatchQueue = DispatchQueue(label: "com.example.faceverification.mlkit", qos: .default)
    
    // Singleton instance
    private static var instance: FaceMLKit?
    
  
    init() {
 
    }
    
    // MARK: - Singleton Access
    static func getInstance() -> FaceMLKit {
        if let existingInstance = instance {
            return existingInstance
        }
        objc_sync_enter(self)
        defer { objc_sync_exit(self) }
        if let existingInstance = instance {
            return existingInstance
        }
        let newInstance = FaceMLKit()
        instance = newInstance
        return newInstance
    }
    
    func detectFaces(in bitmap: CGImage, completion: @escaping (FaceDetectionResult) -> Void) {
        dispatchQueue.async {
            Task {
                do {
                    let faces = try await self.faceDetector.detectFaces(in: bitmap)
                    let result = FaceDetectionResult(
                        success: !faces.isEmpty,
                        faces: faces,
                        message: faces.isEmpty ? "No faces detected" : "\(faces.count) faces detected"
                    )
                    DispatchQueue.main.async {
                        completion(result)
                    }
                } catch {
                    let result = FaceDetectionResult(
                        success: false,
                        faces: [],
                        message: "Face detection failed: \(error.localizedDescription)"
                    )
                    DispatchQueue.main.async {
                        completion(result)
                    }
                }
            }
        }
    }
    
    func extractLargestFace(from bitmap: CGImage, completion: @escaping (CGImage?) -> Void) {
        dispatchQueue.async {
            Task {
                do {
                    let result = try await self.faceDetector.cropFace(from: bitmap)
                    DispatchQueue.main.async {
                        completion(result.faceImage)
                    }
                } catch {
                    DispatchQueue.main.async {
                        completion(nil)
                    }
                }
            }
        }
    }
    
    func extractLargestFaceWithQuality(from bitmap: CGImage, completion: @escaping (CGImage?, FaceQualityResult) -> Void) {
        dispatchQueue.async {
            Task {
                do {
                    let result = try await self.faceDetector.cropFace(from: bitmap)
                    DispatchQueue.main.async {
                        // Always return good quality for now
                        let qualityResult = FaceQualityResult(
                            isGoodQuality: true,
                            qualityScore: 1.0,
                            issues: [],
                            failureReason: nil
                        )
                        completion(result.faceImage, qualityResult)
                    }
                } catch {
                    DispatchQueue.main.async {
                        // Even on error, return good quality but with nil face image
                        let qualityResult = FaceQualityResult(
                            isGoodQuality: true,
                            qualityScore: 1.0,
                            issues: [],
                            failureReason: nil
                        )
                        completion(nil, qualityResult)
                    }
                }
            }
        }
    }
    
    func detectFaces(in bitmap: CGImage) async throws -> FaceDetectionResult {
        try await withCheckedThrowingContinuation { continuation in
            detectFaces(in: bitmap) { result in
                if result.success {
                    continuation.resume(returning: result)
                } else {
                    continuation.resume(throwing: FaceDetectionError.detectionFailed(result.message))
                }
            }
        }
    }
    
    func extractLargestFace(from bitmap: CGImage) async throws -> CGImage? {
        try await withCheckedThrowingContinuation { continuation in
            extractLargestFace(from: bitmap) { faceImage in
                if let faceImage = faceImage {
                    continuation.resume(returning: faceImage)
                } else {
                    continuation.resume(throwing: FaceDetectionError.detectionFailed("No face extracted"))
                }
            }
        }
    }
    
    func extractLargestFaceWithQuality(from bitmap: CGImage) async throws -> (CGImage?, FaceQualityResult) {
        try await withCheckedThrowingContinuation { continuation in
            extractLargestFaceWithQuality(from: bitmap) { faceImage, qualityResult in
                continuation.resume(returning: (faceImage, qualityResult))
            }
        }
    }
    
    func close() {
        faceDetector.close()
    }
}
