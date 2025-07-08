import Foundation
import CoreGraphics
import os.log
import UIKit

@available(iOS 13.0, *)
class FaceVerificationManager {
    private let TAG = "FaceVerificationManager"
    private let log = OSLog(subsystem: "com.example.faceverification.core", category: "FaceVerificationManager")
    
    let faceEmbedding: FaceEmbedding
    let faceDatabase: FaceDatabase
    private var useCombinedVerification: Bool = false
    let context: Any
    
    init(faceDatabase: FaceDatabase, context: Any = Bundle.main) {
        self.faceEmbedding = FaceEmbedding()
        self.faceDatabase = faceDatabase
        self.context = context
        os_log(.debug, log: log, "%{public}s: Initializing", TAG)
    }
    
    func initialize() async -> Bool {
        do {
            return await faceEmbedding.initialize()
        } catch {
            os_log(.error, log: log, "%{public}s: Failed to initialize: %{public}s", TAG, error.localizedDescription)
            return false
        }
    }
    
    func setCombinedVerification(enabled: Bool) {
        useCombinedVerification = enabled
    }
    
    func setVerificationThresholds(cosineThreshold: Float? = nil) {
        if let cosine = cosineThreshold {
            faceEmbedding.setCosineThreshold(threshold: cosine)
        }
    }
    
    func verifyFace(faceBitmap: UIImage, userId: String) async -> VerificationResult {
        do {
            guard let stored = try await faceDatabase.getFaceEmbeddingByUserId(userId: userId) else {
                return .error(message: "No registered face found")
            }
            
            let embeddingResult = await faceEmbedding.getFaceEmbedding(faceImage: faceBitmap, source: "verifyFace")
            guard let currentEmbedding = embeddingResult.0 else {
                return .error(message: "Failed to extract embedding")
            }
            
            let similarity = faceEmbedding.cosineSimilarity(embedding1: stored.embedding, embedding2: currentEmbedding)
            let isMatch = faceEmbedding.verifyFaces(embedding1: stored.embedding, embedding2: currentEmbedding)
            
            if isMatch {
                try await faceDatabase.updateMatchStatistics(userId: userId)
                return .success(
                    message: "Face verified successfully",
                    similarity: similarity,
                    userName: stored.name ?? "",
                    userId: stored.userId
                )
            } else {
                return .failure(
                    message: "Face verification failed",
                    similarity: similarity,
                    bestMatchName: stored.name ?? ""
                )
            }
        } catch {
            return .error(message: "Verification error: \(error.localizedDescription)")
        }
    }
    
    func searchFace(faceBitmap: UIImage) async -> VerificationResult {
        do {
            let allEmbeddings = try await faceDatabase.getAllFaceEmbeddings()
            if allEmbeddings.isEmpty {
                return .error(message: "No registered users")
            }
            
            let embeddingResult = await faceEmbedding.getFaceEmbedding(faceImage: faceBitmap, source: "searchFace")
            guard let currentEmbedding = embeddingResult.0 else {
                return .error(message: "Failed to extract embedding")
            }
            
            var bestMatch: VerificationResult? = nil
            
            for (userId, embeddings) in allEmbeddings {
                guard let stored = embeddings.first else { continue }
                
                let similarity = faceEmbedding.cosineSimilarity(embedding1: stored.embedding, embedding2: currentEmbedding)
                let isMatch = faceEmbedding.verifyFaces(embedding1: stored.embedding, embedding2: currentEmbedding)
                
                if isMatch {
                    try await faceDatabase.updateMatchStatistics(userId: userId)
                    return .success(
                        message: "Match found",
                        similarity: similarity,
                        userName: stored.name ?? "",
                        userId: stored.userId
                    )
                }
                
                if bestMatch == nil || similarity > (bestMatch?.similarity ?? 0.0) {
                    bestMatch = .failure(
                        message: "Best match so far",
                        similarity: similarity,
                        bestMatchName: stored.name ?? ""
                    )
                }
            }
            
            return bestMatch ?? .error(message: "No match found")
        } catch {
            return .error(message: "Internal error: \(error.localizedDescription)")
        }
    }
    
    func verifyFaceViaApi(faceBitmap: UIImage, userId: String) async -> VerificationResult {
        do {
            let embeddingResult = await faceEmbedding.getFaceEmbedding(faceImage: faceBitmap, source: "verifyFaceViaApi")
            guard let alignedBitmap = embeddingResult.1,
                  let imageData = alignedBitmap.jpegData(compressionQuality: 1.0) else {
                return .error(message: "Failed to process face image")
            }
            
            let response = try await FaceApiClient.verifyFace(userId: userId, file: imageData)
            
            switch response.status {
            case "success":
                return .success(
                    message: response.message ?? "Verification successful",
                    similarity: response.details?.similarity ?? 0.0,
                    userName: userId,
                    userId: userId
                )
            case "fail":
                switch response.code {
                case 400:
                    let failureReason: String
                    if response.details?.spoofLabel == "spoof" {
                        failureReason = "Spoof detected in the image"
                    } else if response.details?.occlusionLabel != "Clear" {
                        failureReason = "Face occlusion detected"
                    } else {
                        failureReason = response.message ?? "Verification failed"
                    }
                    return .error(message: failureReason)
                case 404:
                    return .error(message: response.message ?? "No face record found for user ID")
                case 407:
                    return .failure(
                        message: response.message ?? "Face does not match",
                        similarity: response.details?.similarity ?? 0.0,
                        bestMatchName: userId
                    )
                default:
                    return .error(message: response.message ?? "Verification failed")
                }
            default:
                return .error(message: response.message ?? "Unknown error")
            }
        } catch {
            return .error(message: "Verification error: \(error.localizedDescription)")
        }
    }
    
    func registerFaceViaApi(faceBitmap: UIImage?, qualityResult: FaceQualityResult, userId: String, userName: String, croppedBitmap: UIImage? = nil) async -> RegistrationResult {
        do {
            guard let faceBitmap = faceBitmap else {
                return .error("No face detected")
            }
            
            let embeddingResult = await faceEmbedding.getFaceEmbedding(faceImage: faceBitmap, source: "registerFaceViaApi")
            guard let alignedBitmap = embeddingResult.1 else {
                return .error("Failed to align face")
            }
            
            let sourceBitmap = croppedBitmap ?? faceBitmap
            guard let sourceData = sourceBitmap.jpegData(compressionQuality: 1.0),
                  let alignedData = alignedBitmap.jpegData(compressionQuality: 1.0) else {
                return .error("Failed to process images")
            }
            
            let response = try await FaceApiClient.registerFace(
                userId: userId,
                skipQualityCheck: false,
                sourceFile: sourceData,
                alignedFile: alignedData
            )
            
            if response.status == "success" {
                let localEmbeddingResult = await faceEmbedding.getFaceEmbedding(faceImage: alignedBitmap, source: "registerFaceViaApi_local")
                guard let embedding = localEmbeddingResult.0 else {
                    return .error("Failed to generate local embedding")
                }
                
                if try await faceDatabase.storeFaceEmbedding(
                    userId: userId,
                    name: userName,
                    embedding: embedding,
                    faceBitmap: alignedBitmap
                ) {
                    return .success("Face registered successfully")
                } else {
                    return .error("Failed to store embedding in database")
                }
            } else {
                return .error("API error: \(response.message ?? "Unknown error")")
            }
        } catch {
            return .error("Registration error: \(error.localizedDescription)")
        }
    }
    
    func registerFaceLocally(faceBitmap: UIImage?, qualityResult: FaceQualityResult, userId: String, name: String) async -> RegistrationResult {
        do {
            if !qualityResult.isGoodQuality {
                return .error(qualityResult.failureReason ?? "Poor image quality")
            }
            
            guard let faceBitmap = faceBitmap else {
                return .error("No face detected")
            }
            
            let embeddingResult = await faceEmbedding.getFaceEmbedding(faceImage: faceBitmap, source: "registerFaceLocally")
            guard let embedding = embeddingResult.0,
                  let alignedBitmap = embeddingResult.1 else {
                return .error("Failed to process face")
            }
            
            if try await faceDatabase.storeFaceEmbedding(
                userId: userId,
                name: name,
                embedding: embedding,
                faceBitmap: alignedBitmap
            ) {
                return .success("Face registered successfully")
            } else {
                return .error("Failed to store face embedding locally")
            }
        } catch {
            return .error("Local registration failed: \(error.localizedDescription)")
        }
    }
    
    func close() {
        faceEmbedding.release()
    }
}

struct VerifyFaceResponse: Codable {
    let status: String
    let message: String?
    let code: String?
    let details: Details?
    
    struct Details: Codable {
        let similarity: Float?
        let spoofLabel: String?
        let occlusionLabel: String?
    }
}

struct RegisterFaceResponse: Codable {
    let status: String
    let msg: String
    let code: String?
    let details: Details?
    
    struct Details: Codable {
        let embedding: [Float]?
    }
}
