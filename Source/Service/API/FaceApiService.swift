import Foundation

protocol FaceApiService {
    func registerFace(
        userId: String,
        skipQualityCheck: Bool,
        sourceFile: Data,
        alignedFile: Data
    ) async throws -> FaceApiClient.EmbeddingResponse
    
    func verifyFace(
        userId: String,
        file: Data
    ) async throws -> FaceApiClient.VerificationResponse
}

// Default implementation for skipQualityCheck
extension FaceApiService {
    func registerFace(
        userId: String,
        sourceFile: Data,
        alignedFile: Data
    ) async throws -> FaceApiClient.EmbeddingResponse {
        try await registerFace(
            userId: userId,
            skipQualityCheck: false,
            sourceFile: sourceFile,
            alignedFile: alignedFile
        )
    }
}

class FaceApiServiceImpl: FaceApiService {
    func registerFace(
        userId: String,
        skipQualityCheck: Bool = false,
        sourceFile: Data,
        alignedFile: Data
    ) async throws -> FaceApiClient.EmbeddingResponse {
        try await FaceApiClient.registerFace(
            userId: userId,
            skipQualityCheck: skipQualityCheck,
            sourceFile: sourceFile,
            alignedFile: alignedFile
        )
    }
    func verifyFace(
        userId: String,
        file: Data
    ) async throws -> FaceApiClient.VerificationResponse {
        try await FaceApiClient.verifyFace(
            userId: userId,
            file: file
        )
    }
}

class FaceApiManager {
    static let shared = FaceApiManager()
    private let apiService: FaceApiService = FaceApiServiceImpl()
    private init() {}
    
    func registerFace(
        userId: String,
        skipQualityCheck: Bool = false,
        sourceFile: Data,
        alignedFile: Data
    ) async throws -> FaceApiClient.EmbeddingResponse {
        try await apiService.registerFace(
            userId: userId,
            skipQualityCheck: skipQualityCheck,
            sourceFile: sourceFile,
            alignedFile: alignedFile
        )
    }
    func verifyFace(
        userId: String,
        file: Data
    ) async throws -> FaceApiClient.VerificationResponse {
        try await apiService.verifyFace(
            userId: userId,
            file: file
        )
    }
}
