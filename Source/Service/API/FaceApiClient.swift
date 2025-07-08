import Foundation
import os.log
import UIKit

enum FaceApiError: Error, LocalizedError {
    case networkError(URLError)
    case invalidImageData
    case faceOccluded
    case apiValidationError(code: Int, message: String)
    case serverError(code: Int, message: String)
    case unknown(String)
    
    var errorDescription: String? {
        switch self {
        case .networkError(let urlError): return "Network error: \(urlError.localizedDescription)"
        case .invalidImageData: return "Invalid image data provided"
        case .faceOccluded: return "Face is occluded. Please ensure face is clearly visible."
        case .apiValidationError(let code, let message): return "Validation error (\(code)): \(message)"
        case .serverError(let code, let message): return "Server error (\(code)): \(message)"
        case .unknown(let message): return "Unknown error: \(message)"
        }
    }
}

@available(iOS 13.0, *)
class FaceApiClient {
    private static let TAG = "FaceApiClient"
    private static let log = OSLog(subsystem: "com.example.faceverification.api", category: "FaceApiClient")
    private static let gistUrl = "https://face.konai.dev/"
    private static let fallbackUrl = "https://face.konai.dev/"
    private static let timeout: TimeInterval = 10.0

    struct EmbeddingResponse: Codable {
        let status: String
        let code: Int?
        let message: String?
        let details: EmbeddingDetails?
        enum CodingKeys: String, CodingKey {
            case status, code, message = "msg", details
        }
        var isSuccess: Bool { status.lowercased() == "success" }
    }
    struct EmbeddingDetails: Codable { let similarity: Float? }
    struct VerificationResponse: Codable {
        let status: String
        let code: Int?
        let message: String?
        let details: VerificationDetails?
    }
    struct VerificationDetails: Codable {
        let similarity: Float?
        let spoofLabel: String?
        let occlusionLabel: String?
    }

    private static var baseUrl: String = {
        let semaphore = DispatchSemaphore(value: 0)
        var urlString = fallbackUrl
        Task {
            do { urlString = try await fetchBaseUrl() }
            catch { urlString = fallbackUrl }
            semaphore.signal()
        }
        _ = semaphore.wait(timeout: .now() + timeout)
        return urlString
    }()

    private static func fetchBaseUrl() async throws -> String {
        let maxRetries = 3
        var lastError: Error?
        for attempt in 1...maxRetries {
            do { return try await fetchBaseUrlAttempt() }
            catch { lastError = error; if attempt < maxRetries { try? await Task.sleep(nanoseconds: UInt64(attempt) * 1_000_000_000) } }
        }
        throw lastError ?? URLError(.cannotConnectToHost)
    }

    private static func fetchBaseUrlAttempt() async throws -> String {
        guard let url = URL(string: gistUrl) else { throw URLError(.badURL) }
        var request = URLRequest(url: url)
        request.timeoutInterval = timeout
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse, (200...299).contains(httpResponse.statusCode) else {
            throw URLError(.badServerResponse)
        }
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let fetchedUrl = json["url"] as? String else {
            throw URLError(.cannotParseResponse)
        }
        let finalUrl = fetchedUrl.hasSuffix("/") ? fetchedUrl : "\(fetchedUrl)/"
        guard URL(string: finalUrl) != nil else { throw URLError(.badURL) }
        return finalUrl
    }

    @discardableResult
    static func registerFace(userId: String, skipQualityCheck: Bool = false, sourceFile: Data, alignedFile: Data) async throws -> EmbeddingResponse {
        guard UIImage(data: sourceFile) != nil, UIImage(data: alignedFile) != nil else { throw FaceApiError.invalidImageData }
        let endpoint = "api/v2/img/register"
        guard var urlComponents = URLComponents(string: "\(baseUrl)\(endpoint)") else { throw FaceApiError.networkError(URLError(.badURL)) }
        urlComponents.queryItems = [
            URLQueryItem(name: "user_id", value: userId),
            URLQueryItem(name: "skip_quality_check", value: skipQualityCheck.description)
        ]
        guard let url = urlComponents.url else { throw FaceApiError.networkError(URLError(.badURL)) }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "accept")
        request.timeoutInterval = 30.0
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        var body = Data()
        body.appendFormData(name: "source_file", filename: "source.jpg", data: sourceFile, boundary: boundary)
        body.appendFormData(name: "aligned_file", filename: "aligned.jpg", data: alignedFile, boundary: boundary, isLast: true)
        request.httpBody = body
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else { throw FaceApiError.networkError(URLError(.badServerResponse)) }
            let embeddingResponse = try JSONDecoder().decode(EmbeddingResponse.self, from: data)
            if !embeddingResponse.isSuccess {
                let errorMessage = embeddingResponse.message ?? "Unknown error"
                let errorCode = embeddingResponse.code ?? 0
                if errorMessage.lowercased().contains("occluded") { throw FaceApiError.faceOccluded }
                else if errorCode >= 400 && errorCode < 500 { throw FaceApiError.apiValidationError(code: errorCode, message: errorMessage) }
                else { throw FaceApiError.serverError(code: errorCode, message: errorMessage) }
            }
            return embeddingResponse
        } catch let error as FaceApiError { throw error }
        catch { throw FaceApiError.networkError(error as? URLError ?? URLError(.unknown)) }
    }

    @discardableResult
    static func verifyFace(userId: String, file: Data) async throws -> VerificationResponse {
        guard UIImage(data: file) != nil else { throw FaceApiError.invalidImageData }
        let endpoint = "api/v2/img/verify"
        guard var urlComponents = URLComponents(string: "\(baseUrl)\(endpoint)") else { throw FaceApiError.networkError(URLError(.badURL)) }
        urlComponents.queryItems = [URLQueryItem(name: "user_id", value: userId)]
        guard let url = urlComponents.url else { throw FaceApiError.networkError(URLError(.badURL)) }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "accept")
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        var body = Data()
        body.appendFormData(name: "file", filename: "image.jpg", data: file, boundary: boundary, isLast: true)
        request.httpBody = body
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else { throw FaceApiError.networkError(URLError(.badServerResponse)) }
            if !(200...299).contains(httpResponse.statusCode) {
                let responseBody = String(data: data, encoding: .utf8) ?? "No response body"
                throw FaceApiError.serverError(code: httpResponse.statusCode, message: responseBody)
            }
            return try JSONDecoder().decode(VerificationResponse.self, from: data)
        } catch let error as FaceApiError { throw error }
        catch { throw FaceApiError.networkError(error as? URLError ?? URLError(.unknown)) }
    }
}

// MARK: - Data extension for multipart form data
private extension Data {
    mutating func appendFormData(name: String, filename: String, data: Data, boundary: String, isLast: Bool = false) {
        append("--\(boundary)\r\n".data(using: .utf8)!)
        append("Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
        append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        append(data)
        append("\r\n".data(using: .utf8)!)
        if isLast { append("--\(boundary)--\r\n".data(using: .utf8)!) }
    }
}
