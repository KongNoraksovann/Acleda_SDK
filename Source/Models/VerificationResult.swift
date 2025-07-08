import Foundation

enum VerificationResult {
    case success(message: String, similarity: Float, userName: String?, userId: String?)
    case failure(message: String, similarity: Float, bestMatchName: String?)
    case error(message: String)
    
    var message: String {
        switch self {
        case .success(let message, _, _, _): return message
        case .failure(let message, _, _): return message
        case .error(let message): return message
        }
    }
    
    var similarity: Float {
        switch self {
        case .success(_, let similarity, _, _): return similarity
        case .failure(_, let similarity, _): return similarity
        case .error: return 0.0
        }
    }
    
    var userName: String? {
        switch self {
        case .success(_, _, let userName, _): return userName
        case .failure(_, _, let bestMatchName): return bestMatchName
        case .error: return nil
        }
    }
    
    var userId: String? {
        switch self {
        case .success(_, _, _, let userId): return userId
        case .failure: return nil
        case .error: return nil
        }
    }
}
