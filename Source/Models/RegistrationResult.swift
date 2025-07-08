import Foundation

enum RegistrationResult {
    case success(String)
    case error(String)
    
    var status: String {
        switch self {
        case .success: return "success"
        case .error: return "error"
        }
    }
    
    var message: String {
        switch self {
        case .success(let message): return message
        case .error(let message): return message
        }
    }
}
