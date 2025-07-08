import UIKit

enum AlbedoDetectorError: Error {
    case invalidImage(String)
    case processingFailed(String)
    case invalidGamma(String)
    
    var localizedDescription: String {
        switch self {
        case .invalidImage(let message), .processingFailed(let message), .invalidGamma(let message):
            return message
        }
    }
}
