import Foundation
import MLKitFaceDetection

/// Result object for face detection operations
struct FaceDetectionResult {
    let success: Bool
    let faces: [Face]
    let message: String
}

/// Result object for face quality assessment
struct FaceQualityResult {
    let isGoodQuality: Bool
    let qualityScore: Float
    let issues: [QualityIssue]
    let failureReason: String?
}

/// Enum representing possible face quality issues
enum QualityIssue {
    case multipleFaces
    case noFaceDetected
    case underexposed
    case blurryFace
}

extension QualityIssue {
    var description: String {
        switch self {
        case .multipleFaces: return "Multiple faces detected"
        case .noFaceDetected: return "No face detected"
        case .underexposed: return "Image underexposed"
        case .blurryFace: return "Blurry face"
        }
    }
}
