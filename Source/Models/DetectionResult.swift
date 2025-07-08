import Foundation

/// Represents the result of a detection operation.
public struct DetectionResult {
    public let label: String
    public let confidence: Float
    
    public init(label: String, confidence: Float) {
        self.label = label
        self.confidence = confidence
    }
}

