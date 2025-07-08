import Foundation

/**
 * Represents the result of the face liveness detection process
 */
@objc public class FaceLivenessModel: NSObject {
    /**
     * The prediction result: "Live" or "Spoof"
     */
    @objc public let prediction: String
    
    /**
     * Confidence level in the prediction (0.0 to 1.0)
     */
    @objc public let confidence: Float
    
    /**
     * Reason for failure if authentication failed
     */
    @objc public let failureReason: String?
    
    /**
     * Initialize a new face liveness model
     */
    @objc public init(prediction: String, confidence: Float, failureReason: String? = nil) {
        self.prediction = prediction
        self.confidence = confidence
        self.failureReason = failureReason
        super.init()
    }
}
