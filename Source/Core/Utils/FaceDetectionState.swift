import Foundation

public enum FaceDetectionState {
    case WAITING          // Waiting for face to be detected
    case DETECTING        // Face detected, checking for good position
    case STABILIZING      // Face in good position, waiting for stability
    case COUNTDOWN        // Countdown before capture
    case CAPTURING        // Capturing the image
    case LIVENESS_CHECK   // Performing liveness detection
    case SUCCESS          // Successfully captured
    case LIVENESS_FAILED  // Liveness check failed
    case BLINK_CHECK      // Checking for blink
}

public enum FaceDistanceStatus {
    case UNKNOWN    // Face distance not yet determined
    case TOO_FAR    // Face is too far from camera
    case OPTIMAL    // Face is at optimal distance
    case TOO_CLOSE  // Face is too close to camera
}
