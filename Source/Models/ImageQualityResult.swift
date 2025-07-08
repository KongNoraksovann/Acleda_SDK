import Foundation

/**
 * Represents the result of image quality check, including:
 *  - Brightness
 *  - Sharpness
 *  - Face presence
 *  - Screen-artifact (moiré) detection
 *  - Glare/specular highlight detection
 */
@objc public class ImageQualityResult: NSObject {
    @objc public var brightnessScore: Float = 0.0
    @objc public var sharpnessScore: Float = 0.0
    @objc public var screenArtifactScore: Float = 1.0  // NEW: moiré/artifact check
    @objc public var glareScore: Float = 1.0          // NEW: specular glare check
    @objc public var faceScore: Float = 0.0
    @objc public var hasFace: Bool = false
    @objc public var overallScore: Float = 0.0
    
    @objc public static let BRIGHTNESS_WEIGHT: Float = 0.25
    @objc public static let SHARPNESS_WEIGHT: Float = 0.25
    @objc public static let FACE_WEIGHT: Float = 0.20
    @objc public static let SCREEN_ARTIFACT_WEIGHT: Float = 0.15
    @objc public static let GLARE_WEIGHT: Float = 0.15
    @objc public static let ACCEPTABLE_SCORE_THRESHOLD: Float = 0.5
    
    /**
     * Create a default instance for cases where quality check is skipped
     */
    @objc public static func createDefault() -> ImageQualityResult {
        let result = ImageQualityResult()
        result.brightnessScore = 0.0
        result.sharpnessScore = 0.0
        result.screenArtifactScore = 1.0
        result.glareScore = 1.0
        result.faceScore = 0.0
        result.hasFace = false
        result.overallScore = 0.0
        return result
    }
    
    /**
     * Calculates the overall score based on weighted components.
     * If no face is detected, overallScore stays 0.
     */
    @objc public func calculateOverallScore() {
        overallScore = hasFace ? (
            brightnessScore * ImageQualityResult.BRIGHTNESS_WEIGHT +
            sharpnessScore * ImageQualityResult.SHARPNESS_WEIGHT +
            faceScore * ImageQualityResult.FACE_WEIGHT +
            screenArtifactScore * ImageQualityResult.SCREEN_ARTIFACT_WEIGHT +
            glareScore * ImageQualityResult.GLARE_WEIGHT
        ) : 0.0
        overallScore = max(0.0, min(1.0, overallScore))
    }
    
    /**
     * Determines if the image quality is acceptable for further processing.
     */
    @objc public func isAcceptable() -> Bool {
        return hasFace && overallScore >= ImageQualityResult.ACCEPTABLE_SCORE_THRESHOLD
    }
    
    /**
     * Get detailed breakdown of all component scores.
     */
    @objc public func getDetailedReport() -> [String: Any] {
        return [
            "brightnessScore": brightnessScore,
            "sharpnessScore": sharpnessScore,
            "screenArtifactScore": screenArtifactScore,
            "glareScore": glareScore,
            "faceScore": faceScore,
            "hasFace": hasFace,
            "overallScore": overallScore,
            "isAcceptable": isAcceptable()
        ]
    }
    
    public override var description: String {
        return String(format: "Quality: %.2f (B: %.2f, S: %.2f, F: %.2f, A: %.2f, G: %.2f)",
                      overallScore, brightnessScore, sharpnessScore, faceScore, screenArtifactScore, glareScore)
    }
}
