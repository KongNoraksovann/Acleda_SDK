
import Foundation
import CoreGraphics
import os.log

class FaceQualityChecker {
    private let LAPLACIAN1_THRESHOLD: Double = 45.0

    func checkFaceQuality(image: CGImage) -> FaceQualityResult {
        var issues: [QualityIssue] = []
        var failureReason: String? = nil
        var qualityScore: Float = 1.0

        // Compute sharpness metric using Laplacian
        let laplacian1 = computeLaplacian1Sharpness(image: image)

        os_log(.debug, log: .default, "Laplacian sharpness score: %{public}f (threshold: %{public}f)", laplacian1, LAPLACIAN1_THRESHOLD)

        // Use Laplacian for pass/fail
        if laplacian1 < LAPLACIAN1_THRESHOLD {
            os_log(.debug, log: .default, "Quality check failed: Image is blurry (Laplacian1=%{public}f)", laplacian1)
            issues.append(.blurryFace)
            failureReason = "Image is blurry"
            qualityScore -= 0.25
        } else {
            os_log(.debug, log: .default, "Quality check passed: Image is sharp (Laplacian1=%{public}f)", laplacian1)
        }

        // Ensure score within [0,1]
        qualityScore = qualityScore.clamped(to: 0...1)

        return FaceQualityResult(
            isGoodQuality: issues.isEmpty,
            qualityScore: qualityScore,
            issues: issues,
            failureReason: failureReason
        )
    }

    // Standard Laplacian-based sharpness calculation
    private func computeLaplacian1Sharpness(image: CGImage) -> Double {
        let width = image.width
        let height = image.height

        // Convert to grayscale
        var gray = [[Double]](repeating: [Double](repeating: 0, count: width), count: height)
        guard let pixelData = image.dataProvider?.data,
              let data = CFDataGetBytePtr(pixelData) else {
            os_log(.error, log: .default, "Failed to access pixel data for Laplacian1")
            return 0.0
        }

        let bytesPerPixel = image.bitsPerPixel / 8
        for y in 0..<height {
            for x in 0..<width {
                let pixelOffset = (y * image.bytesPerRow) + (x * bytesPerPixel)
                let r = Double(data[pixelOffset])
                let g = Double(data[pixelOffset + 1])
                let b = Double(data[pixelOffset + 2])
                gray[y][x] = 0.299 * r + 0.587 * g + 0.114 * b
            }
        }

        // Laplacian kernel
        let kernel = [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ]

        // Apply kernel
        var laplacian = [[Double]](repeating: [Double](repeating: 0, count: width), count: height)
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                var sum: Double = 0
                for i in -1...1 {
                    for j in -1...1 {
                        sum += Double(kernel[i + 1][j + 1]) * gray[y + i][x + j]
                    }
                }
                laplacian[y][x] = sum
            }
        }

        // Calculate variance (mean squared value)
        let flatValues = laplacian.flatMap { $0 }
        let variance = flatValues.map { $0 * $0 }.reduce(0, +) / Double(flatValues.count)
        return variance
    }
}

// Extension to add clamping functionality
extension Comparable {
    func clamped(to limits: ClosedRange<Self>) -> Self {
        return min(max(self, limits.lowerBound), limits.upperBound)
    }
}

