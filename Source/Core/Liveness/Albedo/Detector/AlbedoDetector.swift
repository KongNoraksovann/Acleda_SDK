import Foundation
import UIKit
import Accelerate
import os.log



// MARK: - AlbedoDetector Class

/// A class responsible for detecting spoofing attempts using albedo analysis on images.
@available(iOS 13.0, *)
class AlbedoDetector {
    
    // MARK: - Private Properties
    
    private let TAG = "AlbedoDetector"
    private let targetSize = 224
    private let brightnessThreshold = 200.0 // Threshold for detecting overexposure (e.g., flash)

    // MARK: - Public Methods
    
    /// Analyzes a `UIImage` to determine if it is a live image or a spoof attempt.
    ///
    /// - Parameters:
    ///   - bitmap: The input `UIImage` to be analyzed.
    /// - Returns: An `AlbedoResult` containing the analysis details and prediction.
    /// - Throws: An `AlbedoDetectorError` if the image is invalid or processing fails.
    func detectSpoof(bitmap: UIImage) throws -> AlbedoResult {
        os_log(.info, log: .default, "%{public}s: Starting albedo spoof detection", TAG)
        
        // Resize image to the target dimensions (224x224)
        guard let resizedBitmap = bitmap.resize(to: CGSize(width: targetSize, height: targetSize)) else {
            throw AlbedoDetectorError.processingFailed("Failed to resize bitmap.")
        }

        // Extract pixel data from the resized image
        guard let pixels = resizedBitmap.pixelData() else {
            throw AlbedoDetectorError.processingFailed("Could not extract pixel data from bitmap.")
        }
        
        os_log(.info, log: .default, "%{public}s: Image shape: %d x %d", TAG, targetSize, targetSize)
        
        // Deconstruct pixel data into separate RGB channels (0-255)
        let (originalR, originalG, originalB) = extractRGBChannels(from: pixels)
        
        // Normalize RGB channels to a 0-1 range
        let redChannelNormalized = originalR.map { $0 / 255.0 }
        let greenChannelNormalized = originalG.map { $0 / 255.0 }
        let blueChannelNormalized = originalB.map { $0 / 255.0 }
        
        // Calculate and log original brightness and contrast
        let originalBrightness = (originalR.average() + originalG.average() + originalB.average()) / 3.0
        let originalContrast = calculateContrast(r: originalR, g: originalG, b: originalB)
        
        os_log(.info, log: .default, "%{public}s: Original Brightness Score: %.2f", TAG, originalBrightness)
        os_log(.info, log: .default, "%{public}s: Original Contrast Score: %.2f", TAG, originalContrast)
        
        // Calculate albedo for each channel using the normalized values
        let albedoR = variance(data: redChannelNormalized)
        let albedoG = variance(data: greenChannelNormalized)
        let albedoB = variance(data: blueChannelNormalized)

        os_log(.info, log: .default, "%{public}s: Albedo (R): %.6f", TAG, albedoR)
        os_log(.info, log: .default, "%{public}s: Albedo (G): %.6f", TAG, albedoG)
        os_log(.info, log: .default, "%{public}s: Albedo (B): %.6f", TAG, albedoB)
        
        // Identify outliers in each channel
        let channelData: [String: [Double]] = [
            "Red": originalR,
            "Green": originalG,
            "Blue": originalB
        ]
        
        var outliersAbove: [String: Int] = [:]
        var channelBounds: [String: (upper: Double, lower: Double)] = [:]

        for (channelName, channel) in channelData {
            let sortedChannel = channel.sorted()
            let q25 = quantile(data: sortedChannel, q: 0.25)
            let q75 = quantile(data: sortedChannel, q: 0.75)
            let iqr = q75 - q25
            
            // Calculate upper bound using mean instead of Q75
            let mean = channel.average()
            let ub = mean + 1.5 * iqr
            let lb = q25 - 1.5 * iqr

            outliersAbove[channelName] = channel.count { $0 > ub }
            channelBounds[channelName] = (upper: ub, lower: lb)

            os_log(.info, log: .default, "%{public}s: %{public}s channel - Outliers above upper bound: %d", TAG, channelName, outliersAbove[channelName] ?? 0)
        }

        
        let isOverexposed = originalBrightness > brightnessThreshold
        let greenOutliers = outliersAbove["Green", default: 0]
        let blueOutliers = outliersAbove["Blue", default: 0]
        let isLive: Bool
        let prediction: String

        if isOverexposed {
            os_log(.info, log: .default, "%{public}s: Image classified as Spoof due to high brightness (%.2f)", TAG, originalBrightness)
            isLive = false
            prediction = "Spoof"
        } else {
            isLive = greenOutliers > 0 && blueOutliers > 0
            prediction = isLive ? "Live" : "Spoof"
            os_log(.info, log: .default, "%{public}s: Classification: %@ (Green: %d, Blue: %d)", TAG, prediction, greenOutliers, blueOutliers)
        }
        
        // Construct and return the final result object
        return AlbedoResult(
            prediction: prediction,
            isLive: isLive,
            albedoValueR: albedoR,
            albedoValueG: albedoG,
            albedoValueB: albedoB,
            redOutliers: outliersAbove["Red", default: 0],
            greenOutliers: greenOutliers,
            blueOutliers: blueOutliers,
            redBounds: channelBounds["Red", default: (0.0, 0.0)],
            greenBounds: channelBounds["Green", default: (0.0, 0.0)],
            blueBounds: channelBounds["Blue", default: (0.0, 0.0)],
            originalBrightness: originalBrightness,
            originalContrast: originalContrast
        )
    }
    
    // MARK: - Private Helper Methods
    
    /// Deconstructs an array of pixel values into separate R, G, and B channels.
    private func extractRGBChannels(from pixels: [UInt32]) -> (r: [Double], g: [Double], b: [Double]) {
        var rChannel = [Double]()
        var gChannel = [Double]()
        var bChannel = [Double]()
        rChannel.reserveCapacity(pixels.count)
        gChannel.reserveCapacity(pixels.count)
        bChannel.reserveCapacity(pixels.count)

        for pixel in pixels {
            rChannel.append(Double((pixel >> 16) & 0xFF))
            gChannel.append(Double((pixel >> 8) & 0xFF))
            bChannel.append(Double(pixel & 0xFF))
        }
        
        return (rChannel, gChannel, bChannel)
    }

    /// Calculates the variance of a dataset.
    private func variance(data: [Double]) -> Double {
        guard !data.isEmpty else { return 0.0 }
        let mean = data.average()
        let sumOfSquaredDiffs = data.map { pow($0 - mean, 2) }.reduce(0, +)
        return sumOfSquaredDiffs / Double(data.count)
    }
    
    /// Calculates the contrast of an image based on its RGB channels.
    private func calculateContrast(r: [Double], g: [Double], b: [Double]) -> Double {
        let allPixels = r + g + b
        guard !allPixels.isEmpty else { return 0.0 }
        let mean = allPixels.average()
        let variance = allPixels.map { pow($0 - mean, 2) }.average()
        return sqrt(variance)
    }

    /// Computes a specific quantile from a sorted dataset.
    private func quantile(data: [Double], q: Double) -> Double {
        guard !data.isEmpty else { return 0.0 }
        let pos = (Double(data.count) - 1) * q
        let index = Int(pos)
        let fraction = pos - Double(index)
        
        if index + 1 < data.count {
            return data[index] * (1 - fraction) + data[index + 1] * fraction
        } else {
            return data[index]
        }
    }
}

private extension Array where Element == Double {
    /// Calculates the average of the elements in the array.
    func average() -> Double {
        guard !isEmpty else { return 0.0 }
        return reduce(0, +) / Double(count)
    }
}

private extension Collection where Element: FloatingPoint {
    /// Calculates the average of the elements in the collection.
    func average() -> Element {
        guard !isEmpty else { return .zero }
        return reduce(.zero, +) / Element(count)
    }
}

private extension Double {
    /// Clips the double to a specified closed range.
    func clipped(to range: ClosedRange<Double>) -> Double {
        return min(max(self, range.lowerBound), range.upperBound)
    }
}

