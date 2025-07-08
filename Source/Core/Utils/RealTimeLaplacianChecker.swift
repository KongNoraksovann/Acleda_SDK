import Foundation
import UIKit
import MLKitFaceDetection
import os.log

class RealTimeLaplacianChecker {
    
    // Make threshold accessible for external reference
    let laplacianThreshold: Double
    
    private let TAG = "FaceQualityChecker"
    
    // Add an initializer to allow custom threshold
    init(laplacianThreshold: Double = 100.0) {
        self.laplacianThreshold = laplacianThreshold
        os_log(.debug, log: .default, "%{public}s: Initialized with Laplacian threshold = %.2f", TAG, laplacianThreshold)
    }
    
    private func imageAgcwd(
        gray: [[Double]],
        width: Int,
        height: Int,
        a: Double = 0.25,
        truncatedCdf: Bool = false
    ) -> [[Double]] {
        var hist = [Int](repeating: 0, count: 256)
        for y in 0..<height {
            for x in 0..<width {
                hist[Int(gray[y][x])] += 1
            }
        }
        
        // Create CDF function equivalent to Kotlin's scan
        var cdf = [Int](repeating: 0, count: hist.count)
        var sum = 0
        for i in 0..<hist.count {
            sum += hist[i]
            cdf[i] = sum
        }
        
        let cdfMax = cdf.max() ?? 1
        let cdfNormalized = cdf.map { Double($0) / Double(cdfMax) }
        
        let probNormalized = hist.map { Double($0) / Double(width * height) }
        let probMin = probNormalized.min() ?? 0.0
        let probMax = probNormalized.max() ?? 1.0
        
        var pnTemp = probNormalized.map { ($0 - probMin) / (probMax - probMin) }
        for i in 0..<pnTemp.count {
            if pnTemp[i] > 0 {
                pnTemp[i] = probMax * pow(pnTemp[i], a)
            } else if pnTemp[i] < 0 {
                pnTemp[i] = probMax * (-(pow((-pnTemp[i]), a)))
            } else {
                pnTemp[i] = 0.0
            }
        }
        let pnSum = pnTemp.reduce(0, +)
        let probNormalizedWd = pnTemp.map { $0 / pnSum }
        
        // Create CDF for probNormalizedWd
        var cdfProbNormalizedWd = [Double](repeating: 0, count: probNormalizedWd.count)
        var cdfSum = 0.0
        for i in 0..<probNormalizedWd.count {
            cdfSum += probNormalizedWd[i]
            cdfProbNormalizedWd[i] = cdfSum
        }
        
        let inverseCdf: [Double]
        if truncatedCdf {
            inverseCdf = cdfProbNormalizedWd.map { max(0.5, 1.0 - $0) }
        } else {
            inverseCdf = cdfProbNormalizedWd.map { 1.0 - $0 }
        }
        
        var result = Array(repeating: Array(repeating: 0.0, count: width), count: height)
        for y in 0..<height {
            for x in 0..<width {
                let intensity = Int(gray[y][x])
                result[y][x] = Double(Int(255 * pow(Double(intensity) / 255.0, inverseCdf[intensity])))
            }
        }
        return result
    }
    
    private func processBright(gray: [[Double]], width: Int, height: Int) -> [[Double]] {
        var negative = Array(repeating: Array(repeating: 0.0, count: width), count: height)
        for y in 0..<height {
            for x in 0..<width {
                negative[y][x] = 255.0 - gray[y][x]
            }
        }
        
        let agcwd = imageAgcwd(gray: negative, width: width, height: height, a: 0.25, truncatedCdf: false)
        
        var result = Array(repeating: Array(repeating: 0.0, count: width), count: height)
        for y in 0..<height {
            for x in 0..<width {
                result[y][x] = 255.0 - agcwd[y][x]
            }
        }
        return result
    }
    
    private func processDimmed(gray: [[Double]], width: Int, height: Int) -> [[Double]] {
        return imageAgcwd(gray: gray, width: width, height: height, a: 0.55, truncatedCdf: true)
    }
    
    private func adjustBrightness(src: UIImage, brightnessOffset: Int) -> UIImage? {
        guard let cgImage = src.cgImage else { return src }
        let width = cgImage.width
        let height = cgImage.height
        
        guard let pixelData = src.pixelData() else { return src }

        var gray = Array(repeating: Array(repeating: 0.0, count: width), count: height)
        var meanIntensity = 0.0
        
        for y in 0..<height {
            for x in 0..<width {
                let pixel = pixelData[y * width + x]
                let r = Double((pixel >> 16) & 0xFF)
                let g = Double((pixel >> 8) & 0xFF)
                let b = Double(pixel & 0xFF)
                gray[y][x] = 0.299 * r + 0.587 * g + 0.114 * b
                meanIntensity += gray[y][x]
            }
        }
        meanIntensity /= Double(width * height)
        
        let threshold = 0.2
        let expIn = 112.0
        let t = (meanIntensity - expIn) / expIn
        
        var processedGray: [[Double]]
        if t < -threshold {
            os_log(.debug, log: .default, "%{public}s: Applying dimmed image processing", TAG)
            processedGray = processDimmed(gray: gray, width: width, height: height)
        } else if t > threshold {
            os_log(.debug, log: .default, "%{public}s: Applying bright image processing", TAG)
            processedGray = processBright(gray: gray, width: width, height: height)
        } else {
            os_log(.debug, log: .default, "%{public}s: No brightness adjustment needed", TAG)
            processedGray = gray
        }
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo: UInt32 = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        
        var processedPixels = [UInt32](repeating: 0, count: width * height)
        
        for y in 0..<height {
            for x in 0..<width {
                let value = Int(max(0, min(processedGray[y][x], 255)))
                let originalPixel = pixelData[y * width + x]
                let a = (originalPixel >> 24) & 0xFF
                processedPixels[y * width + x] = (a << 24) | (UInt32(value) << 16) | (UInt32(value) << 8) | UInt32(value)
            }
        }
        
        guard let context = CGContext(
            data: &processedPixels,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo
        ) else { return src }
        
        guard let newCGImage = context.makeImage() else { return src }
        return UIImage(cgImage: newCGImage, scale: src.scale, orientation: src.imageOrientation)
    }
    
    func checkFaceQuality(
        face: Face,
        bitmap: UIImage,
        brightnessOffset: Int = 0
    ) -> FaceQualityResult {
        var issues = [QualityIssue]()
        var failureReason: String? = nil
        var qualityScore: Float = 1.0
        
        let processedBitmap: UIImage
        if brightnessOffset != 0 {
            processedBitmap = adjustBrightness(src: bitmap, brightnessOffset: brightnessOffset) ?? bitmap
        } else {
            processedBitmap = bitmap
        }
        
        let laplacian1 = computeLaplacian1Sharpness(bitmap: processedBitmap)
        
        os_log(.debug, log: .default, "%{public}s: After brightness(%d): Lap1=%.2f (thr=%.2f)",
               TAG, brightnessOffset, laplacian1, laplacianThreshold)
        
        if laplacian1 < laplacianThreshold {
            os_log(.debug, log: .default, "%{public}s: Quality check failed: Image is blurry (Laplacian1=%.2f)",
                   TAG, laplacian1)
            issues.append(.blurryFace)
            failureReason = "Image is too blurry"
            qualityScore -= 0.25
        } else {
            os_log(.debug, log: .default, "%{public}s: Quality check passed: Image is sharp (Laplacian1=%.2f)",
                   TAG, laplacian1)
        }
        
        qualityScore = max(0.0, min(qualityScore, 1.0))
        
        return FaceQualityResult(
            isGoodQuality: issues.isEmpty,
            qualityScore: qualityScore, issues: issues,
            failureReason: failureReason
        )
    }
    
    // Add a method to check image sharpness directly, similar to the Android implementation

    func checkImageSharpness(bitmap: UIImage) -> (isSharp: Bool, score: Float) {
        let laplacianScore = computeLaplacian1Sharpness(bitmap: bitmap)
        let isSharp = laplacianScore >= laplacianThreshold
        let normalizedScore = Float(min(1.0, laplacianScore / laplacianThreshold))
        
        os_log(.debug, log: .default, "%{public}s: Image sharpness check - Score: %.2f, Threshold: %.2f, IsSharp: %@",
               TAG, laplacianScore, laplacianThreshold, isSharp ? "YES" : "NO")
        
        return (isSharp, normalizedScore)
    }
    
    func computeLaplacian1Sharpness(bitmap: UIImage) -> Double {
        guard let cgImage = bitmap.cgImage else { return 0.0 }
        let width = cgImage.width
        let height = cgImage.height
        
        guard let pixelData = bitmap.pixelData() else { return 0.0 }
        
        var gray = Array(repeating: Array(repeating: 0.0, count: width), count: height)
        
        for y in 0..<height {
            for x in 0..<width {
                let pixel = pixelData[y * width + x]
                let r = Double((pixel >> 16) & 0xFF)
                let g = Double((pixel >> 8) & 0xFF)
                let b = Double(pixel & 0xFF)
                gray[y][x] = 0.299 * r + 0.587 * g + 0.114 * b
            }
        }
        
        let kernel = [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ]
        
        var laplacian = Array(repeating: Array(repeating: 0.0, count: width), count: height)
        for y in 1..<(height-1) {
            for x in 1..<(width-1) {
                var sum = 0.0
                for i in -1...1 {
                    for j in -1...1 {
                        sum += Double(kernel[i+1][j+1]) * gray[y+i][x+j]
                    }
                }
                laplacian[y][x] = sum
            }
        }
        
        // Flatten the 2D array to 1D and calculate mean of squared values
        var flatValues = [Double]()
        for row in laplacian {
            flatValues.append(contentsOf: row)
        }
        
        let squaredValues = flatValues.map { $0 * $0 }
        return squaredValues.reduce(0.0, +) / Double(squaredValues.count)
    }
}

//// Extension to UIImage for pixelData method
//extension UIImage {
//    func pixelData() -> [UInt32]? {
//        guard let cgImage = cgImage else { return nil }
//        let width = cgImage.width
//        let height = cgImage.height
//        let bytesPerPixel = 4
//        let bytesPerRow = bytesPerPixel * width
//        let bitsPerComponent = 8
//        var pixelData = [UInt32](repeating: 0, count: width * height)
//        
//        let colorSpace = CGColorSpaceCreateDeviceRGB()
//        let bitmapInfo: UInt32 = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
//        guard let context = CGContext(
//            data: &pixelData,
//            width: width,
//            height: height,
//            bitsPerComponent: bitsPerComponent,
//            bytesPerRow: bytesPerRow,
//            space: colorSpace,
//            bitmapInfo: bitmapInfo
//        ) else { return nil }
//        
//        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
//        return pixelData
//    }
//}
