import Foundation
import UIKit

@objc public class BitmapUtils: NSObject {
    private static let TAG = "BitmapUtils"
    
    @objc public static let MIN_IMAGE_SIZE = 64
    @objc public static let MAX_IMAGE_SIZE = 4096
    
    @objc public static func validateImage(_ image: UIImage?) -> Bool {
        guard let image = image else {
            LogUtils.e(TAG, "Input image is null")
            return false
        }
        
        let width = Int(image.size.width * image.scale)
        let height = Int(image.size.height * image.scale)
        
        if width <= MIN_IMAGE_SIZE || height <= MIN_IMAGE_SIZE {
            LogUtils.e(TAG, "Image too small: \(width)x\(height)")
            return false
        }
        
        if width >= MAX_IMAGE_SIZE || height >= MAX_IMAGE_SIZE {
            LogUtils.e(TAG, "Image too large: \(width)x\(height)")
            return false
        }
        
        if image.cgImage == nil {
            LogUtils.e(TAG, "Image has no CGImage representation")
            return false
        }
        
        return true
    }
    
    @objc public static func resizeImage(_ image: UIImage, width: Int, height: Int) -> UIImage? {
        if Int(image.size.width * image.scale) == width && Int(image.size.height * image.scale) == height {
            return image
        }
        
        let size = CGSize(width: width, height: height)
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        defer { UIGraphicsEndImageContext() }
        
        image.draw(in: CGRect(origin: .zero, size: size))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
        
        LogUtils.d(TAG, "Image resized to: \(width)x\(height)")
        return resizedImage
    }
    
    @objc public static func calculateAverageBrightness(_ image: UIImage) -> Float {
        guard validateImage(image),
              let cgImage = image.cgImage,
              let pixelData = cgImage.dataProvider?.data,
              let data = CFDataGetBytePtr(pixelData) else {
            LogUtils.e(TAG, "Failed to access image data for brightness calculation")
            return 0.0
        }
        
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = cgImage.bitsPerPixel / 8
        let bytesPerRow = cgImage.bytesPerRow
        
        var totalBrightness: Float = 0.0
        var pixelCount: Float = 0.0
        
        let stepSize = max(1, min(width, height) / 50)
        
        for y in stride(from: 0, to: height, by: stepSize) {
            for x in stride(from: 0, to: width, by: stepSize) {
                let offset = y * bytesPerRow + x * bytesPerPixel
                let r = Float(data[offset]) / 255.0
                let g = Float(data[offset + 1]) / 255.0
                let b = Float(data[offset + 2]) / 255.0
                let brightness = 0.299 * r + 0.587 * g + 0.114 * b
                totalBrightness += brightness
                pixelCount += 1.0
            }
        }
        
        let avgBrightness = pixelCount > 0 ? (totalBrightness / pixelCount) * 255.0 : 0.0
        LogUtils.d(TAG, "Calculated average brightness: \(avgBrightness)")
        return avgBrightness
    }
}
