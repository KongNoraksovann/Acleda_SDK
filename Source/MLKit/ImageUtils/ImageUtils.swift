import Foundation
import UIKit
import os.log

@available(iOS 13.0, *)
struct ImageUtils {
    private static let TAG = "ImageUtils"
    private static let log = OSLog(subsystem: "com.example.faceverification.mlkit", category: "ImageUtils")
    
    /// Apply the resize and center crop transformations
    /// Matches Python transforms.Resize(256) followed by transforms.CenterCrop(224)
    ///
    /// - Parameter image: Input CGImage
    /// - Returns: Processed CGImage with standard size (224x224)
    static func applyResizeAndCenterCrop(image: CGImage) -> CGImage {
        do {
            let uiImage = UIImage(cgImage: image)
            
            // Step 1: Resize to 256x256
            let targetSize = CGSize(width: 256, height: 256)
            UIGraphicsBeginImageContextWithOptions(targetSize, false, 1.0)
            uiImage.draw(in: CGRect(origin: .zero, size: targetSize))
            guard let resizedImage = UIGraphicsGetImageFromCurrentImageContext(),
                  let resizedCGImage = resizedImage.cgImage else {
                os_log(.error, log: log, "%{public}s: Failed to resize image to 256x256", TAG)
                UIGraphicsEndImageContext()
                return createFallbackImage()
            }
            UIGraphicsEndImageContext()
            os_log(.debug, log: log, "%{public}s: Resized image size: %dx%d", TAG, resizedImage.size.width, resizedImage.size.height)
            
            // Step 2: Center crop to 224x224
            let cropSize = CGSize(width: 224, height: 224)
            let x = (resizedImage.size.width - cropSize.width) / 2
            let y = (resizedImage.size.height - cropSize.height) / 2
            let cropRect = CGRect(x: x, y: y, width: cropSize.width, height: cropSize.height)
            
            guard let croppedCGImage = resizedCGImage.cropping(to: cropRect) else {
                os_log(.error, log: log, "%{public}s: Failed to crop image to 224x224", TAG)
                return createFallbackImage()
            }
            
            os_log(.debug, log: log, "%{public}s: Cropped image size: %dx%d", TAG, 224, 224)
            return croppedCGImage
        } catch {
            os_log(.error, log: log, "%{public}s: Error in applyResizeAndCenterCrop: %{public}s", TAG, error.localizedDescription)
            return createFallbackImage()
        }
    }
    
    /// Crop exactly to the ML Kit bounding box (zero margin)
    ///
    /// - Parameters:
    ///   - image: Input CGImage
    ///   - boundingBox: Face bounding box
    /// - Returns: Cropped face or nil if invalid dimensions
    static func cropfitFaceTightly(image: CGImage, boundingBox: CGRect) -> CGImage? {
        do {
            // Validate input image
            guard image.width > 0, image.height > 0 else {
                os_log(.error, log: log, "%{public}s: Invalid image dimensions: %dx%d", TAG, image.width, image.height)
                return nil
            }
            
            // Validate bounding box
            guard boundingBox.width > 0, boundingBox.height > 0 else {
                os_log(.error, log: log, "%{public}s: Invalid bounding box dimensions: %fx%f", TAG, boundingBox.width, boundingBox.height)
                return nil
            }
            
            // Clamp to image bounds
            let left = Int(max(0, boundingBox.origin.x))
            let top = Int(max(0, boundingBox.origin.y))
            let right = Int(min(boundingBox.maxX, CGFloat(image.width)))
            let bottom = Int(min(boundingBox.maxY, CGFloat(image.height)))
            
            let width = right - left
            let height = bottom - top
            
            os_log(.debug, log: log, "%{public}s: Tight crop box: left=%d top=%d width=%d height=%d", TAG, left, top, width, height)
            
            guard width > 0, height > 0 else {
                os_log(.error, log: log, "%{public}s: Invalid crop dimensions: %dx%d", TAG, width, height)
                return nil
            }
            
            guard left + width <= image.width, top + height <= image.height else {
                os_log(.error, log: log, "%{public}s: Crop region exceeds image bounds", TAG)
                return nil
            }
            
            let cropRect = CGRect(x: CGFloat(left), y: CGFloat(top), width: CGFloat(width), height: CGFloat(height))
            guard let croppedCGImage = image.cropping(to: cropRect) else {
                os_log(.error, log: log, "%{public}s: Failed to crop image to rect: x=%d, y=%d, w=%d, h=%d",
                       TAG, left, top, width, height)
                return nil
            }
            
            return croppedCGImage
        } catch {
            os_log(.error, log: log, "%{public}s: Error in cropFaceTightly: %{public}s", TAG, error.localizedDescription)
            return nil
        }
    }
    
    /// Compress image for fallback detection
    ///
    /// - Parameters:
    ///   - image: Input CGImage
    ///   - quality: Compression quality (0-100)
    /// - Returns: Compressed CGImage
    static func compressImage(_ image: CGImage, quality: Int) -> CGImage {
        do {
            let uiImage = UIImage(cgImage: image)
            let compressionQuality = CGFloat(quality) / 100.0
            guard let data = uiImage.jpegData(compressionQuality: compressionQuality),
                  let compressedImage = UIImage(data: data, scale: 1.0),
                  let compressedCGImage = compressedImage.cgImage else {
                os_log(.error, log: log, "%{public}s: Failed to compress image with quality %d", TAG, quality)
                return image // Return original image on error
            }
            os_log(.debug, log: log, "%{public}s: Compressed image (quality=%d): %dx%d",
                   TAG, quality, compressedImage.size.width, compressedImage.size.height)
            return compressedCGImage
        } catch {
            os_log(.error, log: log, "%{public}s: Error in compressImage: %{public}s", TAG, error.localizedDescription)
            return image // Return original image on error
        }
    }
    
    /// Create an image with expanded margins around a face
    ///
    /// - Parameters:
    ///   - image: Original CGImage
    ///   - boundingBox: Face bounding box
    ///   - marginPercent: Margin to add around face (percentage of face dimensions)
    /// - Returns: Image with margins around face
    static func cropFaceWithMargin(image: CGImage, boundingBox: CGRect, marginPercent: Float = 0.3) -> CGImage? {
        do {
            // Validate input image
            guard image.width > 0, image.height > 0 else {
                os_log(.error, log: log, "%{public}s: Invalid image dimensions: %dx%d", TAG, image.width, image.height)
                return nil
            }
            
            // Validate bounding box
            guard boundingBox.width > 0, boundingBox.height > 0 else {
                os_log(.error, log: log, "%{public}s: Invalid bounding box dimensions: %fx%f", TAG, boundingBox.width, boundingBox.height)
                return nil
            }
            
            let width = boundingBox.width
            let height = boundingBox.height
            
            let widthMargin = Int(width * CGFloat(marginPercent))
            let heightMargin = Int(height * CGFloat(marginPercent))
            
            // Calculate new bounds with margins
            let left = Int(max(0, boundingBox.origin.x - CGFloat(widthMargin)))
            let top = Int(max(0, boundingBox.origin.y - CGFloat(heightMargin)))
            let right = Int(min(boundingBox.maxX + CGFloat(widthMargin), CGFloat(image.width)))
            let bottom = Int(min(boundingBox.maxY + CGFloat(heightMargin), CGFloat(image.height)))
            
            let croppedWidth = right - left
            let croppedHeight = bottom - top
            
            os_log(.debug, log: log, "%{public}s: Margin crop box: left=%d top=%d width=%d height=%d",
                   TAG, left, top, croppedWidth, croppedHeight)
            
            guard croppedWidth > 0, croppedHeight > 0 else {
                os_log(.error, log: log, "%{public}s: Invalid crop dimensions with margin: %dx%d", TAG, croppedWidth, croppedHeight)
                return nil
            }
            
            guard left + croppedWidth <= image.width, top + croppedHeight <= image.height else {
                os_log(.error, log: log, "%{public}s: Margin crop region exceeds image bounds", TAG)
                return nil
            }
            
            let cropRect = CGRect(x: CGFloat(left), y: CGFloat(top), width: CGFloat(croppedWidth), height: CGFloat(croppedHeight))
            guard let croppedCGImage = image.cropping(to: cropRect) else {
                os_log(.error, log: log, "%{public}s: Failed to crop image to rect: x=%d, y=%d, w=%d, h=%d",
                       TAG, left, top, croppedWidth, croppedHeight)
                return nil
            }
            
            return croppedCGImage
        } catch {
            os_log(.error, log: log, "%{public}s: Error in cropFaceWithMargin: %{public}s", TAG, error.localizedDescription)
            return nil
        }
    }
    
    /// Create a fallback solid black image (224x224)
    private static func createFallbackImage() -> CGImage {
        let size = CGSize(width: 224, height: 224)
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        let context = UIGraphicsGetCurrentContext()
        context?.setFillColor(UIColor.black.cgColor)
        context?.fill(CGRect(origin: .zero, size: size))
        let fallbackImage = UIGraphicsGetImageFromCurrentImageContext()?.cgImage ?? {
            // If even this fails, create a minimal CGImage
            let bitmapInfo = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
            let context = CGContext(data: nil, width: 224, height: 224, bitsPerComponent: 8, bytesPerRow: 224 * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: bitmapInfo)!
            context.setFillColor(UIColor.black.cgColor)
            context.fill(CGRect(x: 0, y: 0, width: 224, height: 224))
            return context.makeImage()!
        }()
        UIGraphicsEndImageContext()
        os_log(.debug, log: log, "%{public}s: Created fallback image: 224x224", TAG)
        return fallbackImage
    }
}
