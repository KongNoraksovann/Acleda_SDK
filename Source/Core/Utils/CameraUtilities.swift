import AVFoundation
import UIKit
import os.log

enum CameraUtilities {
    static func calculateMovement(_ history: [CGRect]) -> Float {
        guard history.count > 1 else { return 0.0 }
        let last = history.last!
        let prev = history[history.count - 2]
        let dx = Float(abs(last.origin.x - prev.origin.x))
        let dy = Float(abs(last.origin.y - prev.origin.y))
        return dx + dy
    }

    static func flipImageIfNeeded(_ image: UIImage, isFrontCamera: Bool) -> UIImage {
        guard isFrontCamera else { return image }
        guard let cgImage = image.cgImage else { return image }
        return UIImage(cgImage: cgImage, scale: image.scale, orientation: .leftMirrored)
    }

    static func toUIImage(_ sampleBuffer: CMSampleBuffer, isFrontCamera: Bool, normalizeOrientation: Bool = false) -> UIImage? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            os_log(.error, log: .default, "Camera: Failed to get pixel buffer from sample buffer")
            return nil
        }

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext(options: nil)
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            os_log(.error, log: .default, "Camera: Failed to create CGImage from CVPixelBuffer")
            return nil
        }

        let orientation: UIImage.Orientation = normalizeOrientation ? .up : (isFrontCamera ? .leftMirrored : .right)
        let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: orientation)
        os_log(.info, log: .default, "Camera: Converted CMSampleBuffer to UIImage: %fx%f, orientation=%d", uiImage.size.width, uiImage.size.height, uiImage.imageOrientation.rawValue)
        return uiImage
    }

    static func videoOrientationFromDeviceOrientation(isFrontCamera: Bool) -> AVCaptureVideoOrientation {
        let deviceOrientation = UIDevice.current.orientation
        var videoOrientation: AVCaptureVideoOrientation

        switch deviceOrientation {
        case .portrait:
            videoOrientation = .portrait
        case .portraitUpsideDown:
            videoOrientation = .portraitUpsideDown
        case .landscapeLeft:
            videoOrientation = isFrontCamera ? .landscapeRight : .landscapeLeft
        case .landscapeRight:
            videoOrientation = isFrontCamera ? .landscapeLeft : .landscapeRight
        default:
            videoOrientation = .portrait
        }

        return videoOrientation
    }
}
