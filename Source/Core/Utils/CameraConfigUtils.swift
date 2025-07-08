import Foundation
import AVFoundation
import UIKit

/**
 * Utility class for safe camera configuration
 */
@objc public class CameraConfigUtils: NSObject {
    private static let TAG = "CameraConfigUtils"
    
    /**
     * Safely configures camera focus mode after checking if it's supported
     *
     * @param device The AVCaptureDevice to configure
     * @param focusMode The desired focus mode to set
     * @return Boolean indicating success
     */
    @objc public static func safelySetFocusMode(_ focusMode: AVCaptureDevice.FocusMode, for device: AVCaptureDevice) -> Bool {
        guard device.isFocusModeSupported(focusMode) else {
            LogUtils.w(TAG, "Focus mode \(focusMode.rawValue) not supported on this device")
            return false
        }
        
        do {
            try device.lockForConfiguration()
            device.focusMode = focusMode
            
            // If focus points are supported, set the focus point to the center
            if device.isFocusPointOfInterestSupported {
                device.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
            }
            
            device.unlockForConfiguration()
            LogUtils.d(TAG, "Successfully set focus mode to \(focusMode.rawValue)")
            return true
        } catch {
            LogUtils.e(TAG, "Failed to lock device for focus configuration: \(error.localizedDescription)")
            return false
        }
    }
    
    /**
     * Safely configures camera exposure mode after checking if it's supported
     *
     * @param device The AVCaptureDevice to configure
     * @param exposureMode The desired exposure mode to set
     * @return Boolean indicating success
     */
    @objc public static func safelySetExposureMode(_ exposureMode: AVCaptureDevice.ExposureMode, for device: AVCaptureDevice) -> Bool {
        guard device.isExposureModeSupported(exposureMode) else {
            LogUtils.w(TAG, "Exposure mode \(exposureMode.rawValue) not supported on this device")
            return false
        }
        
        do {
            try device.lockForConfiguration()
            device.exposureMode = exposureMode
            
            // If exposure points are supported, set the exposure point to the center
            if device.isExposurePointOfInterestSupported {
                device.exposurePointOfInterest = CGPoint(x: 0.5, y: 0.5)
            }
            
            device.unlockForConfiguration()
            LogUtils.d(TAG, "Successfully set exposure mode to \(exposureMode.rawValue)")
            return true
        } catch {
            LogUtils.e(TAG, "Failed to lock device for exposure configuration: \(error.localizedDescription)")
            return false
        }
    }
    
    /**
     * Configures optimal camera settings for face detection with fallbacks
     * If a setting isn't supported, it will try alternative modes
     *
     * @param device The AVCaptureDevice to configure
     * @return Boolean indicating success
     */
    @objc public static func configureOptimalCameraSettings(for device: AVCaptureDevice) -> Bool {
        // Try to lock the device for configuration
        do {
            try device.lockForConfiguration()
            
            // Configure focus mode with fallbacks
            if device.isFocusModeSupported(.continuousAutoFocus) {
                device.focusMode = .continuousAutoFocus
                LogUtils.d(TAG, "Set focus mode to continuousAutoFocus")
            } else if device.isFocusModeSupported(.autoFocus) {
                device.focusMode = .autoFocus
                LogUtils.d(TAG, "Set focus mode to autoFocus")
            } else if device.isFocusModeSupported(.locked) {
                device.focusMode = .locked
                LogUtils.d(TAG, "Set focus mode to locked")
            } else {
                LogUtils.w(TAG, "No supported focus modes found")
            }
            
            // Configure exposure mode with fallbacks
            if device.isExposureModeSupported(.continuousAutoExposure) {
                device.exposureMode = .continuousAutoExposure
                LogUtils.d(TAG, "Set exposure mode to continuousAutoExposure")
            } else if device.isExposureModeSupported(.autoExpose) {
                device.exposureMode = .autoExpose
                LogUtils.d(TAG, "Set exposure mode to autoExpose")
            } else if device.isExposureModeSupported(.locked) {
                device.exposureMode = .locked
                LogUtils.d(TAG, "Set exposure mode to locked")
            } else {
                LogUtils.w(TAG, "No supported exposure modes found")
            }
            
            // Configure white balance mode with fallbacks
            if device.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
                device.whiteBalanceMode = .continuousAutoWhiteBalance
                LogUtils.d(TAG, "Set white balance mode to continuousAutoWhiteBalance")
            } else if device.isWhiteBalanceModeSupported(.autoWhiteBalance) {
                device.whiteBalanceMode = .autoWhiteBalance
                LogUtils.d(TAG, "Set white balance mode to autoWhiteBalance")
            } else if device.isWhiteBalanceModeSupported(.locked) {
                device.whiteBalanceMode = .locked
                LogUtils.d(TAG, "Set white balance mode to locked")
            } else {
                LogUtils.w(TAG, "No supported white balance modes found")
            }
            
            // Set focus and exposure points if supported
            if device.isFocusPointOfInterestSupported {
                device.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
                LogUtils.d(TAG, "Set focus point to center")
            }
            
            if device.isExposurePointOfInterestSupported {
                device.exposurePointOfInterest = CGPoint(x: 0.5, y: 0.5)
                LogUtils.d(TAG, "Set exposure point to center")
            }
            
            device.unlockForConfiguration()
            return true
            
        } catch {
            LogUtils.e(TAG, "Failed to lock device for camera configuration: \(error.localizedDescription)")
            return false
        }
    }
    
    /**
     * Logs all supported camera capabilities for debugging
     *
     * @param device The AVCaptureDevice to check
     */
    @objc public static func logCameraCapabilities(_ device: AVCaptureDevice) {
        let focusModes = [
            AVCaptureDevice.FocusMode.locked: "locked",
            AVCaptureDevice.FocusMode.autoFocus: "autoFocus",
            AVCaptureDevice.FocusMode.continuousAutoFocus: "continuousAutoFocus"
        ]
        
        let exposureModes = [
            AVCaptureDevice.ExposureMode.locked: "locked",
            AVCaptureDevice.ExposureMode.autoExpose: "autoExpose",
            AVCaptureDevice.ExposureMode.continuousAutoExposure: "continuousAutoExposure",
            AVCaptureDevice.ExposureMode.custom: "custom"
        ]
        
        let whiteBalanceModes = [
            AVCaptureDevice.WhiteBalanceMode.locked: "locked",
            AVCaptureDevice.WhiteBalanceMode.autoWhiteBalance: "autoWhiteBalance",
            AVCaptureDevice.WhiteBalanceMode.continuousAutoWhiteBalance: "continuousAutoWhiteBalance"
        ]
        
        LogUtils.i(TAG, "Camera capabilities for device: \(device.localizedName)")
        
        // Log supported focus modes
        var supportedFocusModes = [String]()
        for (mode, name) in focusModes {
            if device.isFocusModeSupported(mode) {
                supportedFocusModes.append(name)
            }
        }
        LogUtils.i(TAG, "Supported focus modes: \(supportedFocusModes.joined(separator: ", "))")
        
        // Log supported exposure modes
        var supportedExposureModes = [String]()
        for (mode, name) in exposureModes {
            if device.isExposureModeSupported(mode) {
                supportedExposureModes.append(name)
            }
        }
        LogUtils.i(TAG, "Supported exposure modes: \(supportedExposureModes.joined(separator: ", "))")
        
        // Log supported white balance modes
        var supportedWhiteBalanceModes = [String]()
        for (mode, name) in whiteBalanceModes {
            if device.isWhiteBalanceModeSupported(mode) {
                supportedWhiteBalanceModes.append(name)
            }
        }
        LogUtils.i(TAG, "Supported white balance modes: \(supportedWhiteBalanceModes.joined(separator: ", "))")
        
        // Log other capabilities
        LogUtils.i(TAG, "Supports focus point of interest: \(device.isFocusPointOfInterestSupported)")
        LogUtils.i(TAG, "Supports exposure point of interest: \(device.isExposurePointOfInterestSupported)")
        LogUtils.i(TAG, "Has flash: \(device.hasFlash)")
        LogUtils.i(TAG, "Has torch: \(device.hasTorch)")
    }
    
    /**
     * Gets the most capable camera device available
     *
     * @param position The desired camera position (front or back)
     * @return The best available AVCaptureDevice or nil if none is available
     */
    @objc public static func getBestCamera(position: AVCaptureDevice.Position) -> AVCaptureDevice? {
        // First try to get a device with depth capabilities
        if #available(iOS 11.1, *) {
            let discoverySession = AVCaptureDevice.DiscoverySession(
                deviceTypes: [.builtInDualCamera, .builtInTrueDepthCamera, .builtInWideAngleCamera],
                mediaType: .video,
                position: position
            )
            
            // Return the first available device
            return discoverySession.devices.first
        } else {
            // Fallback for older iOS versions
            let devices = AVCaptureDevice.devices(for: .video)
            for device in devices {
                if device.position == position {
                    return device
                }
            }
            return nil
        }
    }
}
