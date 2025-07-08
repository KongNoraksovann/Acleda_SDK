//  LogUtils.swift
import Foundation
import os.log

/**
 * Utility class for consistent logging throughout the SDK
 */
@objc public class LogUtils: NSObject {
    private static let SDK_TAG_PREFIX = "FaceSDK-"
    private static var isDebugEnabled = false
    private static let log = OSLog(subsystem: "com.acleda.sdk", category: "ModelEncryption")
    
    /**
     * Enable or disable debug logging
     *
     * @param enabled True to enable debug logs, false to disable
     */
    @objc public static func setDebugEnabled(_ enabled: Bool) {
        isDebugEnabled = enabled
    }
    
    /**
     * Log a debug message
     *
     * @param tag Component tag
     * @param message Log message
     */
    @objc public static func d(_ tag: String, _ message: String) {
        if isDebugEnabled {
            os_log("[DEBUG] %{public}@: %{public}@", log: log, type: .debug, tag, message)
            #if DEBUG
            print("D/\(tag): \(message)")
            #endif
        }
    }
    
    /**
     * Log an info message
     *
     * @param tag Component tag
     * @param message Log message
     */
    @objc public static func i(_ tag: String, _ message: String) {
        os_log("[INFO] %{public}@: %{public}@", log: log, type: .info, tag, message)
        #if DEBUG
        print("I/\(tag): \(message)")
        #endif
    }
    
    /**
     * Log a warning message
     *
     * @param tag Component tag
     * @param message Log message
     */
    @objc public static func w(_ tag: String, _ message: String) {
        os_log("[WARN] %{public}@: %{public}@", log: log, type: .error, tag, message)
        #if DEBUG
        print("W/\(tag): \(message)")
        #endif
    }
    
    /**
     * Log an error message
     *
     * @param tag Component tag
     * @param message Log message
     */
    @objc public static func e(_ tag: String, _ message: String, _ error: Error? = nil) {
        if let error = error {
            os_log("[ERROR] %{public}@: %{public}@ - %{public}@", log: log, type: .error, tag, message, error.localizedDescription)
            #if DEBUG
            print("E/\(tag): \(message) - \(error)")
            #endif
        } else {
            os_log("[ERROR] %{public}@: %{public}@", log: log, type: .error, tag, message)
            #if DEBUG
            print("E/\(tag): \(message)")
            #endif
        }
    }
}
