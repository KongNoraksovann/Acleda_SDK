import Foundation
import UIKit

@available(iOS 13.0, *)
public class LivenessSDKManager {
    // MARK: - Singleton
    public static let shared = LivenessSDKManager()
    
    private let tag = "LivenessSDKManager"
    
    // MARK: - Properties
    private var faceLivenessSDK: FaceLivenessSDK?
    private var realtimeLivenessSDK: FaceLivenessSDK?
    private var context: Bundle!
    public var isEnabled: Bool = true
    public var isOcclusionDetectionEnabled: Bool = true
    public var isRealtimeEnabled: Bool = true
    
    // MARK: - Initialization
    private init() {}
    
    // Create configs dynamically instead of lazy initialization
    private func createRealtimeConfig() -> FaceLivenessSDK.Config {
        return FaceLivenessSDK.Config.Builder()
            .setDebugLoggingEnabled(true)
            .setSkipOcclusionCheck(!isOcclusionDetectionEnabled)
//            .setSkipAlbedoCheck(false)
            .build()
    }
    
    private func createVerificationConfig() -> FaceLivenessSDK.Config {
        return FaceLivenessSDK.Config.Builder()
            .setDebugLoggingEnabled(true)
            .setSkipOcclusionCheck(!isOcclusionDetectionEnabled)
//            .setSkipAlbedoCheck(false)
            .build()
    }
    
    public func initialize(context: Bundle = .main, occlusionEnabled: Bool = true, realtimeEnabled: Bool = true) {
        self.context = context
        self.isOcclusionDetectionEnabled = occlusionEnabled
        self.isRealtimeEnabled = realtimeEnabled
        
        // Initialize both SDK instances with their respective configurations
        self.faceLivenessSDK = FaceLivenessSDK.create(context, config: createVerificationConfig())
        
        if realtimeEnabled {
            self.realtimeLivenessSDK = FaceLivenessSDK.create(context, config: createRealtimeConfig())
        }
        
        LogUtils.d(tag, "Initialized with occlusion detection \(occlusionEnabled ? "enabled" : "disabled") and real-time checks \(realtimeEnabled ? "enabled" : "disabled")")
    }
    
    public func updateOcclusionConfig(enabled: Bool) {
        guard context != nil else {
            LogUtils.w(tag, "Cannot update occlusion config: context not initialized")
            return
        }
        
        self.isOcclusionDetectionEnabled = enabled
        
        // Recreate both SDK instances with updated occlusion settings
        faceLivenessSDK?.close()
        faceLivenessSDK = FaceLivenessSDK.create(context, config: createVerificationConfig())
        
        if realtimeLivenessSDK != nil {
            realtimeLivenessSDK?.close()
            realtimeLivenessSDK = FaceLivenessSDK.create(context, config: createRealtimeConfig())
        }
        
        LogUtils.d(tag, "Updated occlusion detection to \(enabled ? "enabled" : "disabled")")
    }
    
    public func updateRealtimeConfig(enabled: Bool) {
        guard context != nil else {
            LogUtils.w(tag, "Cannot update real-time config: context not initialized")
            return
        }
        
        self.isRealtimeEnabled = enabled
        
        if !enabled && realtimeLivenessSDK != nil {
            realtimeLivenessSDK?.close()
            realtimeLivenessSDK = nil
            LogUtils.d(tag, "Disabled real-time liveness checks")
        } else if enabled && realtimeLivenessSDK == nil {
            realtimeLivenessSDK = FaceLivenessSDK.create(context, config: createRealtimeConfig())
            LogUtils.d(tag, "Enabled real-time liveness checks")
        }
    }
    
    // Standard liveness check for final verification
    public func checkLiveness(image: UIImage) async throws -> (Result<FaceLivenessModel, Error>, [String: Float]?, [String: Float]?) {
        LogUtils.d(tag, "checkLivenessWithDetailedScores called with enabled=\(isEnabled), occlusionDetection=\(isOcclusionDetectionEnabled)")
        
        if !isEnabled {
            let model = FaceLivenessModel(
                prediction: "Live",
                confidence: 1.0,
                failureReason: nil
            )
            return (
                .success(model),
                ["liveScore": 1.0, "spoofScore": 0.0],
                nil
            )
        }
        
        guard let faceLivenessSDK = faceLivenessSDK else {
            let error = NSError(
                domain: "LivenessSDKManager",
                code: -1001,
                userInfo: [NSLocalizedDescriptionKey: "FaceLivenessSDK not initialized"]
            )
            LogUtils.e(tag, "Liveness check error: \(error.localizedDescription)")
            return (.failure(error), nil, nil)
        }
        
        do {
            let (livenessModel, livenessScores, occlusionScores) = try await faceLivenessSDK.detectLiveness(image)
            LogUtils.d(tag, "SDK Result: prediction=\(livenessModel.prediction), confidence=\(livenessModel.confidence)")
            LogUtils.d(tag, "Detailed scores - Liveness: \(livenessScores?.description ?? "null"), Occlusion: \(occlusionScores?.description ?? "null")")
            return (.success(livenessModel), livenessScores, occlusionScores)
        } catch {
            LogUtils.e(tag, "Liveness check error: \(error.localizedDescription)", error)
            return (.failure(error), nil, nil)
        }
    }
    
    public func close() {
        do {
            faceLivenessSDK?.close()
            faceLivenessSDK = nil
            
            realtimeLivenessSDK?.close()
            realtimeLivenessSDK = nil
            
            LogUtils.d(tag, "Closed LivenessSDKManager")
        } catch {
            LogUtils.e(tag, "Error closing SDK resources: \(error.localizedDescription)", error)
        }
    }
}
