import Foundation
import UIKit
// Assuming FaceLivenessModel is provided by the SDK
// import com.acleda.facelivenesssdk

/// Singleton to hold registration data between screens
@objc public class RegistrationDataHolder: NSObject {
    /// Shared singleton instance
    @objc public static let shared = RegistrationDataHolder()
    
    /// Captured face image
    @objc public var faceImage: UIImage?
    
    /// Result of face liveness detection
    @objc public var livenessResult: FaceLivenessModel?
    
    /// Whether liveness detection is enabled
    @objc public var isLivenessEnabled: Bool = false
    
    /// Face embedding from API
    @objc public var apiEmbedding: [Double]?
    
    /// Local face embedding
    @objc public var localEmbedding: [Double]?
    
    /// User ID
    @objc public var userId: String?
    
    /// User name
    @objc public var userName: String?
    
    private override init() {
        super.init()
    }
    
    /// Clears all registration data
    @objc public func clear() {
        faceImage = nil
        livenessResult = nil
        isLivenessEnabled = false
        apiEmbedding = nil
        localEmbedding = nil
        userId = nil
        userName = nil
    }
}
