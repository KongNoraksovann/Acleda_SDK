import UIKit

public struct AlbedoResult {
    let prediction: String
    let isLive: Bool
    let albedoValueR: Double
    let albedoValueG: Double
    let albedoValueB: Double
    let redOutliers: Int
    let greenOutliers: Int
    let blueOutliers: Int
    let redBounds: (upper: Double, lower: Double)
    let greenBounds: (upper: Double, lower: Double)
    let blueBounds: (upper: Double, lower: Double)
    let originalBrightness: Double
//    let gammaBrightness: Double
    let originalContrast: Double
//    let gammaContrast: Double
}
