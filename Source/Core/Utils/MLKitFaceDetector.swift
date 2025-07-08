//import MLKitVision
//import MLKitFaceDetection
//import os.log
//
//@available(iOS 13.0, *)
//extension MLKitFaceDetector {
//    struct LandmarkPoint {
//        let x: Float
//        let y: Float
//    }
//    
//    func detectFace(in image: CGImage) async -> (CGImage?, [[Float]])? {
//        let uiImage = UIImage(cgImage: image)
//        let visionImage = VisionImage(image: uiImage)
//        visionImage.orientation = .up
//        
//        let dateFormatter = DateFormatter()
//        dateFormatter.dateFormat = "hh:mm a z, MMMM dd, yyyy"
//        let currentTime = dateFormatter.string(from: Date())
//        os_log(.debug, log: log, "%{public}s: Detecting face for cropping and landmarks at %{public}s", TAG, currentTime)
//        
//        do {
//            let faces = try await processVisionImage(visionImage)
//            
//            if faces.isEmpty {
//                os_log(.debug, log: log, "%{public}s: No faces detected at %{public}s", TAG, currentTime)
//                return nil
//            }
//            
//            let face = faces[0]
//            let boundingBox = face.frame
//            let x = max(0, Int(boundingBox.origin.x))
//            let y = max(0, Int(boundingBox.origin.y))
//            let width = min(Int(boundingBox.width), image.width - x)
//            let height = min(Int(boundingBox.height), image.height - y)
//            
//            guard width > 0, height > 0 else {
//                let currentTimeError = dateFormatter.string(from: Date())
//                os_log(.error, log: log, "%{public}s: Invalid bounding box dimensions: width=%d, height=%d at %{public}s", TAG, width, height, currentTimeError)
//                return nil
//            }
//            
//            let cropRect = CGRect(x: CGFloat(x), y: CGFloat(y), width: CGFloat(width), height: CGFloat(height))
//            guard let cgImage = uiImage.cgImage else {
//                let currentTimeError = dateFormatter.string(from: Date())
//                os_log(.error, log: log, "%{public}s: Failed to get CGImage at %{public}s", TAG, currentTimeError)
//                return nil
//            }
//            
//            guard let croppedCGImage = cgImage.cropping(to: cropRect) else {
//                let currentTimeError = dateFormatter.string(from: Date())
//                os_log(.error, log: log, "%{public}s: Failed to crop face to rect: x=%d, y=%d, w=%d, h=%d at %{public}s", TAG, x, y, width, height, currentTimeError)
//                return nil
//            }
//            
//            let croppedImage = UIImage(cgImage: croppedCGImage, scale: 1.0, orientation: .up)
//            let currentTimeSuccess = dateFormatter.string(from: Date())
//            os_log(.debug, log: log, "%{public}s: Cropped face: %dx%d at %{public}s", TAG, Int(croppedImage.size.width), Int(croppedImage.size.height), currentTimeSuccess)
//            
//            let landmarks = extractLandmarks(face: face, offsetX: x, offsetY: y, width: width, height: height)
//            
//            return (croppedCGImage, landmarks)
//        } catch {
//            let currentTimeError = dateFormatter.string(from: Date())
//            os_log(.error, log: log, "%{public}s: Face detection error: %{public}s at %{public}s", TAG, error.localizedDescription, currentTimeError)
//            return nil
//        }
//    }
//    
//    private func extractLandmarks(face: Face, offsetX: Int, offsetY: Int, width: Int, height: Int) -> [[Float]] {
//        let leftEye = face.landmark(ofType: .leftEye)
//        let rightEye = face.landmark(ofType: .rightEye)
//        let noseBase = face.landmark(ofType: .noseBase)
//        let leftMouth = face.landmark(ofType: .mouthLeft)
//        let rightMouth = face.landmark(ofType: .mouthRight)
//        
//        // Helper function to compute coordinates with fallback
//        func getCoordinate(landmark: FaceLandmark?, xFactor: Float, yFactor: Float, offsetX: Int, offsetY: Int, width: Int, height: Int) -> [Float] {
//            let x = landmark.map { Float($0.position.x) } ?? (Float(offsetX) + Float(width) * xFactor)
//            let y = landmark.map { Float($0.position.y) } ?? (Float(offsetY) + Float(height) * yFactor)
//            return [x, y]
//        }
//        
//        // Construct the landmarks array step-by-step
//        let leftEyeCoords = getCoordinate(landmark: leftEye, xFactor: 0.3, yFactor: 0.3, offsetX: offsetX, offsetY: offsetY, width: width, height: height)
//        let rightEyeCoords = getCoordinate(landmark: rightEye, xFactor: 0.7, yFactor: 0.3, offsetX: offsetX, offsetY: offsetY, width: width, height: height)
//        let noseBaseCoords = getCoordinate(landmark: noseBase, xFactor: 0.5, yFactor: 0.5, offsetX: offsetX, offsetY: offsetY, width: width, height: height)
//        let leftMouthCoords = getCoordinate(landmark: leftMouth, xFactor: 0.4, yFactor: 0.7, offsetX: offsetX, offsetY: offsetY, width: width, height: height)
//        let rightMouthCoords = getCoordinate(landmark: rightMouth, xFactor: 0.6, yFactor: 0.7, offsetX: offsetX, offsetY: offsetY, width: width, height: height)
//        
//        return [
//            leftEyeCoords,
//            rightEyeCoords,
//            noseBaseCoords,
//            leftMouthCoords,
//            rightMouthCoords
//        ]
//    }
//}
