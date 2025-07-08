import Foundation
import CommonCrypto

/**
 * Utility functions for ML model handling, aligned with Kotlin ModelUtils.kt for loading and decrypting ONNX models
 */
@objc public class ModelUtils: NSObject {
    private static let TAG = "ModelUtils"

    /**
     * Loads model data from the bundle, decrypting if necessary
     *
     * @param bundle The bundle containing the model file
     * @param modelName The name of the model file (with extension, e.g., "edgeface_s_gamma_05.onnx.enc")
     * @return Data containing the decrypted or raw model bytes
     * @throws ModelLoadingException if loading or decryption fails
     */
    @objc public static func loadModelDataFromBundle(_ bundle: Bundle = Bundle.main, modelName: String) throws -> Data {
        LogUtils.d(TAG, "Attempting to load model: \(modelName)")
        
        guard let modelURL = bundle.url(forResource: modelName, withExtension: nil) else {
            let message = "Model file \(modelName) not found in bundle"
            LogUtils.e(TAG, message)
            throw ModelLoadingException(message)
        }
        
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            let message = "Model file \(modelName) does not exist at path: \(modelURL.path)"
            LogUtils.e(TAG, message)
            throw ModelLoadingException(message)
        }
        
        do {
            let data = try Data(contentsOf: modelURL)
            LogUtils.d(TAG, "Loaded model \(modelName), size: \(data.count) bytes")
            
            if modelName.hasSuffix(".enc") {
                LogUtils.d(TAG, "Model \(modelName) is encrypted, proceeding with decryption")
                let key = try getEncryptionKey(bundle: bundle)
                return try decryptModelBytes(data, key: key)
            } else {
                LogUtils.d(TAG, "Model \(modelName) is not encrypted")
                return data
            }
        } catch let error {
            let message = "Failed to load model bytes for \(modelName): \(error.localizedDescription)"
            LogUtils.e(TAG, message, error)
            throw ModelLoadingException(message, error)
        }
    }

    /**
     * Decrypt encrypted model bytes using AES in CBC mode with PKCS7 padding
     *
     * @param encryptedData The encrypted model data
     * @param key The AES key
     * @return Decrypted Data
     * @throws ModelLoadingException if decryption fails
     */
    private static func decryptModelBytes(_ encryptedData: Data, key: Data) throws -> Data {
        LogUtils.d(TAG, "Decrypting model data of size: \(encryptedData.count) bytes")
        
        guard encryptedData.count >= 16 else {
            let message = "Invalid encrypted data, too short for IV"
            LogUtils.e(TAG, message)
            throw ModelLoadingException(message)
        }
        
        let iv = encryptedData.prefix(16)
        let encryptedContent = encryptedData.dropFirst(16)
        LogUtils.d(TAG, "Extracted IV (16 bytes), encrypted content size: \(encryptedContent.count) bytes")

        let keyLength = key.count
        guard keyLength == kCCKeySizeAES256 else {
            let message = "Invalid key length, expected 32 bytes, got \(keyLength) bytes"
            LogUtils.e(TAG, message)
            throw ModelLoadingException(message)
        }

        let bufferSize = encryptedContent.count + kCCBlockSizeAES128
        var decrypted = [UInt8](repeating: 0, count: bufferSize)
        var numBytesDecrypted: size_t = 0

        let status = key.withUnsafeBytes { keyPtr in
            iv.withUnsafeBytes { ivPtr in
                encryptedContent.withUnsafeBytes { dataPtr in
                    CCCrypt(
                        CCOperation(kCCDecrypt),
                        CCAlgorithm(kCCAlgorithmAES),
                        CCOptions(kCCOptionPKCS7Padding),
                        keyPtr.baseAddress,
                        keyLength,
                        ivPtr.baseAddress,
                        dataPtr.baseAddress,
                        encryptedContent.count,
                        &decrypted,
                        bufferSize,
                        &numBytesDecrypted
                    )
                }
            }
        }

        guard status == kCCSuccess else {
            let message = "Decryption failed with CCCrypt error: \(status)"
            LogUtils.e(TAG, message)
            throw ModelLoadingException(message)
        }

        LogUtils.d(TAG, "Decryption successful, decrypted size: \(numBytesDecrypted) bytes")
        return Data(decrypted.prefix(numBytesDecrypted))
    }

    /**
     * Encrypt model bytes using AES in CBC mode with PKCS7 padding
     *
     * @param data The raw model data to encrypt
     * @param key The AES key
     * @return Encrypted Data with IV prepended
     * @throws ModelLoadingException if encryption fails
     */
    @objc public static func encryptModelBytes(_ data: Data, key: Data) throws -> Data {
        LogUtils.d(TAG, "Encrypting model data of size: \(data.count) bytes")
        
        guard key.count == kCCKeySizeAES256 else {
            let message = "Invalid key length, expected 32 bytes, got \(key.count) bytes"
            LogUtils.e(TAG, message)
            throw ModelLoadingException(message)
        }
        
        // Generate random IV
        var iv = [UInt8](repeating: 0, count: kCCBlockSizeAES128)
        let status = SecRandomCopyBytes(kSecRandomDefault, iv.count, &iv)
        guard status == errSecSuccess else {
            let message = "Failed to generate IV"
            LogUtils.e(TAG, message)
            throw ModelLoadingException(message)
        }
        let ivData = Data(iv)
        
        let bufferSize = data.count + kCCBlockSizeAES128
        var encrypted = [UInt8](repeating: 0, count: bufferSize)
        var numBytesEncrypted: size_t = 0
        
        let cryptStatus = key.withUnsafeBytes { keyPtr in
            ivData.withUnsafeBytes { ivPtr in
                data.withUnsafeBytes { dataPtr in
                    CCCrypt(
                        CCOperation(kCCEncrypt),
                        CCAlgorithm(kCCAlgorithmAES),
                        CCOptions(kCCOptionPKCS7Padding),
                        keyPtr.baseAddress,
                        key.count,
                        ivPtr.baseAddress,
                        dataPtr.baseAddress,
                        data.count,
                        &encrypted,
                        bufferSize,
                        &numBytesEncrypted
                    )
                }
            }
        }
        
        guard cryptStatus == kCCSuccess else {
            let message = "Encryption failed with CCCrypt error: \(cryptStatus)"
            LogUtils.e(TAG, message)
            throw ModelLoadingException(message)
        }
        
        LogUtils.d(TAG, "Encryption successful, encrypted size: \(numBytesEncrypted) bytes")
        return ivData + Data(encrypted.prefix(Int(numBytesEncrypted)))
    }

    /**
     * Load the AES encryption key from bundle resources
     *
     * @param bundle The bundle containing the key file
     * @return Data containing the AES key
     * @throws ModelLoadingException if key loading fails
     */
    private static func getEncryptionKey(bundle: Bundle) throws -> Data {
        // In a production app, store the key in secure storage (e.g., Keychain)
        // For this example, assume the key is stored in bundle as model_key.bin
        guard let keyURL = bundle.url(forResource: "model_key", withExtension: "bin") else {
            let message = "Encryption key file model_key.bin not found in bundle"
            LogUtils.e(TAG, message)
            throw ModelLoadingException(message)
        }
        
        do {
            let keyData = try Data(contentsOf: keyURL)
            guard keyData.count == kCCKeySizeAES256 else {
                let message = "Invalid key length, expected 32 bytes, got \(keyData.count) bytes"
                LogUtils.e(TAG, message)
                throw ModelLoadingException(message)
            }
            LogUtils.d(TAG, "Loaded encryption key, size: \(keyData.count) bytes")
            return keyData
        } catch let error {
            let message = "Failed to load encryption key: \(error.localizedDescription)"
            LogUtils.e(TAG, message, error)
            throw ModelLoadingException(message, error)
        }
    }
}

