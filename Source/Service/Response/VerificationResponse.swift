struct VerificationResponse: Codable {
    let status: String
    let code: Int?
    let message: String?
    let details: VerificationDetails?
}

struct VerificationDetails: Codable {
    let similarity: Float?
    let spoofLabel: String?
    let occlusionLabel: String?
}
