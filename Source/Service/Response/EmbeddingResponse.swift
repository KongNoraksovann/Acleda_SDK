struct EmbeddingResponse: Codable {
    let status: String
    let code: Int?
    let message: String?
    let details: EmbeddingDetails?
}

struct EmbeddingDetails: Codable {
    let similarity: Float?
}
