//
//  FaceEmbeddingEntity.swift
//  TestSDKDDD
//
//  Created by Acleda on 26/5/25.
//

import Foundation
import UIKit

@objc public class FaceEmbeddingEntity: NSObject {
    @objc public let id: Int64
    @objc public let userId: String
    @objc public let name: String
    @objc public let embedding: [Double]
    @objc public let image: NSData?
    @objc public let timestamp: Int64
    @objc public let matchCount: Int
    @objc public let lastMatch: String?
    
    public init(
        id: Int64,
        userId: String,
        name: String,
        embedding: [Double],
        image: NSData? = nil,
        timestamp: Int64,
        matchCount: Int = 0,
        lastMatch: String? = nil
    ) {
        self.id = id
        self.userId = userId
        self.name = name
        self.embedding = embedding
        self.image = image
        self.timestamp = timestamp
        self.matchCount = matchCount
        self.lastMatch = lastMatch
        super.init()
    }
}
