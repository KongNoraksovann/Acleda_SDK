import Foundation
import SQLite3
import os.log
import UIKit

@available(iOS 13.0, *)
class FaceDatabase {
    private let SQLITE_TRANSIENT = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
    private var db: OpaquePointer?
    private let log = OSLog(subsystem: "com.example.faceverification.data.db", category: "FaceDatabase")
    private let TAG = "FaceDatabase"
    
    private static let DATABASE_VERSION = 4
    private static let DATABASE_NAME = "face_embeddings.db"
    private static let TABLE_FACES = "faces"
    private static let KEY_ID = "id"
    private static let KEY_USER_ID = "user_id"
    private static let KEY_NAME = "name"
    private static let KEY_EMBED = "embed"
    private static let KEY_IMAGE = "image"
    private static let KEY_MATCH_COUNT = "match_count"
    private static let KEY_LAST_MATCH = "last_match"
    private static let KEY_TIMESTAMP = "timestamp"
    
    enum SQLiteError: Error {
        case openFailed(message: String)
        case prepareFailed(message: String)
        case stepFailed(message: String)
        case bindFailed(message: String)
        case execFailed(message: String)
    }
    
    struct FaceEmbedding {
        let id: Int64
        let userId: String
        let name: String
        let embedding: [Double]
        let image: Data?
        let timestamp: Int64
    }
    
    init(context: Any? = nil) throws {
        let fileManager = FileManager.default
        guard let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first else {
            throw SQLiteError.openFailed(message: "Cannot access documents directory")
        }
        let dbPath = documentsURL.appendingPathComponent(Self.DATABASE_NAME).path
        
        let sqliteVersion = String(cString: sqlite3_libversion())
        os_log(.debug, log: log, "%{public}s: SQLite version: %{public}s", TAG, sqliteVersion)
        
        if fileManager.fileExists(atPath: dbPath) {
            os_log(.debug, log: log, "%{public}s: Database file exists at %{public}s", TAG, dbPath)
        } else {
            os_log(.debug, log: log, "%{public}s: Database file does not exist at %{public}s; will be created", TAG, dbPath)
        }
        
        guard sqlite3_open(dbPath, &db) == SQLITE_OK else {
            let errorMsg = String(cString: sqlite3_errmsg(db))
            sqlite3_close(db)
            db = nil
            throw SQLiteError.openFailed(message: errorMsg)
        }
        
        let createTableSQL = """
        CREATE TABLE IF NOT EXISTS \(Self.TABLE_FACES) (
            \(Self.KEY_ID) INTEGER PRIMARY KEY AUTOINCREMENT,
            \(Self.KEY_USER_ID) TEXT NOT NULL UNIQUE,
            \(Self.KEY_NAME) TEXT,
            \(Self.KEY_EMBED) TEXT,
            \(Self.KEY_IMAGE) BLOB,
            \(Self.KEY_MATCH_COUNT) INTEGER DEFAULT 0,
            \(Self.KEY_LAST_MATCH) TEXT,
            \(Self.KEY_TIMESTAMP) INTEGER NOT NULL
        );
        """
        
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, createTableSQL, -1, &statement, nil) == SQLITE_OK else {
            let errorMsg = String(cString: sqlite3_errmsg(db))
            sqlite3_finalize(statement)
            throw SQLiteError.prepareFailed(message: errorMsg)
        }
        
        guard sqlite3_step(statement) == SQLITE_DONE else {
            let errorMsg = String(cString: sqlite3_errmsg(db))
            sqlite3_finalize(statement)
            throw SQLiteError.stepFailed(message: errorMsg)
        }
        
        sqlite3_finalize(statement)
        os_log(.debug, log: log, "%{public}s: Database table created", TAG)
        
        try upgradeIfNeeded()
    }
    
    private func upgradeIfNeeded() throws {
        let userVersion = getUserVersion()
        if userVersion < Self.DATABASE_VERSION {
            var statement: OpaquePointer?
            
            // Rename existing table
            let renameSQL = "ALTER TABLE \(Self.TABLE_FACES) RENAME TO \(Self.TABLE_FACES)_old;"
            guard sqlite3_prepare_v2(db, renameSQL, -1, &statement, nil) == SQLITE_OK else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                sqlite3_finalize(statement)
                throw SQLiteError.prepareFailed(message: errorMsg)
            }
            guard sqlite3_step(statement) == SQLITE_DONE else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                sqlite3_finalize(statement)
                throw SQLiteError.stepFailed(message: errorMsg)
            }
            sqlite3_finalize(statement)
            
            // Create new table
            let createTableSQL = """
            CREATE TABLE \(Self.TABLE_FACES) (
                \(Self.KEY_ID) INTEGER PRIMARY KEY AUTOINCREMENT,
                \(Self.KEY_USER_ID) TEXT NOT NULL UNIQUE,
                \(Self.KEY_NAME) TEXT,
                \(Self.KEY_EMBED) TEXT,
                \(Self.KEY_IMAGE) BLOB,
                \(Self.KEY_MATCH_COUNT) INTEGER DEFAULT 0,
                \(Self.KEY_LAST_MATCH) TEXT,
                \(Self.KEY_TIMESTAMP) INTEGER NOT NULL
            );
            """
            guard sqlite3_prepare_v2(db, createTableSQL, -1, &statement, nil) == SQLITE_OK else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                sqlite3_finalize(statement)
                throw SQLiteError.prepareFailed(message: errorMsg)
            }
            guard sqlite3_step(statement) == SQLITE_DONE else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                sqlite3_finalize(statement)
                throw SQLiteError.stepFailed(message: errorMsg)
            }
            sqlite3_finalize(statement)
            
            // Migrate data
            let migrateSQL = """
            INSERT INTO \(Self.TABLE_FACES) (
                \(Self.KEY_ID), \(Self.KEY_USER_ID), \(Self.KEY_NAME), \(Self.KEY_EMBED),
                \(Self.KEY_IMAGE), \(Self.KEY_MATCH_COUNT), \(Self.KEY_LAST_MATCH), \(Self.KEY_TIMESTAMP)
            )
            SELECT
                \(Self.KEY_ID), \(Self.KEY_USER_ID), \(Self.KEY_NAME), \(Self.KEY_EMBED),
                \(Self.KEY_IMAGE), \(Self.KEY_MATCH_COUNT), \(Self.KEY_LAST_MATCH), \(Self.KEY_TIMESTAMP)
            FROM \(Self.TABLE_FACES)_old;
            """
            guard sqlite3_prepare_v2(db, migrateSQL, -1, &statement, nil) == SQLITE_OK else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                sqlite3_finalize(statement)
                throw SQLiteError.prepareFailed(message: errorMsg)
            }
            guard sqlite3_step(statement) == SQLITE_DONE else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                sqlite3_finalize(statement)
                throw SQLiteError.stepFailed(message: errorMsg)
            }
            sqlite3_finalize(statement)
            
            // Drop old table
            let dropSQL = "DROP TABLE \(Self.TABLE_FACES)_old;"
            guard sqlite3_prepare_v2(db, dropSQL, -1, &statement, nil) == SQLITE_OK else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                sqlite3_finalize(statement)
                throw SQLiteError.prepareFailed(message: errorMsg)
            }
            guard sqlite3_step(statement) == SQLITE_DONE else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                sqlite3_finalize(statement)
                throw SQLiteError.stepFailed(message: errorMsg)
            }
            sqlite3_finalize(statement)
            
            // Update version
            let versionSQL = "PRAGMA user_version = \(Self.DATABASE_VERSION);"
            guard sqlite3_exec(db, versionSQL, nil, nil, nil) == SQLITE_OK else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                throw SQLiteError.execFailed(message: errorMsg)
            }
            
            os_log(.debug, log: log, "%{public}s: Database upgraded from version %d to %d", TAG, userVersion, Self.DATABASE_VERSION)
        }
    }
    
    private func getUserVersion() -> Int {
        var statement: OpaquePointer?
        let versionSQL = "PRAGMA user_version;"
        guard sqlite3_prepare_v2(db, versionSQL, -1, &statement, nil) == SQLITE_OK else {
            sqlite3_finalize(statement)
            return 0
        }
        
        defer { sqlite3_finalize(statement) }
        
        guard sqlite3_step(statement) == SQLITE_ROW else {
            return 0
        }
        
        let version = sqlite3_column_int(statement, 0)
        return Int(version)
    }
    
    // Matching Kotlin storeFaceImage exactly
    func storeFaceImage(userId: String, faceBitmap: UIImage?) async throws -> Bool {
        do {
            let timestamp = Int64(Date().timeIntervalSince1970 * 1000)
            var imageData: Data? = nil
            if let image = faceBitmap {
                imageData = image.pngData()
            }
            
            // Query for existing record first (matching Kotlin)
            let selectSQL = "SELECT \(Self.KEY_ID) FROM \(Self.TABLE_FACES) WHERE \(Self.KEY_USER_ID) = ?;"
            var selectStatement: OpaquePointer?
            guard sqlite3_prepare_v2(db, selectSQL, -1, &selectStatement, nil) == SQLITE_OK else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                sqlite3_finalize(selectStatement)
                throw SQLiteError.prepareFailed(message: errorMsg)
            }
            
            defer { sqlite3_finalize(selectStatement) }
            
            guard let userIdCString = (userId as NSString).utf8String,
                  sqlite3_bind_text(selectStatement, 1, userIdCString, -1, SQLITE_TRANSIENT) == SQLITE_OK else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                throw SQLiteError.bindFailed(message: errorMsg)
            }
            
            let resultId: Int64
            
            if sqlite3_step(selectStatement) == SQLITE_ROW {
                // Record exists - update by ID (matching Kotlin exactly)
                let id = sqlite3_column_int64(selectStatement, 0)
                
                let updateSQL = """
                UPDATE \(Self.TABLE_FACES) 
                SET \(Self.KEY_IMAGE) = ?, \(Self.KEY_TIMESTAMP) = ?
                WHERE \(Self.KEY_ID) = ?;
                """
                
                var updateStatement: OpaquePointer?
                guard sqlite3_prepare_v2(db, updateSQL, -1, &updateStatement, nil) == SQLITE_OK else {
                    let errorMsg = String(cString: sqlite3_errmsg(db))
                    sqlite3_finalize(updateStatement)
                    throw SQLiteError.prepareFailed(message: errorMsg)
                }
                
                defer { sqlite3_finalize(updateStatement) }
                
                if let imageData = imageData {
                    guard sqlite3_bind_blob(updateStatement, 1, (imageData as NSData).bytes, Int32(imageData.count), SQLITE_TRANSIENT) == SQLITE_OK else {
                        let errorMsg = String(cString: sqlite3_errmsg(db))
                        throw SQLiteError.bindFailed(message: errorMsg)
                    }
                } else {
                    guard sqlite3_bind_null(updateStatement, 1) == SQLITE_OK else {
                        let errorMsg = String(cString: sqlite3_errmsg(db))
                        throw SQLiteError.bindFailed(message: errorMsg)
                    }
                }
                
                guard sqlite3_bind_int64(updateStatement, 2, timestamp) == SQLITE_OK else {
                    let errorMsg = String(cString: sqlite3_errmsg(db))
                    throw SQLiteError.bindFailed(message: errorMsg)
                }
                
                guard sqlite3_bind_int64(updateStatement, 3, id) == SQLITE_OK else {
                    let errorMsg = String(cString: sqlite3_errmsg(db))
                    throw SQLiteError.bindFailed(message: errorMsg)
                }
                
                guard sqlite3_step(updateStatement) == SQLITE_DONE else {
                    let errorMsg = String(cString: sqlite3_errmsg(db))
                    throw SQLiteError.stepFailed(message: errorMsg)
                }
                
                resultId = id
            } else {
                // No record exists - insert new (matching Kotlin exactly)
                let insertSQL = """
                INSERT INTO \(Self.TABLE_FACES) (
                    \(Self.KEY_USER_ID), \(Self.KEY_IMAGE), \(Self.KEY_TIMESTAMP)
                ) VALUES (?, ?, ?);
                """
                
                var insertStatement: OpaquePointer?
                guard sqlite3_prepare_v2(db, insertSQL, -1, &insertStatement, nil) == SQLITE_OK else {
                    let errorMsg = String(cString: sqlite3_errmsg(db))
                    sqlite3_finalize(insertStatement)
                    throw SQLiteError.prepareFailed(message: errorMsg)
                }
                
                defer { sqlite3_finalize(insertStatement) }
                
                guard sqlite3_bind_text(insertStatement, 1, userIdCString, -1, SQLITE_TRANSIENT) == SQLITE_OK else {
                    let errorMsg = String(cString: sqlite3_errmsg(db))
                    throw SQLiteError.bindFailed(message: errorMsg)
                }
                
                if let imageData = imageData {
                    guard sqlite3_bind_blob(insertStatement, 2, (imageData as NSData).bytes, Int32(imageData.count), SQLITE_TRANSIENT) == SQLITE_OK else {
                        let errorMsg = String(cString: sqlite3_errmsg(db))
                        throw SQLiteError.bindFailed(message: errorMsg)
                    }
                } else {
                    guard sqlite3_bind_null(insertStatement, 2) == SQLITE_OK else {
                        let errorMsg = String(cString: sqlite3_errmsg(db))
                        throw SQLiteError.bindFailed(message: errorMsg)
                    }
                }
                
                guard sqlite3_bind_int64(insertStatement, 3, timestamp) == SQLITE_OK else {
                    let errorMsg = String(cString: sqlite3_errmsg(db))
                    throw SQLiteError.bindFailed(message: errorMsg)
                }
                
                guard sqlite3_step(insertStatement) == SQLITE_DONE else {
                    let errorMsg = String(cString: sqlite3_errmsg(db))
                    throw SQLiteError.stepFailed(message: errorMsg)
                }
                
                resultId = sqlite3_last_insert_rowid(db)
            }
            
            let success = resultId != -1
            if success {
                os_log(.debug, log: log, "%{public}s: Face image stored for user %{public}s", TAG, userId)
            } else {
                os_log(.error, log: log, "%{public}s: Failed to store face image for user %{public}s", TAG, userId)
            }
            
            return success
        } catch {
            os_log(.error, log: log, "%{public}s: Error saving face image: %{public}s", TAG, error.localizedDescription)
            return false
        }
    }
    
    // Matching Kotlin storeFaceEmbedding exactly
    func storeFaceEmbedding(userId: String, name: String?, embedding: [Double], faceBitmap: UIImage?) async throws -> Bool {
        do {
            let embeddingData = try JSONEncoder().encode(embedding)
            guard let embeddingJson = String(data: embeddingData, encoding: .utf8) else {
                throw SQLiteError.bindFailed(message: "Failed to encode embedding")
            }
            
            let timestamp = Int64(Date().timeIntervalSince1970 * 1000)
            var imageData: Data? = nil
            if let image = faceBitmap {
                imageData = image.pngData()
            }
            
            // Try update first (matching Kotlin exactly)
            let updateSQL = """
            UPDATE \(Self.TABLE_FACES) 
            SET \(Self.KEY_NAME) = ?, \(Self.KEY_EMBED) = ?, \(Self.KEY_IMAGE) = ?, \(Self.KEY_TIMESTAMP) = ?
            WHERE \(Self.KEY_USER_ID) = ?;
            """
            
            var updateStatement: OpaquePointer?
            guard sqlite3_prepare_v2(db, updateSQL, -1, &updateStatement, nil) == SQLITE_OK else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                sqlite3_finalize(updateStatement)
                throw SQLiteError.prepareFailed(message: errorMsg)
            }
            
            defer { sqlite3_finalize(updateStatement) }
            
            // Bind parameters for update
            if let name = name, let nameCString = (name as NSString).utf8String {
                guard sqlite3_bind_text(updateStatement, 1, nameCString, -1, SQLITE_TRANSIENT) == SQLITE_OK else {
                    let errorMsg = String(cString: sqlite3_errmsg(db))
                    throw SQLiteError.bindFailed(message: errorMsg)
                }
            } else {
                guard sqlite3_bind_null(updateStatement, 1) == SQLITE_OK else {
                    let errorMsg = String(cString: sqlite3_errmsg(db))
                    throw SQLiteError.bindFailed(message: errorMsg)
                }
            }
            
            guard let jsonCString = (embeddingJson as NSString).utf8String,
                  sqlite3_bind_text(updateStatement, 2, jsonCString, -1, SQLITE_TRANSIENT) == SQLITE_OK else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                throw SQLiteError.bindFailed(message: errorMsg)
            }
            
            if let imageData = imageData {
                guard sqlite3_bind_blob(updateStatement, 3, (imageData as NSData).bytes, Int32(imageData.count), SQLITE_TRANSIENT) == SQLITE_OK else {
                    let errorMsg = String(cString: sqlite3_errmsg(db))
                    throw SQLiteError.bindFailed(message: errorMsg)
                }
            } else {
                guard sqlite3_bind_null(updateStatement, 3) == SQLITE_OK else {
                    let errorMsg = String(cString: sqlite3_errmsg(db))
                    throw SQLiteError.bindFailed(message: errorMsg)
                }
            }
            
            guard sqlite3_bind_int64(updateStatement, 4, timestamp) == SQLITE_OK else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                throw SQLiteError.bindFailed(message: errorMsg)
            }
            
            guard let userIdCString = (userId as NSString).utf8String,
                  sqlite3_bind_text(updateStatement, 5, userIdCString, -1, SQLITE_TRANSIENT) == SQLITE_OK else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                throw SQLiteError.bindFailed(message: errorMsg)
            }
            
            guard sqlite3_step(updateStatement) == SQLITE_DONE else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                throw SQLiteError.stepFailed(message: errorMsg)
            }
            
            let rowsUpdated = sqlite3_changes(db)
            
            let resultId: Int64
            
            if rowsUpdated > 0 {
                // Updated existing record - get the ID
                let selectSQL = "SELECT \(Self.KEY_ID) FROM \(Self.TABLE_FACES) WHERE \(Self.KEY_USER_ID) = ?;"
                var selectStatement: OpaquePointer?
                guard sqlite3_prepare_v2(db, selectSQL, -1, &selectStatement, nil) == SQLITE_OK else {
                    let errorMsg = String(cString: sqlite3_errmsg(db))
                    sqlite3_finalize(selectStatement)
                    throw SQLiteError.prepareFailed(message: errorMsg)
                }
                
                defer { sqlite3_finalize(selectStatement) }
                
                guard sqlite3_bind_text(selectStatement, 1, userIdCString, -1, SQLITE_TRANSIENT) == SQLITE_OK else {
                    let errorMsg = String(cString: sqlite3_errmsg(db))
                    throw SQLiteError.bindFailed(message: errorMsg)
                }
                
                guard sqlite3_step(selectStatement) == SQLITE_ROW else {
                    throw SQLiteError.stepFailed(message: "Failed to get updated record ID")
                }
                
                resultId = sqlite3_column_int64(selectStatement, 0)
                os_log(.debug, log: log, "%{public}s: Embedding updated for user %{public}s", TAG, userId)
            } else {
                // Insert new record (matching Kotlin logic)
                let insertSQL = """
                INSERT INTO \(Self.TABLE_FACES) (
                    \(Self.KEY_USER_ID), \(Self.KEY_NAME), \(Self.KEY_EMBED), \(Self.KEY_IMAGE), \(Self.KEY_TIMESTAMP)
                ) VALUES (?, ?, ?, ?, ?);
                """
                
                var insertStatement: OpaquePointer?
                guard sqlite3_prepare_v2(db, insertSQL, -1, &insertStatement, nil) == SQLITE_OK else {
                    let errorMsg = String(cString: sqlite3_errmsg(db))
                    sqlite3_finalize(insertStatement)
                    throw SQLiteError.prepareFailed(message: errorMsg)
                }
                
                defer { sqlite3_finalize(insertStatement) }
                
                // Bind parameters for insert
                guard sqlite3_bind_text(insertStatement, 1, userIdCString, -1, SQLITE_TRANSIENT) == SQLITE_OK else {
                    let errorMsg = String(cString: sqlite3_errmsg(db))
                    throw SQLiteError.bindFailed(message: errorMsg)
                }
                
                if let name = name, let nameCString = (name as NSString).utf8String {
                    guard sqlite3_bind_text(insertStatement, 2, nameCString, -1, SQLITE_TRANSIENT) == SQLITE_OK else {
                        let errorMsg = String(cString: sqlite3_errmsg(db))
                        throw SQLiteError.bindFailed(message: errorMsg)
                    }
                } else {
                    guard sqlite3_bind_null(insertStatement, 2) == SQLITE_OK else {
                        let errorMsg = String(cString: sqlite3_errmsg(db))
                        throw SQLiteError.bindFailed(message: errorMsg)
                    }
                }
                
                guard sqlite3_bind_text(insertStatement, 3, jsonCString, -1, SQLITE_TRANSIENT) == SQLITE_OK else {
                    let errorMsg = String(cString: sqlite3_errmsg(db))
                    throw SQLiteError.bindFailed(message: errorMsg)
                }
                
                if let imageData = imageData {
                    guard sqlite3_bind_blob(insertStatement, 4, (imageData as NSData).bytes, Int32(imageData.count), SQLITE_TRANSIENT) == SQLITE_OK else {
                        let errorMsg = String(cString: sqlite3_errmsg(db))
                        throw SQLiteError.bindFailed(message: errorMsg)
                    }
                } else {
                    guard sqlite3_bind_null(insertStatement, 4) == SQLITE_OK else {
                        let errorMsg = String(cString: sqlite3_errmsg(db))
                        throw SQLiteError.bindFailed(message: errorMsg)
                    }
                }
                
                guard sqlite3_bind_int64(insertStatement, 5, timestamp) == SQLITE_OK else {
                    let errorMsg = String(cString: sqlite3_errmsg(db))
                    throw SQLiteError.bindFailed(message: errorMsg)
                }
                
                guard sqlite3_step(insertStatement) == SQLITE_DONE else {
                    let errorMsg = String(cString: sqlite3_errmsg(db))
                    throw SQLiteError.stepFailed(message: errorMsg)
                }
                
                resultId = sqlite3_last_insert_rowid(db)
                os_log(.debug, log: log, "%{public}s: Embedding stored for user %{public}s", TAG, userId)
            }
            
            let success = resultId != -1
            if !success {
                os_log(.error, log: log, "%{public}s: Failed to store embedding for user %{public}s", TAG, userId)
            }
            
            return success
        } catch {
            os_log(.error, log: log, "%{public}s: Error saving embedding: %{public}s", TAG, error.localizedDescription)
            return false
        }
    }
    
    // Matching Kotlin updateMatchStatistics exactly
    func updateMatchStatistics(userId: String) async throws -> Bool {
        do {
            let updateSQL = """
            UPDATE \(Self.TABLE_FACES)
            SET \(Self.KEY_MATCH_COUNT) = \(Self.KEY_MATCH_COUNT) + 1,
                \(Self.KEY_LAST_MATCH) = datetime('now')
            WHERE \(Self.KEY_USER_ID) = ?;
            """
            
            guard let userIdCString = (userId as NSString).utf8String else {
                throw SQLiteError.bindFailed(message: "Failed to convert userId to C string")
            }
            
            guard sqlite3_exec(db, "BEGIN TRANSACTION;", nil, nil, nil) == SQLITE_OK else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                throw SQLiteError.execFailed(message: errorMsg)
            }
            
            var statement: OpaquePointer?
            guard sqlite3_prepare_v2(db, updateSQL, -1, &statement, nil) == SQLITE_OK else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                sqlite3_finalize(statement)
                sqlite3_exec(db, "ROLLBACK;", nil, nil, nil)
                throw SQLiteError.prepareFailed(message: errorMsg)
            }
            
            defer { sqlite3_finalize(statement) }
            
            guard sqlite3_bind_text(statement, 1, userIdCString, -1, SQLITE_TRANSIENT) == SQLITE_OK else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                sqlite3_exec(db, "ROLLBACK;", nil, nil, nil)
                throw SQLiteError.bindFailed(message: errorMsg)
            }
            
            guard sqlite3_step(statement) == SQLITE_DONE else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                sqlite3_exec(db, "ROLLBACK;", nil, nil, nil)
                throw SQLiteError.stepFailed(message: errorMsg)
            }
            
            guard sqlite3_exec(db, "COMMIT;", nil, nil, nil) == SQLITE_OK else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                sqlite3_exec(db, "ROLLBACK;", nil, nil, nil)
                throw SQLiteError.execFailed(message: errorMsg)
            }
            
            return true
        } catch {
            os_log(.error, log: log, "%{public}s: Error updating match statistics: %{public}s", TAG, error.localizedDescription)
            return false
        }
    }
    
    // Matching Kotlin getFaceEmbeddingByUserId exactly
    func getFaceEmbeddingByUserId(userId: String) async throws -> FaceEmbedding? {
        do {
            let selectSQL = """
            SELECT \(Self.KEY_ID), \(Self.KEY_USER_ID), \(Self.KEY_NAME), \(Self.KEY_EMBED), \(Self.KEY_IMAGE), \(Self.KEY_TIMESTAMP)
            FROM \(Self.TABLE_FACES)
            WHERE \(Self.KEY_USER_ID) = ?;
            """
            
            var statement: OpaquePointer?
            guard sqlite3_prepare_v2(db, selectSQL, -1, &statement, nil) == SQLITE_OK else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                sqlite3_finalize(statement)
                throw SQLiteError.prepareFailed(message: errorMsg)
            }
            
            defer { sqlite3_finalize(statement) }
            
            guard let userIdCString = (userId as NSString).utf8String,
                  sqlite3_bind_text(statement, 1, userIdCString, -1, SQLITE_TRANSIENT) == SQLITE_OK else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                throw SQLiteError.bindFailed(message: errorMsg)
            }
            
            guard sqlite3_step(statement) == SQLITE_ROW else {
                os_log(.debug, log: log, "%{public}s: No embedding found for user %{public}s", TAG, userId)
                return nil
            }
            
            let id = sqlite3_column_int64(statement, 0)
            guard let userIdPtr = sqlite3_column_text(statement, 1) else {
                throw SQLiteError.stepFailed(message: "Invalid user_id")
            }
            let retrievedUserId = String(cString: userIdPtr)
            
            // Handle name (can be null)
            let namePtr = sqlite3_column_text(statement, 2)
            let name = namePtr != nil ? String(cString: namePtr!) : nil
            
            // Handle embedding (can be null)
            let embedPtr = sqlite3_column_text(statement, 3)
            let embedding: [Double]
            if let embedPtr = embedPtr,
               let embedJson = String(cString: embedPtr).data(using: .utf8),
               !embedJson.isEmpty {
                embedding = try JSONDecoder().decode([Double].self, from: embedJson)
            } else {
                embedding = []
            }
            
            // Handle image (can be null)
            let imageBlob = sqlite3_column_blob(statement, 4)
            let imageLength = sqlite3_column_bytes(statement, 4)
            let imageData = imageBlob != nil ? Data(bytes: imageBlob!, count: Int(imageLength)) : nil
            
            let timestamp = sqlite3_column_int64(statement, 5)
            
            // Only create result if name is not null (matching Kotlin)
            guard let name = name else {
                return nil
            }
            
            let result = FaceEmbedding(
                id: id,
                userId: retrievedUserId,
                name: name,
                embedding: embedding,
                image: imageData,
                timestamp: timestamp
            )
            
            os_log(.debug, log: log, "%{public}s: Retrieved embedding for user %{public}s", TAG, retrievedUserId)
            return result
        } catch {
            os_log(.error, log: log, "%{public}s: Error retrieving embedding: %{public}s", TAG, error.localizedDescription)
            return nil
        }
    }
    
    // Matching Kotlin getAllFaceEmbeddings exactly
    func getAllFaceEmbeddings() async throws -> [String: [FaceEmbedding]] {
        var resultMap: [String: [FaceEmbedding]] = [:]
        
        do {
            let selectSQL = "SELECT * FROM \(Self.TABLE_FACES);"
            
            var statement: OpaquePointer?
            guard sqlite3_prepare_v2(db, selectSQL, -1, &statement, nil) == SQLITE_OK else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                sqlite3_finalize(statement)
                throw SQLiteError.prepareFailed(message: errorMsg)
            }
            
            defer { sqlite3_finalize(statement) }
            
            while sqlite3_step(statement) == SQLITE_ROW {
                let id = sqlite3_column_int64(statement, 0)
                guard let userIdPtr = sqlite3_column_text(statement, 1) else { continue }
                let userId = String(cString: userIdPtr)
                
                // Handle name (can be null) - matching Kotlin logic exactly
                let namePtr = sqlite3_column_text(statement, 2)
                let name = namePtr != nil ? String(cString: namePtr!) : nil
                
                // Handle embedding (can be null)
                let embedPtr = sqlite3_column_text(statement, 3)
                let embedding: [Double]
                if let embedPtr = embedPtr,
                   let embedJson = String(cString: embedPtr).data(using: .utf8),
                   !embedJson.isEmpty {
                    embedding = try JSONDecoder().decode([Double].self, from: embedJson)
                } else {
                    embedding = []
                }
                
                // Handle image (can be null)
                let imageBlob = sqlite3_column_blob(statement, 4)
                let imageLength = sqlite3_column_bytes(statement, 4)
                let imageData = imageBlob != nil ? Data(bytes: imageBlob!, count: Int(imageLength)) : nil
                
                let timestamp = sqlite3_column_int64(statement, 7)
                
                // Only create FaceEmbedding if name is not null (matching Kotlin exactly)
                if let name = name {
                    let faceEmbedding = FaceEmbedding(
                        id: id,
                        userId: userId,
                        name: name,
                        embedding: embedding,
                        image: imageData,
                        timestamp: timestamp
                    )
                    
                    // Add to result map
                    if resultMap[userId] == nil {
                        resultMap[userId] = []
                    }
                    resultMap[userId]?.append(faceEmbedding)
                    
                    os_log(.debug, log: log, "%{public}s: Retrieved embedding for user %{public}s", TAG, userId)
                }
            }
            
            os_log(.debug, log: log, "%{public}s: Retrieved all embeddings, count: %d", TAG, resultMap.count)
            return resultMap
        } catch {
            os_log(.error, log: log, "%{public}s: Error in getAllFaceEmbeddings: %{public}s", TAG, error.localizedDescription)
            return [:]
        }
    }
    
    // Matching Kotlin deleteFaceEmbedding exactly
    func deleteFaceEmbedding(userId: String) async throws -> Bool {
        do {
            let deleteSQL = "DELETE FROM \(Self.TABLE_FACES) WHERE \(Self.KEY_USER_ID) = ?;"
            
            var statement: OpaquePointer?
            guard sqlite3_prepare_v2(db, deleteSQL, -1, &statement, nil) == SQLITE_OK else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                sqlite3_finalize(statement)
                throw SQLiteError.prepareFailed(message: errorMsg)
            }
            
            defer { sqlite3_finalize(statement) }
            
            guard let userIdCString = (userId as NSString).utf8String,
                  sqlite3_bind_text(statement, 1, userIdCString, -1, SQLITE_TRANSIENT) == SQLITE_OK else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                throw SQLiteError.bindFailed(message: errorMsg)
            }
            
            guard sqlite3_step(statement) == SQLITE_DONE else {
                let errorMsg = String(cString: sqlite3_errmsg(db))
                throw SQLiteError.stepFailed(message: errorMsg)
            }
            
            let result = sqlite3_changes(db) > 0
            return result
        } catch {
            os_log(.error, log: log, "%{public}s: Error deleting embedding: %{public}s", TAG, error.localizedDescription)
            return false
        }
    }
    
    deinit {
        if let db = db {
            sqlite3_close(db)
            os_log(.debug, log: log, "%{public}s: Closed SQLite database", TAG)
        }
    }
}
