package com.aura.memory

import android.content.Context
import androidx.room.Room
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import com.aura.data.RoomConfig
import com.aura.creative.CreativeProjectDao
import com.aura.documents.DocumentDao
import com.aura.documents.DocumentChunkDao
import com.aura.documents.DocumentChunkEntity
import com.aura.providers.ProviderKeys
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import okhttp3.OkHttpClient
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object MemoryModule {

    val MIGRATION_1_2 = object : Migration(1, 2) {
        override fun migrate(db: SupportSQLiteDatabase) {
            db.execSQL("""
                CREATE TABLE IF NOT EXISTS kg_nodes (
                    id TEXT NOT NULL PRIMARY KEY,
                    label TEXT NOT NULL,
                    type TEXT NOT NULL,
                    properties TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    sourceTurnId TEXT NOT NULL,
                    createdAt INTEGER NOT NULL,
                    updatedAt INTEGER NOT NULL,
                    accessCount INTEGER NOT NULL,
                    lastAccessed INTEGER NOT NULL
                )
            """.trimIndent())
            db.execSQL("CREATE INDEX IF NOT EXISTS index_kg_nodes_label ON kg_nodes(label)")
            db.execSQL("CREATE INDEX IF NOT EXISTS index_kg_nodes_type ON kg_nodes(type)")
            db.execSQL("CREATE UNIQUE INDEX IF NOT EXISTS index_kg_nodes_label_type ON kg_nodes(label, type)")
            db.execSQL("""
                CREATE TABLE IF NOT EXISTS kg_edges (
                    id TEXT NOT NULL PRIMARY KEY,
                    type TEXT NOT NULL,
                    sourceId TEXT NOT NULL,
                    targetId TEXT NOT NULL,
                    weight REAL NOT NULL,
                    properties TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    sourceTurnId TEXT NOT NULL,
                    createdAt INTEGER NOT NULL,
                    lastReinforced INTEGER NOT NULL,
                    FOREIGN KEY(sourceId) REFERENCES kg_nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY(targetId) REFERENCES kg_nodes(id) ON DELETE CASCADE
                )
            """.trimIndent())
            db.execSQL("CREATE INDEX IF NOT EXISTS index_kg_edges_sourceId ON kg_edges(sourceId)")
            db.execSQL("CREATE INDEX IF NOT EXISTS index_kg_edges_targetId ON kg_edges(targetId)")
            db.execSQL("CREATE UNIQUE INDEX IF NOT EXISTS index_kg_edges_sourceId_targetId_type ON kg_edges(sourceId, targetId, type)")
        }
    }

    val MIGRATION_2_3 = object : Migration(2, 3) {
        override fun migrate(db: SupportSQLiteDatabase) {
            // memory_edits table: audit trail for memory edits.
            // ForeignKey CASCADE on memoryId so deleting a memory
            // also cleans up its edit history.
            db.execSQL("""
                CREATE TABLE IF NOT EXISTS memory_edits (
                    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                    memoryId TEXT NOT NULL,
                    oldContent TEXT NOT NULL,
                    newContent TEXT NOT NULL,
                    oldCategory TEXT NOT NULL,
                    newCategory TEXT NOT NULL,
                    editedAt INTEGER NOT NULL DEFAULT 0,
                    editedBy TEXT NOT NULL DEFAULT 'user',
                    FOREIGN KEY(memoryId) REFERENCES memories(id) ON DELETE CASCADE
                )
            """.trimIndent())
            db.execSQL("CREATE INDEX IF NOT EXISTS index_memory_edits_memoryId ON memory_edits(memoryId)")
        }
    }

    val MIGRATION_3_4 = object : Migration(3, 4) {
        override fun migrate(db: SupportSQLiteDatabase) {
            db.execSQL("ALTER TABLE memories ADD COLUMN sourceConversationId TEXT NOT NULL DEFAULT ''")
            db.execSQL("ALTER TABLE memories ADD COLUMN sourceTurnTimestamp INTEGER NOT NULL DEFAULT 0")
            db.execSQL("CREATE INDEX IF NOT EXISTS index_memories_sourceConversationId ON memories(sourceConversationId)")
            db.execSQL("ALTER TABLE kg_nodes ADD COLUMN sourceConversationId TEXT NOT NULL DEFAULT ''")
            db.execSQL("ALTER TABLE kg_nodes ADD COLUMN sourceTurnTimestamp INTEGER NOT NULL DEFAULT 0")
            db.execSQL("CREATE INDEX IF NOT EXISTS index_kg_nodes_sourceConversationId ON kg_nodes(sourceConversationId)")
            db.execSQL("ALTER TABLE kg_edges ADD COLUMN sourceConversationId TEXT NOT NULL DEFAULT ''")
            db.execSQL("ALTER TABLE kg_edges ADD COLUMN sourceTurnTimestamp INTEGER NOT NULL DEFAULT 0")
            db.execSQL("CREATE INDEX IF NOT EXISTS index_kg_edges_sourceConversationId ON kg_edges(sourceConversationId)")
        }
    }

    val MIGRATION_4_5 = object : Migration(4, 5) {
        override fun migrate(db: SupportSQLiteDatabase) {
            db.execSQL("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT NOT NULL PRIMARY KEY,
                    name TEXT NOT NULL,
                    mimeType TEXT NOT NULL,
                    sourceUri TEXT NOT NULL,
                    importedAt INTEGER NOT NULL,
                    characterCount INTEGER NOT NULL,
                    chunkCount INTEGER NOT NULL
                )
            """.trimIndent())
            db.execSQL("CREATE INDEX IF NOT EXISTS index_documents_name ON documents(name)")
            db.execSQL("CREATE INDEX IF NOT EXISTS index_documents_importedAt ON documents(importedAt)")
        }
    }

    val MIGRATION_5_6 = object : Migration(5, 6) {
        override fun migrate(db: SupportSQLiteDatabase) {
            db.execSQL("""
                CREATE TABLE IF NOT EXISTS creative_projects (
                    id TEXT NOT NULL PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    genre TEXT NOT NULL,
                    tone TEXT NOT NULL,
                    worldJson TEXT NOT NULL,
                    templateId TEXT NOT NULL,
                    metadataJson TEXT NOT NULL,
                    turnCount INTEGER NOT NULL,
                    lastSessionEnded INTEGER NOT NULL,
                    createdAt INTEGER NOT NULL,
                    updatedAt INTEGER NOT NULL
                )
            """.trimIndent())
            db.execSQL("CREATE INDEX IF NOT EXISTS index_creative_projects_updatedAt ON creative_projects(updatedAt)")
            db.execSQL("CREATE INDEX IF NOT EXISTS index_creative_projects_name ON creative_projects(name)")
        }
    }

    val MIGRATION_6_7 = object : Migration(6, 7) {
        override fun migrate(db: SupportSQLiteDatabase) {
            // First-class document chunks table with embedding metadata
            db.execSQL("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT NOT NULL PRIMARY KEY,
                    documentId TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    charStart INTEGER NOT NULL,
                    charEnd INTEGER NOT NULL,
                    pageNumber INTEGER NOT NULL DEFAULT 0,
                    text TEXT NOT NULL,
                    contentHash TEXT NOT NULL,
                    embedding BLOB,
                    embeddingModel TEXT,
                    embeddingVersion INTEGER NOT NULL DEFAULT 0,
                    embeddedAt INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(documentId) REFERENCES documents(id) ON DELETE CASCADE
                )
            """.trimIndent())
            db.execSQL("CREATE INDEX IF NOT EXISTS index_document_chunks_documentId ON document_chunks(documentId)")
            db.execSQL("CREATE INDEX IF NOT EXISTS index_document_chunks_documentId_ordinal ON document_chunks(documentId, ordinal)")
            db.execSQL("CREATE INDEX IF NOT EXISTS index_document_chunks_contentHash ON document_chunks(contentHash)")

            // Add embedding model/version metadata to memories
            db.execSQL("ALTER TABLE memories ADD COLUMN embeddingModel TEXT")
            db.execSQL("ALTER TABLE memories ADD COLUMN embeddingVersion INTEGER NOT NULL DEFAULT 0")

            // Add indexing status to documents
            db.execSQL("ALTER TABLE documents ADD COLUMN indexStatus TEXT NOT NULL DEFAULT 'pending'")
            db.execSQL("ALTER TABLE documents ADD COLUMN indexError TEXT NOT NULL DEFAULT ''")
        }
    }

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): MemoryDatabase =
        RoomConfig.builder(
            context,
            MemoryDatabase::class.java,
            "aura-memory.db",
            migrations = arrayOf(MIGRATION_1_2, MIGRATION_2_3, MIGRATION_3_4, MIGRATION_4_5, MIGRATION_5_6, MIGRATION_6_7),
        ).build()

    @Provides
    fun provideMemoryDao(db: MemoryDatabase): MemoryDao = db.memoryDao()

    @Provides
    fun provideMemoryEditDao(db: MemoryDatabase): MemoryEditDao = db.memoryEditDao()

    @Provides
    fun provideDocumentDao(db: MemoryDatabase): DocumentDao = db.documentDao()

    @Provides
    fun provideCreativeProjectDao(db: MemoryDatabase): CreativeProjectDao = db.creativeProjectDao()

    @Provides
    fun provideDocumentChunkDao(db: MemoryDatabase): DocumentChunkDao = db.documentChunkDao()

    @Provides
    @Singleton
    fun provideLocalEmbedder(): LocalEmbedder = LocalEmbedder()

    @Provides
    @Singleton
    fun provideEmbedder(
        localEmbedder: LocalEmbedder,
        providerKeys: ProviderKeys,
        httpClient: OkHttpClient,
    ): Embedder = CloudEmbedder(localEmbedder, providerKeys, httpClient)

    @Provides
    @Singleton
    fun provideVectorIndex(): VectorIndex = VectorIndex()

    @Provides
    @Singleton
    fun provideWriteGate(): WriteGate = WriteGate()
}
