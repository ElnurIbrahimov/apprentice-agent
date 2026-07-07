package com.aura.memory

import android.content.Context
import androidx.room.Room
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import com.aura.data.RoomConfig
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
                    confidence REAL NOT NULL DEFAULT 1.0,
                    properties TEXT DEFAULT '{}',
                    sourceTurnId TEXT DEFAULT '',
                    accessCount INTEGER NOT NULL DEFAULT 0,
                    lastAccessed INTEGER NOT NULL DEFAULT 0,
                    updatedAt INTEGER NOT NULL DEFAULT 0
                )
            """.trimIndent())
            db.execSQL("""
                CREATE TABLE IF NOT EXISTS kg_edges (
                    id TEXT NOT NULL PRIMARY KEY,
                    type TEXT NOT NULL,
                    sourceId TEXT NOT NULL,
                    targetId TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    sourceTurnId TEXT DEFAULT '',
                    lastReinforced INTEGER NOT NULL DEFAULT 0
                )
            """.trimIndent())
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

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): MemoryDatabase =
        RoomConfig.builder(
            context,
            MemoryDatabase::class.java,
            "aura-memory.db",
            migrations = arrayOf(MIGRATION_1_2, MIGRATION_2_3),
        ).build()

    @Provides
    fun provideMemoryDao(db: MemoryDatabase): MemoryDao = db.memoryDao()

    @Provides
    fun provideMemoryEditDao(db: MemoryDatabase): MemoryEditDao = db.memoryEditDao()

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
