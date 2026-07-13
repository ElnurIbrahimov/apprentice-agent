package com.aura.agent

import android.content.Context
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import com.aura.data.RoomConfig
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object ConversationModule {

    val MIGRATION_1_2 = object : Migration(1, 2) {
        override fun migrate(db: SupportSQLiteDatabase) {
            // Add embedding column for semantic conversation search.
            // Null by default — embeddings are lazily populated on
            // first semantic search, not on every save.
            db.execSQL("ALTER TABLE conversations ADD COLUMN embedding BLOB DEFAULT NULL")
        }
    }

    /**
     * Migration 2→3: adds an index on `updatedAt` so the History screen's
     * `ORDER BY updatedAt DESC` query is O(log n) instead of a full table
     * scan. As the conversation list grows past a few hundred rows this
     * turns from "noticeable" to "obviously slow" without the index.
     */
    val MIGRATION_2_3 = object : Migration(2, 3) {
        override fun migrate(db: SupportSQLiteDatabase) {
            db.execSQL("CREATE INDEX IF NOT EXISTS index_conversations_updatedAt ON conversations(updatedAt)")
        }
    }

    /** Durable rolling-summary state; full turns remain untouched. */
    val MIGRATION_3_4 = object : Migration(3, 4) {
        override fun migrate(db: SupportSQLiteDatabase) {
            db.execSQL("ALTER TABLE conversations ADD COLUMN contextSummary TEXT NOT NULL DEFAULT ''")
            db.execSQL("ALTER TABLE conversations ADD COLUMN summaryThroughTurn INTEGER NOT NULL DEFAULT 0")
        }
    }

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): ConversationDatabase =
        RoomConfig.builder(
            context,
            ConversationDatabase::class.java,
            "aura-conversations.db",
            migrations = arrayOf(MIGRATION_1_2, MIGRATION_2_3, MIGRATION_3_4),
        ).build()

    @Provides
    fun provideConversationDao(db: ConversationDatabase): ConversationDao = db.conversationDao()
}
