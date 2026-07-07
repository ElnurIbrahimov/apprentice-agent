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

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): ConversationDatabase =
        RoomConfig.builder(
            context,
            ConversationDatabase::class.java,
            "aura-conversations.db",
            migrations = arrayOf(MIGRATION_1_2),
        ).build()

    @Provides
    fun provideConversationDao(db: ConversationDatabase): ConversationDao = db.conversationDao()
}
