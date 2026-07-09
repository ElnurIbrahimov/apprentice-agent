package com.aura.agent

import androidx.room.testing.MigrationTestHelper
import androidx.sqlite.db.framework.FrameworkSQLiteOpenHelperFactory
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Rule
import org.junit.Test

class ConversationDatabaseMigrationTest {

    @get:Rule
    val helper = MigrationTestHelper(
        InstrumentationRegistry.getInstrumentation(),
        ConversationDatabase::class.java.canonicalName,
        FrameworkSQLiteOpenHelperFactory(),
    )

    @Test
    fun migrate1To2_preservesConversations_andAddsEmbedding() {
        val db = helper.createDatabase("test-conversations.db", 1)
        // Insert a conversation using the v1 schema (no embedding column).
        db.execSQL(
            """
            INSERT INTO conversations (id, title, createdAt, updatedAt, systemPrompt, model, metadataJson, turnsJson)
            VALUES ('conv-1', 'Test chat', 1700000000, 1700000100, 'You are Aura', 'ollama:glm-5.1:cloud', '{}', '[]')
            """.trimIndent(),
        )
        db.close()

        val migrated = helper.runMigrationsAndValidate(
            "test-conversations.db",
            2,
            true,
            ConversationModule.MIGRATION_1_2,
        )

        // Verify the conversation survived.
        val cursor = migrated.query("SELECT * FROM conversations WHERE id = 'conv-1'")
        cursor.use {
            assert(it.moveToFirst()) { "Conversation row should survive migration" }
            assert(it.getString(it.getColumnIndexOrThrow("title")) == "Test chat")
            // The embedding column should exist and be NULL (lazily populated).
            val embeddingIdx = it.getColumnIndexOrThrow("embedding")
            assert(it.isNull(embeddingIdx)) { "embedding should be NULL after migration" }
        }

        migrated.close()
    }
}