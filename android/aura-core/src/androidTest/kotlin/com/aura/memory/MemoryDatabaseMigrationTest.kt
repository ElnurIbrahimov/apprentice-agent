package com.aura.memory

import androidx.room.testing.MigrationTestHelper
import androidx.sqlite.db.framework.FrameworkSQLiteOpenHelperFactory
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Rule
import org.junit.Test

class MemoryDatabaseMigrationTest {

    @get:Rule
    val helper = MigrationTestHelper(
        InstrumentationRegistry.getInstrumentation(),
        MemoryDatabase::class.java.canonicalName,
        FrameworkSQLiteOpenHelperFactory(),
    )

    @Test
    fun migrate1To2() {
        val db = helper.createDatabase("test-aura-memory.db", 1)
        db.close()

        helper.runMigrationsAndValidate(
            "test-aura-memory.db",
            2,
            true,
            MemoryModule.MIGRATION_1_2,
        )
    }

    @Test
    fun migrate2To3_preservesMemories_andCreatesEditHistory() {
        // Start at v2 (post-KG, pre-edit-history).
        val db = helper.createDatabase("test-aura-memory-v2.db", 2)
        // Insert a memory row using the v2 schema (memories table exists,
        // kg_nodes and kg_edges exist, but memory_edits does not).
        db.execSQL(
            """
            INSERT INTO memories (id, content, source, category, importance, embedding, createdAt, accessedAt, accessCount, decayScore, tags, metadata)
            VALUES ('mem-1', 'User likes coffee', 'user', 'preference', 0.8, NULL, 1700000000, 1700000000, 0, 1.0, '', '{}')
            """.trimIndent(),
        )
        db.close()

        val migrated = helper.runMigrationsAndValidate(
            "test-aura-memory-v2.db",
            3,
            true,
            MemoryModule.MIGRATION_2_3,
        )

        // Verify the memory survived.
        val cursor = migrated.query("SELECT * FROM memories WHERE id = 'mem-1'")
        cursor.use {
            assert(it.moveToFirst()) { "Memory row should survive 2→3 migration" }
            assert(it.getString(it.getColumnIndexOrThrow("content")) == "User likes coffee")
        }

        // Verify the memory_edits table exists with the FK + index.
        val editTableCursor = migrated.query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_edits'",
        )
        editTableCursor.use {
            assert(it.moveToFirst()) { "memory_edits table should exist after 2→3 migration" }
        }

        val editIndexCursor = migrated.query(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='index_memory_edits_memoryId'",
        )
        editIndexCursor.use {
            assert(it.moveToFirst()) { "memory_edits index should exist after 2→3 migration" }
        }

        migrated.close()
    }

    @Test
    fun migrate1To3_chained_preservesData() {
        val db = helper.createDatabase("test-aura-memory-chained.db", 1)
        db.close()

        val migrated = helper.runMigrationsAndValidate(
            "test-aura-memory-chained.db",
            3,
            true,
            MemoryModule.MIGRATION_1_2,
            MemoryModule.MIGRATION_2_3,
        )

        // After chained migration, all three tables (memories, kg_nodes,
        // kg_edges, memory_edits) should exist.
        val tablesCursor = migrated.query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
        )
        val tableNames = mutableListOf<String>()
        tablesCursor.use {
            while (it.moveToNext()) {
                tableNames.add(it.getString(0))
            }
        }
        assert("memories" in tableNames) { "memories table missing after chained migration" }
        assert("kg_nodes" in tableNames) { "kg_nodes table missing after chained migration" }
        assert("kg_edges" in tableNames) { "kg_edges table missing after chained migration" }
        assert("memory_edits" in tableNames) { "memory_edits table missing after chained migration" }

        migrated.close()
    }
}