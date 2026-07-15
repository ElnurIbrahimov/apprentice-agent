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
    fun migrate1To2_preservesKgSchema() {
        val db = helper.createDatabase("test-aura-memory.db", 1)
        db.close()

        val migrated = helper.runMigrationsAndValidate(
            "test-aura-memory.db",
            2,
            true,
            MemoryModule.MIGRATION_1_2,
        )

        // Verify kg_nodes columns match the v2 schema.
        val nodeCursor = migrated.query("PRAGMA table_info(kg_nodes)")
        val nodeColumns = mutableSetOf<String>()
        nodeCursor.use {
            while (it.moveToNext()) {
                nodeColumns.add(it.getString(it.getColumnIndexOrThrow("name")))
            }
        }
        assert("createdAt" in nodeColumns) { "kg_nodes should have createdAt after 1→2" }

        // Verify kg_edges columns match the v2 schema.
        val edgeCursor = migrated.query("PRAGMA table_info(kg_edges)")
        val edgeColumns = mutableSetOf<String>()
        edgeCursor.use {
            while (it.moveToNext()) {
                edgeColumns.add(it.getString(it.getColumnIndexOrThrow("name")))
            }
        }
        assert("properties" in edgeColumns) { "kg_edges should have properties after 1→2" }
        assert("confidence" in edgeColumns) { "kg_edges should have confidence after 1→2" }
        assert("createdAt" in edgeColumns) { "kg_edges should have createdAt after 1→2" }

        // Verify all expected indices were created.
        val indexCursor = migrated.query(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'index_kg_%'",
        )
        val indexNames = mutableSetOf<String>()
        indexCursor.use {
            while (it.moveToNext()) {
                indexNames.add(it.getString(0))
            }
        }
        assert("index_kg_nodes_label" in indexNames)
        assert("index_kg_nodes_type" in indexNames)
        assert("index_kg_nodes_label_type" in indexNames)
        assert("index_kg_edges_sourceId" in indexNames)
        assert("index_kg_edges_targetId" in indexNames)
        assert("index_kg_edges_sourceId_targetId_type" in indexNames)

        migrated.close()
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
    fun migrate3To4_preservesRows_andAddsConversationProvenance() {
        val db = helper.createDatabase("test-aura-memory-v3.db", 3)
        db.execSQL(
            """
            INSERT INTO memories (id, content, source, category, importance, embedding, createdAt, accessedAt, accessCount, decayScore, tags, metadata)
            VALUES ('mem-1', 'User likes coffee', 'user', 'preference', 0.8, NULL, 1700000000, 1700000000, 0, 1.0, '', '{}')
            """.trimIndent(),
        )
        db.close()

        val migrated = helper.runMigrationsAndValidate(
            "test-aura-memory-v3.db",
            4,
            true,
            MemoryModule.MIGRATION_3_4,
        )

        migrated.query(
            "SELECT content, sourceConversationId, sourceTurnTimestamp FROM memories WHERE id = 'mem-1'",
        ).use {
            assert(it.moveToFirst())
            assert(it.getString(0) == "User likes coffee")
            assert(it.getString(1).isEmpty())
            assert(it.getLong(2) == 0L)
        }
        migrated.close()
    }

    @Test
    fun migrate5To6_createsCreativeProjectStore() {
        val db = helper.createDatabase("test-aura-memory-v5.db", 5)
        db.close()

        val migrated = helper.runMigrationsAndValidate(
            "test-aura-memory-v5.db",
            6,
            true,
            MemoryModule.MIGRATION_5_6,
        )
        migrated.query("PRAGMA table_info(creative_projects)").use { cursor ->
            val names = mutableSetOf<String>()
            val nameColumn = cursor.getColumnIndex("name")
            while (cursor.moveToNext()) names += cursor.getString(nameColumn)
            assertTrue(names.containsAll(setOf("id", "name", "worldJson", "templateId", "turnCount", "updatedAt")))
        }
        migrated.close()
    }

    @Test
    fun migrate4To5_createsDocumentLibrary() {
        val db = helper.createDatabase("test-aura-memory-v4.db", 4)
        db.close()

        val migrated = helper.runMigrationsAndValidate(
            "test-aura-memory-v4.db",
            5,
            true,
            MemoryModule.MIGRATION_4_5,
        )

        migrated.query("PRAGMA table_info(documents)").use { cursor ->
            val columns = mutableSetOf<String>()
            while (cursor.moveToNext()) {
                columns += cursor.getString(cursor.getColumnIndexOrThrow("name"))
            }
            assert(columns == setOf(
                "id", "name", "mimeType", "sourceUri", "importedAt", "characterCount", "chunkCount",
            )) { "Unexpected document columns: $columns" }
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