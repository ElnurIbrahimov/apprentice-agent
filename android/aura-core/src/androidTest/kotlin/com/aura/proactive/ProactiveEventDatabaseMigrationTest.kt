package com.aura.proactive

import androidx.room.testing.MigrationTestHelper
import androidx.sqlite.db.framework.FrameworkSQLiteOpenHelperFactory
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Rule
import org.junit.Test

class ProactiveEventDatabaseMigrationTest {

    @get:Rule
    val helper = MigrationTestHelper(
        InstrumentationRegistry.getInstrumentation(),
        ProactiveEventDatabase::class.java.canonicalName,
        FrameworkSQLiteOpenHelperFactory(),
    )

    @Test
    fun migrate1To2_preservesEvents_andAddsPayload() {
        val db = helper.createDatabase("test-proactive.db", 1)
        // Insert an event using the v1 schema (no payload column).
        db.execSQL(
            """
            INSERT INTO proactive_events (id, eventType, title, body, timestamp)
            VALUES (1, 'morning_brief', 'Morning brief', 'You have 2 tasks today.', 1700000000)
            """.trimIndent(),
        )
        db.close()

        val migrated = helper.runMigrationsAndValidate(
            "test-proactive.db",
            2,
            true,
            ProactiveEventModule.MIGRATION_1_2,
        )

        // Verify the event survived and payload has the default empty string.
        val cursor = migrated.query("SELECT * FROM proactive_events WHERE id = 1")
        cursor.use {
            assert(it.moveToFirst()) { "Event row should survive migration" }
            assert(it.getString(it.getColumnIndexOrThrow("title")) == "Morning brief")
            val payloadIdx = it.getColumnIndexOrThrow("payload")
            assert(it.getString(payloadIdx) == "") { "payload should default to empty string" }
        }

        migrated.close()
    }

    @Test
    fun migrate2To3_preservesEvents_andAddsTimestampIndex() {
        val db = helper.createDatabase("test-proactive-2-3.db", 2)
        db.execSQL("INSERT INTO proactive_events (id, eventType, title, body, timestamp, payload) VALUES (2, 'test', 'Indexed', '', 1, '{}')")
        db.close()
        val migrated = helper.runMigrationsAndValidate(
            "test-proactive-2-3.db", 3, true, ProactiveEventModule.MIGRATION_2_3,
        )
        migrated.query("SELECT title FROM proactive_events WHERE id = 2").use {
            assert(it.moveToFirst() && it.getString(0) == "Indexed")
        }
        migrated.query("PRAGMA index_list('proactive_events')").use {
            var found = false
            while (it.moveToNext()) found = found || it.getString(it.getColumnIndexOrThrow("name")) == "index_proactive_events_timestamp"
            assert(found)
        }
        migrated.close()
    }
}