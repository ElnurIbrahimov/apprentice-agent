package com.aura.hands

import androidx.room.testing.MigrationTestHelper
import androidx.sqlite.db.framework.FrameworkSQLiteOpenHelperFactory
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Rule
import org.junit.Test

class HandDatabaseMigrationTest {

    @get:Rule
    val helper = MigrationTestHelper(
        InstrumentationRegistry.getInstrumentation(),
        HandDatabase::class.java.canonicalName,
        FrameworkSQLiteOpenHelperFactory(),
    )

    @Test
    fun repositoryTemplatePattern_initializesOnAndroidIcu() {
        // Android ICU is stricter than the host JVM about literal closing braces.
        // Loading the class pins the exact failure mode that once crashed app startup.
        Class.forName(HandRepository::class.java.name)
    }

    @Test
    fun migrate1To2_preservesHands_andAddsAutomationRuntime() {
        val db = helper.createDatabase("test-hands-1-2.db", 1)
        db.execSQL(
            "INSERT INTO hands (id, name, triggerPhrase, steps, enabled, createdAt) " +
                "VALUES ('h1', 'Morning', 'start day', '[]', 1, 123)",
        )
        db.close()

        val migrated = helper.runMigrationsAndValidate(
            "test-hands-1-2.db", 2, true, HandsModule.MIGRATION_1_2,
        )

        migrated.query(
            "SELECT name, variables, conditions, scheduleType, scheduleHour, scheduleMinute, " +
                "scheduleDayOfWeek, updatedAt FROM hands WHERE id = 'h1'",
        ).use {
            assert(it.moveToFirst())
            assert(it.getString(0) == "Morning")
            assert(it.getString(1) == "{}")
            assert(it.getString(2) == "[]")
            assert(it.getString(3) == "none")
            assert(it.getInt(4) == 9)
            assert(it.getInt(5) == 0)
            assert(it.getInt(6) == 1)
            assert(it.getLong(7) == 0L)
        }
        migrated.query("PRAGMA table_info('hand_runs')").use {
            var columns = 0
            while (it.moveToNext()) columns++
            assert(columns == 10)
        }
        migrated.close()
    }
}
